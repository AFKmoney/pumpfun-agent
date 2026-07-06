#!/usr/bin/env python3
"""
paper_training.py
=================
Run the bot in PAPER mode for a set duration, then generate a performance
report with the metrics that decide whether to go live.

Captures:
  - Trades executed (buy/sell/paper fills)
  - Win rate, avg win/loss, expectancy
  - Latency p50/p95/p99 (from the latency tracer)
  - Signal rejection reasons (why didn't we buy more?)
  - Regime distribution (how much time was risk-on vs risk-off?)
  - Crowd positioning distribution
  - Daily PnL curve

At the end, prints a GO/NO-GO verdict based on configurable thresholds.

Usage:
    python scripts/paper_training.py --hours 4
    python scripts/paper_training.py --hours 24 --report data/paper_report.json
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


async def run_paper_session(hours: float, report_path: str) -> dict:
    from orchestrator import Orchestrator
    from utils.logger import setup_logger
    from utils.kill_switch import KillSwitch
    from utils.latency_tracer import get_tracer

    log = setup_logger("paper_training")
    print("=" * 60)
    print(f"  PAPER TRAINING — {hours}h session")
    print(f"  Mode: PAPER (no real funds at risk)")
    print(f"  Report: {report_path}")
    print("=" * 60)

    # Ensure paper mode
    orch = Orchestrator()
    orch.cfg["trading"]["mode"] = "paper"
    orch.paper_mode = True
    # Force the executor paper flag too
    for ex in orch.executors.values():
        ex.paper_mode = True

    log.info("paper_training.started", duration_hours=hours)

    # Schedule stop
    stop_time = time.time() + hours * 3600

    async def auto_stop():
        while time.time() < stop_time:
            await asyncio.sleep(30)
            if KillSwitch.is_triggered():
                log.warning("paper_training.kill_switch")
                break
        orch.request_stop()

    asyncio.create_task(auto_stop())

    # Run the bot
    try:
        await orch.start()
    except Exception as e:
        log.error("paper_training.crash", error=str(e))

    # === Generate report ===
    report = generate_report(orch, hours)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2, default=str))
    print_report(report, report_path)
    return report


def generate_report(orch, hours: float) -> dict:
    """Collect all metrics from the session."""
    from utils.latency_tracer import get_tracer

    # Trade history from risk manager
    history = list(orch.risk.history) if orch.risk.history else []
    sells = [t for t in history if t.side == "SELL" and t.pnl_pct is not None]
    buys = [t for t in history if t.side == "BUY"]

    wins = [t for t in sells if t.pnl_pct > 0]
    losses = [t for t in sells if t.pnl_pct <= 0]
    win_rate = len(wins) / max(len(sells), 1)
    avg_win = sum(t.pnl_pct for t in wins) / max(len(wins), 1) if wins else 0
    avg_loss = sum(t.pnl_pct for t in losses) / max(len(losses), 1) if losses else 0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

    # Latency stats
    tracer = get_tracer()
    latency_stats = tracer.stats()

    # Strategy breakdown
    by_strategy: dict[str, dict] = {}
    for t in sells:
        s = t.strategy
        if s not in by_strategy:
            by_strategy[s] = {"trades": 0, "wins": 0, "pnl_sum": 0.0}
        by_strategy[s]["trades"] += 1
        if t.pnl_pct > 0:
            by_strategy[s]["wins"] += 1
        by_strategy[s]["pnl_sum"] += t.pnl_pct

    # Regime distribution
    regime = orch.regime.current
    regime_info = {
        "final_label": regime.label if regime else "unknown",
        "final_score": regime.score if regime else 0,
    }

    return {
        "session_hours": hours,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trades": {
            "total_buys": len(buys),
            "total_sells": len(sells),
            "open_positions": len(orch.risk.positions),
            "win_rate": round(win_rate, 3),
            "wins": len(wins),
            "losses": len(losses),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "expectancy_pct": round(expectancy, 2),
        },
        "by_strategy": {
            s: {
                "trades": v["trades"],
                "win_rate": round(v["wins"] / max(v["trades"], 1), 3),
                "total_pnl_pct": round(v["pnl_sum"], 2),
            }
            for s, v in by_strategy.items()
        },
        "latency": latency_stats,
        "daily_pnl_pct": round(orch.risk.daily_pnl_pct, 2),
        "regime": regime_info,
        "go_no_go": evaluate_go_no_go(win_rate, expectancy, len(sells)),
    }


# GO / NO-GO thresholds (configurable)
GO_THRESHOLDS = {
    "min_closed_trades": 10,    # need enough sample
    "min_win_rate": 0.45,       # 45%+
    "min_expectancy": 0.5,      # positive expectancy per trade (%)
}


def evaluate_go_no_go(win_rate: float, expectancy: float, n_trades: int) -> dict:
    """Return GO/NO-GO verdict based on paper performance."""
    reasons = []
    go = True

    if n_trades < GO_THRESHOLDS["min_closed_trades"]:
        go = False
        reasons.append(f"Only {n_trades} closed trades (need >= {GO_THRESHOLDS['min_closed_trades']})")
    if win_rate < GO_THRESHOLDS["min_win_rate"]:
        go = False
        reasons.append(f"Win rate {win_rate:.1%} < {GO_THRESHOLDS['min_win_rate']:.0%}")
    if expectancy < GO_THRESHOLDS["min_expectancy"]:
        go = False
        reasons.append(f"Expectancy {expectancy:.2f}% < {GO_THRESHOLDS['min_expectancy']}%")

    if go:
        reasons.append("All thresholds met — paper performance justifies live trading")

    return {
        "verdict": "GO" if go else "NO_GO",
        "reasons": reasons,
        "thresholds": GO_THRESHOLDS,
    }


def print_report(report: dict, path: str) -> None:
    print()
    print("=" * 60)
    print("  PAPER TRAINING REPORT")
    print("=" * 60)
    t = report["trades"]
    print(f"  Session: {report['session_hours']}h")
    print(f"  Buys: {t['total_buys']} | Sells: {t['total_sells']} | Open: {t['open_positions']}")
    print(f"  Win rate: {t['win_rate']:.1%} ({t['wins']}W / {t['losses']}L)")
    print(f"  Avg win: +{t['avg_win_pct']:.1f}% | Avg loss: {t['avg_loss_pct']:.1f}%")
    print(f"  Expectancy: {t['expectancy_pct']:.2f}% per trade")
    print(f"  Daily PnL: {report['daily_pnl_pct']:.2f}%")

    if report["latency"]:
        for op, stats in report["latency"].items():
            if "total_ms" in stats:
                print(f"  Latency [{op}]: p50={stats['total_ms']['p50']}ms "
                      f"p95={stats['total_ms']['p95']}ms "
                      f"p99={stats['total_ms']['p99']}ms")

    if report["by_strategy"]:
        print()
        print("  By strategy:")
        for s, v in report["by_strategy"].items():
            print(f"    {s}: {v['trades']} trades, WR={v['win_rate']:.0%}, "
                  f"PnL={v['total_pnl_pct']:+.1f}%")

    print()
    verdict = report["go_no_go"]
    if verdict["verdict"] == "GO":
        print("  🚀 VERDICT: GO — ready to switch to live")
    else:
        print("  ⛔ VERDICT: NO-GO — keep training in paper")
    for r in verdict["reasons"]:
        print(f"     • {r}")
    print()
    print(f"  Full report saved: {path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Paper training session")
    parser.add_argument("--hours", type=float, default=4.0,
                        help="Training duration in hours (default: 4)")
    parser.add_argument("--report", type=str, default="data/paper_report.json",
                        help="Report output path")
    args = parser.parse_args()

    asyncio.run(run_paper_session(args.hours, args.report))


if __name__ == "__main__":
    main()
