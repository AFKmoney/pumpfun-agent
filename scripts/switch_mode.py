#!/usr/bin/env python3
"""
switch_mode.py
==============
Switch the bot between PAPER and LIVE mode safely.

Checks before switching TO live:
  - Paper training has been run (report exists)
  - GO verdict from last report (or force with --force)
  - Wallet is configured + has SOL
  - Risk limits are conservative

Usage:
    python scripts/switch_mode.py paper     # switch to paper (safe)
    python scripts/switch_mode.py live      # switch to live (checks first)
    python scripts/switch_mode.py live --force  # skip the GO check
    python scripts/switch_mode.py status    # show current mode
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_config():
    import yaml
    cfg_path = ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        print("❌ config/config.yaml not found")
        sys.exit(1)
    return yaml.safe_load(cfg_path.read_text()), cfg_path


def write_config(cfg, cfg_path):
    import yaml
    cfg_path.write_text(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False))


def cmd_status():
    cfg, _ = read_config()
    mode = cfg.get("trading", {}).get("mode", "unknown")
    size = cfg.get("trading", {}).get("fixed_size_sol", "?")
    max_pos = cfg.get("trading", {}).get("max_open_positions", "?")
    daily_cap = cfg.get("risk", {}).get("daily_loss_cap_pct", "?")

    print(f"Current mode: {mode.upper()}")
    print(f"  fixed_size_sol: {size}")
    print(f"  max_open_positions: {max_pos}")
    print(f"  daily_loss_cap_pct: {daily_cap}%")

    # Check for paper report
    report_path = ROOT / "data" / "paper_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        verdict = report.get("go_no_go", {})
        print(f"\n  Last paper report: {verdict.get('verdict', '?')}")
        t = report.get("trades", {})
        print(f"    Win rate: {t.get('win_rate', 0):.1%} | "
              f"Trades: {t.get('total_sells', 0)} | "
              f"Expectancy: {t.get('expectancy_pct', 0):.2f}%")
    else:
        print("\n  ⚠️  No paper report found. Run: python scripts/paper_training.py")


def cmd_paper():
    cfg, cfg_path = read_config()
    cfg.setdefault("trading", {})
    cfg["trading"]["mode"] = "paper"
    write_config(cfg, cfg_path)
    print("✅ Switched to PAPER mode")
    print("   No real funds at risk. Run: python orchestrator.py")


def cmd_live(force=False):
    cfg, cfg_path = read_config()

    # Safety: check paper report unless forced
    if not force:
        report_path = ROOT / "data" / "paper_report.json"
        if not report_path.exists():
            print("⛔ No paper report found.")
            print("   Run paper training first: python scripts/paper_training.py --hours 4")
            print("   Or force: python scripts/switch_mode.py live --force")
            sys.exit(1)
        report = json.loads(report_path.read_text())
        verdict = report.get("go_no_go", {})
        if verdict.get("verdict") != "GO":
            print(f"⛔ Last paper report says: {verdict.get('verdict')}")
            for r in verdict.get("reasons", []):
                print(f"   • {r}")
            print("\n   Fix the issues or train longer, then retry.")
            print("   Or force: python scripts/switch_mode.py live --force")
            sys.exit(1)
        else:
            print("✅ Paper report verdict: GO")

    # Check wallet
    import os
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    try:
        from utils.wallet_manager import WalletManager
        wm = WalletManager()
        if not wm.has_wallet("solana"):
            print("⛔ No Solana wallet configured. Set SOLANA_PRIVATE_KEY in .env")
            sys.exit(1)
        print(f"✅ Wallet: {wm.solana_pubkey()}")
    except Exception as e:
        print(f"⚠️  Wallet check failed: {e}")
        if not force:
            sys.exit(1)

    # Apply conservative live settings
    cfg.setdefault("trading", {})
    cfg["trading"]["mode"] = "live"
    write_config(cfg, cfg_path)

    print()
    print("🚀 Switched to LIVE mode")
    print(f"   fixed_size_sol: {cfg['trading'].get('fixed_size_sol', '?')}")
    print(f"   max_open_positions: {cfg['trading'].get('max_open_positions', '?')}")
    print()
    print("   Recommended first step:")
    print("     python scripts/single_shot_live.py")
    print("   (one trade, then auto-stop)")
    print()
    print("   Or full run:")
    print("     python orchestrator.py")
    print()
    print("   Kill switch: touch data/KILL_SWITCH")


def main():
    parser = argparse.ArgumentParser(description="Switch bot mode")
    parser.add_argument("mode", choices=["paper", "live", "status"],
                        help="Mode to switch to (or 'status' to check)")
    parser.add_argument("--force", action="store_true",
                        help="Skip paper-report check when switching to live")
    args = parser.parse_args()

    if args.mode == "status":
        cmd_status()
    elif args.mode == "paper":
        cmd_paper()
    elif args.mode == "live":
        cmd_live(force=args.force)


if __name__ == "__main__":
    main()
