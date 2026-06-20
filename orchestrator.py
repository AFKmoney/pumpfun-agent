"""
orchestrator.py
===============
Main 24/7 orchestrator.

Responsibilities:
1. Initialize all components (adapters, strategies, risk, notifier,
   data providers, persistence, dashboard, plus all the new profit-optimization
   modules: TokenScorer, SmartExitManager, DevTracker, GasOptimizer,
   AntiSandwich, BondingCurveAnalyzer, MigrationDetector, Pyramider).
2. Launch each strategy as an asyncio task; signals land in a shared queue.
3. Consume the queue, route each signal through Executor.
4. Periodically check open positions for SL/TP/max-hold exits (via SmartExitManager).
5. Periodically run migration detection on open positions.
6. Periodically run pyramiding evaluation on profitable positions.
7. Periodically poll gas prices (Solana + EVM).
8. Watch kill switch (file sentinel + Telegram /kill + daily loss cap).
9. Graceful shutdown on SIGINT/SIGTERM.
"""
from __future__ import annotations

import asyncio
import signal
import time
import sys
from typing import Optional

from chains.chain_factory import build_adapters
from chains.base_chain import BaseChainAdapter
from execution.executor import Executor
from risk.backtester import Backtester
from risk.pyramiding import Pyramider
from risk.risk_manager import RiskManager
from risk.smart_exits import SmartExitManager
from strategies.anti_rugpull import AntiRugpullStrategy
from strategies.base_strategy import BaseStrategy, Signal, SignalType
from strategies.copy_trade import CopyTradeStrategy
from strategies.grid_scalping import GridScalpingStrategy
from strategies.momentum import MomentumStrategy
from strategies.sniping import SnipingStrategy
from analysis.alpha_signal import AlphaSignalGenerator
from analysis.lifecycle import LifecycleAnalyzer
from analysis.liquidity_depth import LiquidityDepthAnalyzer
from analysis.mev_detector import MevDetector
from analysis.order_flow import OrderFlowAnalyzer
from analysis.sentiment import SentimentAnalyzer
from analysis.social_graph import SocialGraphAnalyzer
from analysis.soft_rug import SoftRugDetector
from utils.analytics_pipeline import AnalyticsPipeline
from utils.anti_sandwich import AntiSandwich
from utils.attribution import AutoTuner
from utils.bonding_curve import BondingCurveAnalyzer
from utils.config_hot_reload import ConfigHotReloader
from utils.config_loader import Config
from utils.data_providers import close_all_providers, get_providers
from utils.dev_tracker import DevTracker
from utils.gas_optimizer import GasOptimizer
from utils.kill_switch import KillSwitch
from utils.logger import setup_logger
from utils.migration_detector import MigrationDetector
from utils.notifier import TelegramNotifier
from utils.persistence import Persistence
from utils.token_scorer import TokenScorer
from utils.wallet_manager import WalletManager

log = setup_logger("orchestrator")


class Orchestrator:
    _instance: Optional["Orchestrator"] = None

    def __init__(self) -> None:
        self.cfg = Config.get()
        self.signal_queue: asyncio.Queue[Signal] = asyncio.Queue()
        self.adapters: dict[str, BaseChainAdapter] = {}
        self.strategies: list[BaseStrategy] = []
        self.executors: dict[str, Executor] = {}
        self.risk = RiskManager()
        self.notifier = TelegramNotifier()
        self.wallet = WalletManager()
        self.anti_rugs: dict[str, AntiRugpullStrategy] = {}
        self.backtester = Backtester()

        # New profit-optimization modules
        self.token_scorer = TokenScorer()
        self.smart_exits = SmartExitManager()
        self.dev_tracker = DevTracker()
        self.gas_optimizer = GasOptimizer()
        self.anti_sandwich = AntiSandwich()
        self.bonding = BondingCurveAnalyzer()
        self.migration_detector = MigrationDetector()
        self.pyramider = Pyramider(self.token_scorer)
        self.hot_reloader = ConfigHotReloader(orchestrator=self)

        # Advanced analyzers
        self.order_flow = OrderFlowAnalyzer()
        self.social_graph = SocialGraphAnalyzer()
        self.mev_detector = MevDetector()
        self.lifecycle = LifecycleAnalyzer()
        self.liquidity_depth = LiquidityDepthAnalyzer()
        self.sentiment = SentimentAnalyzer()
        self.alpha_signal = AlphaSignalGenerator(
            token_scorer=self.token_scorer,
            order_flow=self.order_flow,
            social_graph=self.social_graph,
            lifecycle=self.lifecycle,
            liquidity_depth=self.liquidity_depth,
            sentiment=self.sentiment,
            bonding=self.bonding,
            mev_detector=self.mev_detector,
        )

        # Live data ingestion pipeline that feeds the analyzers (order_flow,
        # lifecycle, social_graph, sentiment) with real-time + periodic data.
        # Without it, AlphaSignal collapses to neutral-50 fallbacks.
        self.analytics_pipeline = AnalyticsPipeline(
            order_flow=self.order_flow,
            lifecycle=self.lifecycle,
            social_graph=self.social_graph,
            sentiment=self.sentiment,
            bonding=self.bonding,
            risk_positions_ref=lambda: self.risk.positions,
        )

        # Weekly AlphaSignal weight re-tuner (closed-loop learning: which of
        # our own signal components actually predict profitable trades).
        self.autotuner = AutoTuner(self.alpha_signal)

        # Soft-rug early-warning model (exits before a gradual dump completes).
        self.soft_rug = SoftRugDetector()

        self._tasks: list[asyncio.Task] = []
        self._stop_event = asyncio.Event()
        self.state = "INIT"
        self._dashboard_task: Optional[asyncio.Task] = None
        Orchestrator._instance = self

    # ------------------------------------------------------------------
    @classmethod
    def get_status(cls) -> dict:
        if cls._instance is None:
            return {"state": "NOT_STARTED"}
        return {
            "state": cls._instance.state,
            "open_positions": len(cls._instance.risk.positions),
            "daily_pnl_pct": cls._instance.risk.daily_pnl_pct,
            "kill_switch": KillSwitch.is_triggered(),
        }

    # ------------------------------------------------------------------
    async def start(self) -> None:
        log.info("orchestrator.starting", env=self.cfg["environment"], mode=self.cfg["trading"]["mode"])
        self.state = "STARTING"

        # Initialize data providers (DexScreener, Helius, Birdeye, TokenSniffer)
        get_providers()

        # Initialize persistence (creates SQLite DB + tables)
        Persistence.get()

        # Notifier first so we can receive /kill during init
        await self.notifier.start()

        # Adapters
        self.adapters = await build_adapters()
        for chain, adapter in self.adapters.items():
            self.anti_rugs[chain] = AntiRugpullStrategy(adapter)
            self.executors[chain] = Executor(
                adapter=adapter, risk_manager=self.risk,
                notifier=self.notifier, anti_rug=self.anti_rugs[chain],
                token_scorer=self.token_scorer,
                smart_exits=self.smart_exits,
                dev_tracker=self.dev_tracker,
                gas_optimizer=self.gas_optimizer,
                anti_sandwich=self.anti_sandwich,
                alpha_signal=self.alpha_signal,
                soft_rug=self.soft_rug,
            )

        # Register migration detector callback -> notify + extend hold time
        async def _on_migration(event):
            await self.notifier.notify_error(
                f"🚀 MIGRATION DETECTED\nToken: {event.token}\n"
                f"Pump: {event.pump_pct:.0f}%\nLiq: ${event.liq_after:,.0f}"
            )
            log.warning("orchestrator.migration_event", token=event.token)
        self.migration_detector.register_callback(_on_migration)

        # Register dev-sell callback -> immediate exit
        async def _on_dev_sell(chain, token):
            log.critical("orchestrator.dev_sold_trigger", token=token)
            # Inject an urgent SELL signal
            await self.signal_queue.put(Signal(
                strategy="dev_tracker", chain=chain, token_address=token,
                signal_type=SignalType.SELL, suggested_size_pct=1.0, confidence=1.0,
                reason="DEV SOLD — instant exit",
            ))
        self.dev_tracker.register_callback(_on_dev_sell)

        # Strategies
        scfg = self.cfg["strategies"]
        for chain, adapter in self.adapters.items():
            if scfg["sniping"]["enabled"]:
                self.strategies.append(
                    SnipingStrategy(adapter, self.signal_queue, anti_rug=self.anti_rugs[chain])
                )
            if scfg["copy_trade"]["enabled"]:
                self.strategies.append(CopyTradeStrategy(
                    adapter, self.signal_queue, social_graph=self.social_graph,
                ))
            if scfg["momentum"]["enabled"]:
                self.strategies.append(MomentumStrategy(adapter, self.signal_queue))
            if scfg["grid_scalping"]["enabled"]:
                self.strategies.append(GridScalpingStrategy(adapter, self.signal_queue))

        # Backtest gate (optional)
        bt_cfg = self.cfg.get_nested("risk", "backtesting", default={})
        if bt_cfg.get("enabled", False):
            await self._run_backtest_gate()

        # Dashboard
        mon_cfg = self.cfg.get_nested("monitoring", "dashboard", default={})
        if mon_cfg.get("enabled", False):
            self._start_dashboard(mon_cfg.get("port", 8080))

        # Launch tasks
        self._tasks.append(asyncio.create_task(self._signal_consumer()))
        self._tasks.append(asyncio.create_task(self._smart_exit_monitor()))
        self._tasks.append(asyncio.create_task(self._migration_monitor()))
        self._tasks.append(asyncio.create_task(self._pyramiding_monitor()))
        self._tasks.append(asyncio.create_task(self._gas_poll_loop()))
        self._tasks.append(asyncio.create_task(KillSwitch().watch_file()))
        for s in self.strategies:
            self._tasks.append(asyncio.create_task(s.run()))

        # Start the analytics ingestion pipeline (feeds order_flow / lifecycle /
        # social_graph / sentiment so AlphaSignal has real data instead of
        # neutral-50 fallbacks).
        await self.analytics_pipeline.start()
        # Start the weekly weight re-tuner (closed-loop learning).
        await self.autotuner.start()
        # Start the nightly soft-rug model retrain.
        await self.soft_rug.start_nightly_retrain()
        # Start the soft-rug position monitor (exits high-risk positions early).
        self._tasks.append(asyncio.create_task(self._soft_rug_monitor()))

        self.state = "RUNNING"
        log.info("orchestrator.running",
                 strategies=[s.name for s in self.strategies],
                 smart_exits=True, dev_tracker=True, gas_optimizer=True,
                 migration_detector=True, pyramider=True,
                 analytics_pipeline=True)

        stop_t = asyncio.create_task(self._stop_event.wait())
        kill_t = asyncio.create_task(self._wait_kill())
        await asyncio.wait({stop_t, kill_t}, return_when=asyncio.FIRST_COMPLETED)

        await self._shutdown()

    async def _run_backtest_gate(self) -> None:
        log.info("orchestrator.backtest_gate_running")
        sample_tokens = ["EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"]
        passed: list[BaseStrategy] = []
        for s in self.strategies:
            try:
                result = await self.backtester.run_for_strategy(s, sample_tokens)
                if self.backtester.passes_gate(result):
                    passed.append(s)
                else:
                    log.warning("orchestrator.strategy_disabled_by_backtest", strategy=s.name)
            except Exception as e:
                log.error("orchestrator.backtest_failed", strategy=s.name, error=str(e))
                passed.append(s)
        self.strategies = passed

    def _start_dashboard(self, port: int) -> None:
        try:
            import uvicorn
            from utils.dashboard import build_app
            app = build_app()
            config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
            server = uvicorn.Server(config)

            async def _serve() -> None:
                try:
                    await server.serve()
                except asyncio.CancelledError:
                    pass

            self._dashboard_task = asyncio.create_task(_serve())
            log.info("orchestrator.dashboard_started", port=port)
        except Exception as e:
            log.warning("orchestrator.dashboard_failed", error=str(e))

    async def _wait_kill(self) -> None:
        while not KillSwitch.is_triggered():
            await asyncio.sleep(2)
        if "loss" in (KillSwitch.reason() or ""):
            await self.notifier.notify_daily_loss_cap()

    # ------------------------------------------------------------------
    async def _signal_consumer(self) -> None:
        while not self._stop_event.is_set():
            try:
                sig = await asyncio.wait_for(self.signal_queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            chain = sig.chain
            executor = self.executors.get(chain)
            if not executor:
                log.warning("orchestrator.no_executor_for_chain", chain=chain)
                continue
            allocated = self.cfg["chains"].get(chain, {}).get(
                "allocated_capital_sol" if chain == "solana" else "allocated_capital_eth", 0
            )
            try:
                await executor.execute(sig, allocated)
            except Exception as e:
                log.error("orchestrator.execute_exception", error=str(e))

    async def _smart_exit_monitor(self) -> None:
        """Periodically run smart exits (trailing, ladder, break-even, dev-sell)."""
        async def price_provider(chain: str, token: str) -> float:
            adapter = self.adapters.get(chain)
            if not adapter:
                return 0.0
            return await adapter.get_price(token)

        async def dev_sell_detector(chain: str, token: str) -> bool:
            # Dev tracker fires async callbacks; this is a fallback sync check.
            return self.dev_tracker.is_dev_selling(chain, token)

        while not self._stop_event.is_set():
            try:
                if self.risk.positions:
                    exit_signals = await self.smart_exits.check_exits(
                        self.risk.positions, price_provider, dev_sell_detector
                    )
                    if exit_signals:
                        log.info("orchestrator.exit_signals", count=len(exit_signals))
                        # Route to executor
                        chain = next(iter(self.adapters.keys())) if self.adapters else None
                        if chain:
                            await self.executors[chain].process_exit_signals(exit_signals)
            except Exception as e:
                log.error("orchestrator.smart_exit_error", error=str(e))
            await asyncio.sleep(10)

    async def _migration_monitor(self) -> None:
        """Check open positions for pump.fun → Raydium migration events."""
        while not self._stop_event.is_set():
            try:
                for key, pos in list(self.risk.positions.items()):
                    if pos.chain == "solana":
                        await self.migration_detector.check_position(pos.chain, pos.token)
            except Exception as e:
                log.error("orchestrator.migration_monitor_error", error=str(e))
            await asyncio.sleep(30)

    async def _pyramiding_monitor(self) -> None:
        """Evaluate pyramiding on profitable positions."""
        async def price_provider(chain, token):
            adapter = self.adapters.get(chain)
            return await adapter.get_price(token) if adapter else 0.0

        while not self._stop_event.is_set():
            try:
                for key, pos in list(self.risk.positions.items()):
                    price = await price_provider(pos.chain, pos.token)
                    if price <= 0:
                        continue
                    signal = await self.pyramider.evaluate(pos, price)
                    if signal:
                        # Emit BUY signal for the pyramid
                        await self.signal_queue.put(Signal(
                            strategy="pyramiding",
                            chain=pos.chain,
                            token_address=pos.token,
                            signal_type=SignalType.BUY,
                            suggested_size_pct=signal.add_size_pct_of_original,
                            confidence=0.7,
                            reason=signal.reason,
                            stop_loss_pct=10.0,
                            take_profit_pct=100.0,
                        ))
            except Exception as e:
                log.error("orchestrator.pyramiding_error", error=str(e))
            await asyncio.sleep(60)

    async def _gas_poll_loop(self) -> None:
        """Poll Solana + EVM gas prices every 30s."""
        while not self._stop_event.is_set():
            try:
                await self.gas_optimizer.poll_solana()
                for chain in ("base", "ethereum"):
                    if chain in self.adapters:
                        await self.gas_optimizer.poll_evm(chain)
            except Exception as e:
                log.error("orchestrator.gas_poll_error", error=str(e))
            await asyncio.sleep(30)

    async def _soft_rug_monitor(self) -> None:
        """
        Periodically score open positions for soft-rug risk. If the model
        flags a position as high-risk, emit an urgent SELL to exit before the
        gradual dump completes.
        """
        from analysis.soft_rug import RugFeatures
        from utils.data_providers import get_providers
        interval = self.cfg.get_nested("analysis", "soft_rug", default={}).get(
            "check_interval_sec", 30)
        while not self._stop_event.is_set():
            try:
                if self.risk.positions:
                    providers = get_providers()
                    for key, pos in list(self.risk.positions.items()):
                        if pos.chain != "solana":
                            continue
                        try:
                            helius = providers["helius"]
                            dex = providers["dexscreener"]
                            holders = await helius.get_token_holders(pos.token)
                            holder_list = holders if isinstance(holders, list) else []
                            top3_pct = 0.0
                            if holder_list:
                                # holder entries carry an 'amount' or 'uiAmount'
                                amounts = []
                                for h in holder_list[:20]:
                                    a = h.get("amount") or h.get("uiAmount") or 0
                                    try: amounts.append(float(a))
                                    except (TypeError, ValueError): pass
                                total = sum(amounts)
                                top3_pct = (sum(sorted(amounts, reverse=True)[:3]) / total * 100) if total > 0 else 0
                            liq_usd = await dex.get_liquidity_usd(pos.token) or 0.0
                            bc_state = await self.bonding.fetch_state(pos.token)
                            # Order-flow snapshot for sell velocity / b/s ratio
                            of = self.order_flow.snapshot(pos.token, 30)
                            age_sec = max(1.0, time.time() - pos.opened_at)
                            features = RugFeatures(
                                initial_holder_count=len(holder_list),
                                top3_concentration_pct=top3_pct,
                                sell_velocity_30s=(of.sell_count * (60.0/30.0)) if of else 0.0,
                                buy_sell_ratio_30s=(of.buy_count / max(1, of.sell_count)) if of else 1.0,
                                liquidity_usd=liq_usd,
                                bonding_curve_completion_pct=(bc_state.completion_pct if bc_state else 0.0),
                                age_seconds=age_sec,
                            )
                            pred = self.soft_rug.predict_proba(features)
                            if pred.is_high_risk:
                                log.warning("orchestrator.soft_rug_exit",
                                            token=pos.token, p=pred.rug_probability,
                                            factors=pred.top_risk_factors)
                                await self.notifier.notify_error(
                                    f"🚨 SOFT-RUG EXIT\nToken: {pos.token}\n"
                                    f"P(rug)={pred.rug_probability:.0%}\n"
                                    f"Factors: {', '.join(pred.top_risk_factors)}"
                                )
                                await self.signal_queue.put(Signal(
                                    strategy="soft_rug", chain=pos.chain, token_address=pos.token,
                                    signal_type=SignalType.SELL, suggested_size_pct=1.0,
                                    confidence=pred.rug_probability,
                                    reason=f"SOFT-RUG EXIT (P={pred.rug_probability:.0%})",
                                ))
                        except Exception as e:
                            log.debug("orchestrator.soft_rug_check_error",
                                      token=pos.token, error=str(e))
            except Exception as e:
                log.error("orchestrator.soft_rug_monitor_error", error=str(e))
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    async def _shutdown(self) -> None:
        log.info("orchestrator.shutdown_starting")
        self.state = "STOPPING"
        if self._dashboard_task:
            self._dashboard_task.cancel()
        await self.analytics_pipeline.stop()
        await self.autotuner.stop()
        await self.soft_rug.stop()
        await self.dev_tracker.stop_all()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        for adapter in self.adapters.values():
            try:
                await adapter.close()
            except Exception:
                pass
        await close_all_providers()
        try:
            await self.token_scorer.close()
        except Exception:
            pass
        try:
            await self.gas_optimizer.close()
        except Exception:
            pass
        await self.notifier.stop()
        try:
            Persistence.get().close()
        except Exception:
            pass
        self.state = "STOPPED"
        log.info("orchestrator.shutdown_complete")

    def request_stop(self) -> None:
        self._stop_event.set()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
async def _main() -> None:
    orch = Orchestrator()

    def _signal_handler(sig, frame):
        log.info("signal.received", sig=sig)
        orch.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, RuntimeError):
            pass

    await orch.start()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        sys.exit(0)
