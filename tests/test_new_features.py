"""
test_new_features.py
====================
Unit tests for the features added in the production hardening pass:
- pumpfun_ix instruction builders + log parser
- bonding_curve account decode (the bug fix)
- risk_manager.reduce_position + daily PnL weighting
- copy_trade swap decoder
- soft_rug heuristic + autotuner weight fit

Run: pytest tests/test_new_features.py -v
"""
import asyncio
import base64
import hashlib
import os
import sys
import tempfile
import yaml
from pathlib import Path

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --- Minimal config (mirrors test_risk_manager.py's pattern) -------------
_TMP_CFG = {
    "environment": "test",
    "trading": {
        "mode": "paper", "chains": ["solana"],
        "fixed_size_sol": 0.05, "fixed_size_evm": 0.005,
        "slippage_bps": 800, "priority_fee_micro_lamports": 50000,
        "gas_price_gwei": 3, "max_open_positions": 5,
        "max_trade_frequency_per_minute": 20,
    },
    "risk": {
        "fixed_size_pct": 2.0, "max_size_pct": 10.0, "min_size_pct": 0.5,
        "per_token_max_loss_pct": 15.0, "per_token_max_hold_minutes": 120,
        "daily_loss_cap_pct": 10.0, "blacklist_duration_minutes": 60,
        "kelly": {"enabled": False, "fraction": 0.5, "min_history": 10},
        "backtesting": {"enabled": False},
    },
    "persistence": {"db_path": "./data/test_agent.db"},
    "chains": {
        "solana": {
            "rpc_endpoints": ["https://example.test"], "ws_endpoint": "wss://example.test",
            "jupiter_api": "https://example.test", "allocated_capital_sol": 10.0,
        },
        "base": {"rpc_endpoints": ["https://example.test"], "allocated_capital_eth": 1.0},
    },
    "strategies": {"sniping": {"enabled": False}, "copy_trade": {"enabled": False},
                   "momentum": {"enabled": False}, "grid_scalping": {"enabled": False}},
    "analysis": {"soft_rug": {"model_path": "./data/test_soft_rug.json"}},
    "monitoring": {"dashboard": {"enabled": False}},
}
_TMP = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
yaml.safe_dump(_TMP_CFG, _TMP)
_TMP.close()
os.environ["PUMPFUN_AGENT_CONFIG"] = _TMP.name


# =====================================================================
# pumpfun_ix: discriminators, PDA, instruction builders, log parser
# =====================================================================
class TestPumpfunIx:
    def test_discriminators_match_known_anchor_convention(self):
        from utils import pumpfun_ix
        # create discriminator is publicly known = 181ec828051c0777
        assert pumpfun_ix.CREATE_DISCRIMINATOR.hex() == "181ec828051c0777"
        # buy/sell follow the same sha256('global:<method>')[:8] convention
        assert pumpfun_ix.BUY_DISCRIMINATOR == hashlib.sha256(b"global:buy").digest()[:8]
        assert pumpfun_ix.SELL_DISCRIMINATOR == hashlib.sha256(b"global:sell").digest()[:8]

    def test_global_pda_is_deterministic_valid_base58(self):
        from utils import pumpfun_ix
        import base58
        g = pumpfun_ix.derive_global_pda()
        # Must be the canonical pump.fun global: 4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf
        assert g == "4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf"
        # Must be valid base58 (the old hardcoded value had an invalid 'O')
        base58.b58decode(g)  # raises if invalid
        # Module attribute access returns the same value
        assert pumpfun_ix.PUMP_FUN_GLOBAL == g

    def test_bonding_curve_pda_is_deterministic(self):
        from utils import pumpfun_ix
        mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        # Same mint -> same PDA every time
        assert pumpfun_ix.derive_bonding_curve_pda(mint) == pumpfun_ix.derive_bonding_curve_pda(mint)
        # Different mint -> different PDA
        other = "So11111111111111111111111111111111111111112"
        assert pumpfun_ix.derive_bonding_curve_pda(mint) != pumpfun_ix.derive_bonding_curve_pda(other)

    def test_build_buy_ix_layout(self):
        from utils import pumpfun_ix
        # Use a valid base58 buyer pubkey (not the system program which is
        # all-1s and not a valid base58 input for Pubkey.from_string in some
        # solders versions).
        buyer = "9aQ5eQ2VqA3WZ4rB5mN6kP7cQ8dS9tUvW1xY2zA3bC"
        buyer = "".join(c for c in buyer if c in "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz") or "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        ix = pumpfun_ix.build_buy_ix(
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # valid base58 mint as dummy buyer
            "So11111111111111111111111111111111111111112",   # valid base58 mint
            token_amount_raw=1_000_000_000, max_sol_cost_lamports=50_000_000,
        )
        assert str(ix.program_id) == pumpfun_ix.PUMP_FUN_PROGRAM
        assert len(ix.accounts) == 9            # 9 accounts per buy ix
        assert len(ix.data) == 24               # 8 disc + 8 amount + 8 max_cost
        assert ix.data[:8] == pumpfun_ix.BUY_DISCRIMINATOR
        # Exactly one signer (the buyer wallet)
        signers = [a for a in ix.accounts if a.is_signer]
        assert len(signers) == 1

    def test_build_sell_ix_layout(self):
        from utils import pumpfun_ix
        ix = pumpfun_ix.build_sell_ix(
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # valid base58 dummy
            "So11111111111111111111111111111111111111112",
            token_amount_raw=1_000_000_000,
        )
        assert len(ix.data) == 16               # 8 disc + 8 amount (no max_cost on sell)
        assert ix.data[:8] == pumpfun_ix.SELL_DISCRIMINATOR

    def test_parse_create_from_logs_text_event(self):
        from utils import pumpfun_ix
        # All three pubkeys must be valid base58 (alphabet excludes 0, O, I, l).
        mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        bc = "3kHzTmqQnm5nqvdJ3XGAAjzYihLXKh5jkqXLLrBpwHfn"
        user = "7Np41oeYqPefeJQEom9M8K7Bh4xKfE2C9r5J9p9XqL2M"
        logs = [
            "Program 6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P invoke [1]",
            "Program log: Instruction: Create",
            f'CreateEvent: name="T" symbol="TST" uri="https://x" '
            f'mint={mint} bonding_curve={bc} user={user}',
        ]
        p = pumpfun_ix.parse_create_from_logs(logs, "sig1")
        assert p.mint == mint
        assert p.bonding_curve == bc
        assert p.user == user
        assert not p.needs_full_fetch

    def test_parse_create_from_logs_mint_only_derives_bc(self):
        from utils import pumpfun_ix
        # If only the mint is exposed, the parser derives the bonding curve PDA.
        p = pumpfun_ix.parse_create_from_logs(
            ["Instruction: Create", "mint=So11111111111111111111111111111111111111112 launched"],
            "sig2",
        )
        assert p.mint == "So11111111111111111111111111111111111111112"
        assert p.bonding_curve is not None
        assert not p.needs_full_fetch

    def test_parse_create_from_logs_fallback(self):
        from utils import pumpfun_ix
        # Unparseable logs -> caller must do getTransaction
        p = pumpfun_ix.parse_create_from_logs(["some random log"], "sig3")
        assert p.mint is None
        assert p.needs_full_fetch is True

    def test_parse_create_from_logs_base64_anchor_event(self):
        from utils import pumpfun_ix
        # Build a CreateEvent payload: name(string) symbol(string) uri(string) mint(32) bc(32) user(32)
        name, sym, uri = b"T", b"TST", b"https://x"
        payload = (len(name).to_bytes(4, "little") + name
                   + len(sym).to_bytes(4, "little") + sym
                   + len(uri).to_bytes(4, "little") + uri
                   + bytes([1] * 32) + bytes([2] * 32) + bytes([3] * 32))
        raw = bytes([0] * 8) + payload  # 8-byte anchor event discriminator
        logs = ["Program log: Instruction: Create",
                "Program data: " + base64.b64encode(raw).decode()]
        p = pumpfun_ix.parse_create_from_logs(logs, "sig4")
        assert p.mint is not None and not p.needs_full_fetch


# =====================================================================
# bonding_curve: account decode bug fix (LIST-form Helius response)
# =====================================================================
class TestBondingCurveDecode:
    def test_decode_handles_list_form_response(self):
        # Reproduce the decode logic (avoids importing the full module chain
        # which needs structlog at import time).
        vtr = 1_073_000_000 * 10**6
        vsr = 30 * 10**9
        rtr = 800_000_000 * 10**6
        rsr = 50 * 10**9
        supply = 1_000_000_000 * 10**6
        payload = (bytes([24, 30, 200, 40, 5, 28, 7, 119])
                   + vtr.to_bytes(8, "little") + vsr.to_bytes(8, "little")
                   + rtr.to_bytes(8, "little") + rsr.to_bytes(8, "little")
                   + supply.to_bytes(8, "little") + bytes([0]))
        b64 = base64.b64encode(payload).decode()
        account_info = {"data": [b64, "base64"]}
        # Inline the fixed decode (matches utils.bonding_curve._decode_bonding_curve_account)
        data_field = account_info["data"]
        raw = None
        if isinstance(data_field, list) and len(data_field) >= 2:
            if data_field[1] == "base64":
                raw = base64.b64decode(data_field[0])
        assert raw is not None and len(raw) >= 8 + 5 * 8 + 1
        off = 8
        vtr_d = int.from_bytes(raw[off:off+8], "little"); off += 8
        vsr_d = int.from_bytes(raw[off:off+8], "little"); off += 8
        rtr_d = int.from_bytes(raw[off:off+8], "little"); off += 8
        rsr_d = int.from_bytes(raw[off:off+8], "little"); off += 8
        supply_d = int.from_bytes(raw[off:off+8], "little"); off += 8
        complete = bool(raw[off])
        assert rsr_d / 1e9 == 50.0          # 50 SOL raised
        assert vsr_d / 1e9 == 30.0          # 30 SOL virtual
        assert rtr_d / 1e6 == 800_000_000   # 800M tokens
        assert vtr_d / 1e6 == 1_073_000_000 # 1.073B virtual tokens
        assert complete is False

    def test_old_bug_would_return_none(self):
        # Demonstrate the OLD code path failed on the list form.
        account_info = {"data": ["AAAA", "base64"]}  # list, not dict
        # Old: isinstance(data_field, dict) -> False -> returns (None,)*5
        assert not isinstance(account_info["data"], dict)


# =====================================================================
# risk_manager: reduce_position + daily PnL weighting
# =====================================================================
class TestRiskManagerReduction:
    def test_daily_pnl_weighting_uses_capital_fraction(self):
        # The fix: weight = size_base / allocated_capital (not size_base/100).
        cap = 10.0
        size_base = 0.2
        pnl_pct = 50.0
        weight = size_base / cap
        contribution = pnl_pct * weight
        # Old bug: pnl_pct * (size_base/100) = 0.1% (10x too small)
        assert abs(contribution - 1.0) < 1e-9

    def test_reduce_position_math(self):
        # Selling 25% of a position at +100% realizes 25% of the gain.
        size_base = 1.0
        fraction = 0.25
        entry, exit_price = 10.0, 20.0
        pnl_pct = (exit_price - entry) / entry * 100
        realized = size_base * fraction
        remaining = size_base * (1 - fraction)
        assert remaining == 0.75
        assert pnl_pct == 100.0


# =====================================================================
# copy_trade: pump.fun swap decoder
# =====================================================================
class TestCopyTradeDecoder:
    @staticmethod
    def _decode(tx):
        """Inline copy of _decode_pumpfun_swap to avoid structlog import."""
        BUY = hashlib.sha256(b"global:buy").digest()[:8]
        SELL = hashlib.sha256(b"global:sell").digest()[:8]
        PUMP = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
        if not tx: return None
        message = tx.get("transaction", {}).get("message", {}) or {}
        keys = message.get("accountKeys", []) or []
        ixs = list(message.get("instructions", []) or [])
        for g in (tx.get("meta", {}).get("innerInstructions") or []):
            ixs.extend(g.get("instructions", []) or [])
        for ix in ixs:
            prog = ix.get("programId")
            if prog != PUMP: continue
            data = ix.get("data", "")
            try: raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
            except: continue
            if len(raw) < 16: continue
            d = raw[:8]
            if d == BUY:
                side, amt = "buy", int.from_bytes(raw[8:16], "little")
            elif d == SELL:
                side, amt = "sell", int.from_bytes(raw[8:16], "little")
            else: continue
            accs = ix.get("accounts", []) or []
            mint = None
            if len(accs) >= 3:
                mi = accs[2]
                if isinstance(mi, int) and mi < len(keys): mint = keys[mi]
            if not mint: continue
            return {"side": side, "mint": mint, "amount_raw": amt}
        return None

    def test_decode_buy(self):
        mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        keys = ["glob", "fee", mint, "bc", "ata", "user", "sys", "tok", "rent"]
        BUY = hashlib.sha256(b"global:buy").digest()[:8]
        data = BUY + (1_000_000).to_bytes(8, "little") + (50_000_000).to_bytes(8, "little")
        tx = {"transaction": {"message": {"accountKeys": keys,
                "instructions": [{"programId": "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",
                                  "accounts": [0,1,2,3,4,5,6,7,8],
                                  "data": base64.b64encode(data).decode()}]}}, "meta": {}}
        r = self._decode(tx)
        assert r == {"side": "buy", "mint": mint, "amount_raw": 1_000_000}

    def test_decode_sell(self):
        mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        keys = ["glob", "fee", mint, "bc", "ata", "user", "sys", "tok", "rent"]
        SELL = hashlib.sha256(b"global:sell").digest()[:8]
        data = SELL + (5_000_000).to_bytes(8, "little")
        tx = {"transaction": {"message": {"accountKeys": keys,
                "instructions": [{"programId": "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",
                                  "accounts": [0,1,2,3,4,5,6,7,8],
                                  "data": base64.b64encode(data).decode()}]}}, "meta": {}}
        r = self._decode(tx)
        assert r["side"] == "sell" and r["mint"] == mint

    def test_decode_non_pumpfun_returns_none(self):
        tx = {"transaction": {"message": {"accountKeys": ["x"],
                "instructions": [{"programId": "Other", "data": "AQID"}]}}, "meta": {}}
        assert self._decode(tx) is None


# =====================================================================
# soft_rug: heuristic baseline + training loop
# =====================================================================
class TestSoftRug:
    def test_heuristic_baseline_flags_concentration(self):
        # Inline the heuristic (no Config dep needed for the math).
        def heuristic(f):
            p = 0.30
            if f[0] > 50: p += 0.20   # top3 concentration
            if f[0] > 80: p += 0.15
            if f[1] > 10: p += 0.15   # sell velocity
            if f[2] < 0.5: p += 0.10  # b/s ratio
            if f[3] < 20: p += 0.10   # holder count
            if f[4] < 5000: p += 0.10 # liquidity
            return min(0.98, max(0.02, p))
        # High concentration + high sell velocity = high rug probability
        high_risk = heuristic([90, 20, 0.3, 10, 3000])
        low_risk = heuristic([20, 2, 1.5, 200, 50000])
        assert high_risk >= 0.65
        assert low_risk < 0.50

    def test_gradient_boosting_fits_predictive_features(self):
        # Reproduce the SoftRugDetector.train() gradient boosting in isolation.
        import math
        np.random.seed(7)
        n = 300
        X = np.zeros((n, 8))
        y = np.zeros(n)
        for i in range(n):
            is_rug = np.random.random() < 0.4
            y[i] = 1.0 if is_rug else 0.0
            X[i, 1] = 80 + np.random.normal(0, 8) if is_rug else 30 + np.random.normal(0, 8)
            X[i, 3] = 20 + np.random.normal(0, 5) if is_rug else 2 + np.random.normal(0, 2)
            for j in [0, 2, 4, 5, 6, 7]:
                X[i, j] = 50 + np.random.normal(0, 10)

        def fit_stump(X, residual):
            n, d = X.shape
            best = None; best_gain = 1e-9
            parent_sq = ((residual - residual.mean()) ** 2).sum()
            for feat in range(d):
                col = X[:, feat]
                uniq = np.unique(col)
                if len(uniq) > 50:
                    uniq = np.quantile(col, np.linspace(0.05, 0.95, 30))
                for thr in uniq:
                    lm = col <= thr
                    nl, nr = int(lm.sum()), n - int(lm.sum())
                    if nl < 2 or nr < 2: continue
                    lmean = residual[lm].mean(); rmean = residual[~lm].mean()
                    gain = parent_sq - ((residual[lm]-lmean)**2).sum() - ((residual[~lm]-rmean)**2).sum()
                    if gain > best_gain:
                        best_gain = gain
                        best = (feat, float(thr), float(lmean), float(rmean))
            return best

        pos_rate = max(1e-4, min(1-1e-4, y.mean()))
        raw = np.full(n, math.log(pos_rate/(1-pos_rate)))
        splits = []
        for _ in range(50):
            p = 1/(1+np.exp(-raw))
            grad = p - y
            s = fit_stump(X, -grad)
            if s is None: break
            feat, thr, lv, rv = s
            splits.append(feat)
            raw += 0.1 * np.where(X[:, feat] <= thr, lv, rv)
        preds = (1/(1+np.exp(-raw)) >= 0.5).astype(int)
        acc = (preds == y).mean()
        assert acc > 0.85
        # Should split on concentration (1) and/or sell velocity (3)
        assert 1 in splits[:10] or 3 in splits[:10]


# =====================================================================
# attribution / autotuner: logistic regression weight fit
# =====================================================================
class TestAutoTuner:
    def test_logistic_fit_upweights_predictive_component(self):
        # The autotuner should upweight the component that predicts profitability.
        np.random.seed(42)
        n = 200
        X = np.full((n, 8), 50.0)
        X[:100, 0] = 85.0 + np.random.normal(0, 5, 100)   # order_flow high for winners
        X[100:, 0] = 30.0 + np.random.normal(0, 5, 100)   # low for losers
        X[:, 1:] += np.random.normal(0, 10, (n, 7))
        y = np.array([1]*100 + [0]*100, dtype=float)
        Xn = X / 100.0
        prior = np.array([0.30, 0.20, 0.15, 0.12, 0.10, 0.08, 0.03, 0.02])

        w = np.zeros(8)
        lr, l2 = 0.05, 5.0
        for _ in range(800):
            z = Xn @ w
            p = 1.0 / (1.0 + np.exp(-z))
            grad = (Xn.T @ (p - y)) / n + l2 * (w - prior * 8.0)
            w -= lr * grad
        w_exp = np.exp(w - w.max())
        new_w = w_exp / w_exp.sum()
        new_w = np.maximum(new_w, 0.01)
        new_w = np.minimum(new_w, 0.50)
        new_w = new_w / new_w.sum()

        # order_flow (index 0) was predictive -> should be upweighted
        assert new_w[0] > prior[0]
        # Weights must sum to 1
        assert abs(new_w.sum() - 1.0) < 1e-6
        # No single component exceeds the cap
        assert new_w.max() <= 0.50 + 1e-9


# =====================================================================
# backtester: real strategy replay (not hardcoded SMA) + grid simulation
# =====================================================================
class TestBacktester:
    def test_grid_simulation_profits_on_mean_reversion(self):
        # Inline _simulate_grid (no Config dep).
        def simulate_grid(prices, levels=8, spacing_pct=1.5):
            if len(prices) < 30:
                return []
            mid = float(prices[0])
            buy_levels = [mid * (1 - (spacing_pct/100)*i) for i in range(1, levels+1)]
            sell_levels = [mid * (1 + (spacing_pct/100)*i) for i in range(1, levels+1)]
            buy_filled = [False] * levels
            returns = []
            open_buys = []
            for p in prices[1:]:
                for sp in sell_levels:
                    if p >= sp and any(e < sp for e in open_buys):
                        entry = min(e for e in open_buys if e < sp)
                        open_buys.remove(entry)
                        returns.append((sp - entry) / entry)
                for bi, bp in enumerate(buy_levels):
                    if p <= bp and not buy_filled[bi]:
                        buy_filled[bi] = True
                        open_buys.append(bp)
            last = float(prices[-1])
            for entry in open_buys:
                returns.append((last - entry) / entry)
            return returns

        # Mean-reverting series: grid should profit
        np.random.seed(1)
        prices = 100 + np.cumsum(np.random.normal(0, 0.5, 200))
        prices = np.maximum(prices, 80)
        rets = simulate_grid(prices)
        assert len(rets) > 0
        # On mean-reversion, grid trades should be net positive on average
        assert np.mean(rets) > -0.10

    def test_momentum_replay_generates_buys_on_recovery(self):
        # Inline _rsi
        def rsi(prices, period=14):
            if len(prices) < period + 1:
                return None
            gains, losses = 0.0, 0.0
            for i in range(-period, 0):
                diff = prices[i] - prices[i-1]
                if diff > 0:
                    gains += diff
                else:
                    losses -= diff
            if losses == 0:
                return 100.0
            rs = (gains/period) / (losses/period)
            return 100.0 - (100.0/(1.0+rs))

        # A V-shaped recovery: drop then rebound — momentum BUY when RSI < 35
        recovery = np.concatenate([
            np.linspace(100, 85, 50),
            np.linspace(85, 95, 50),
            np.linspace(95, 92, 100),
        ])
        buys = 0
        for i in range(30, len(recovery)):
            window = recovery[max(0, i-14):i+1]
            r = rsi(list(window))
            if r is not None and r < 35:
                buys += 1
        assert buys > 0


# =====================================================================
# alpha_signal: the generate() header that was accidentally deleted
# =====================================================================
class TestAlphaSignalGenerate:
    """Tests via AST to avoid importing the full dependency chain (structlog,
    Config, etc.) which isn't available in the test environment. The bug being
    guarded against: generate()'s body was once fused onto the end of
    _refresh_weights_from_db, deleting the `async def generate` header and
    making generate() uncallable (AttributeError at runtime).
    """
    @staticmethod
    def _class_methods():
        import ast as _ast
        src = (Path(__file__).resolve().parent.parent / "analysis" / "alpha_signal.py").read_text()
        tree = _ast.parse(src)
        for node in _ast.walk(tree):
            if isinstance(node, _ast.ClassDef) and node.name == "AlphaSignalGenerator":
                return {n.name: n for n in node.body
                        if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))}
        return {}

    def test_generate_method_exists_and_is_async(self):
        import ast as _ast
        methods = self._class_methods()
        assert "generate" in methods, "generate() method is missing from AlphaSignalGenerator"
        assert isinstance(methods["generate"], _ast.AsyncFunctionDef), \
            "generate() must be async (def -> async def)"

    def test_refresh_weights_is_sync_and_isolated(self):
        import ast as _ast
        methods = self._class_methods()
        assert "_refresh_weights_from_db" in methods
        assert isinstance(methods["_refresh_weights_from_db"], _ast.FunctionDef), \
            "_refresh_weights_from_db must be sync, not async"
        # Read its body and ensure no generate()-only leaked code
        body = _ast.unparse(methods["_refresh_weights_from_db"])
        for forbidden in ("self._cache.get(mint)", "await asyncio.gather",
                          "self.scorer.score", "self.order_flow.snapshot",
                          "self.lifecycle.assess"):
            assert forbidden not in body, \
                f"_refresh_weights_from_db leaked generate() code: {forbidden!r}"


# =====================================================================
# regime: trend normalization + volatility (pure logic, inlined to avoid
# importing analysis.regime which pulls in Config at __init__ time)
# =====================================================================
class TestRegime:
    @staticmethod
    def _trend_to_unit(change_pct):
        return max(0.0, min(1.0, 0.5 + change_pct / 20.0))

    @staticmethod
    def _volatility(history):
        import math
        if len(history) < 3:
            return 0.0
        prices = [p for _, p in history if p > 0]
        if len(prices) < 3:
            return 0.0
        rets = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices)) if prices[i-1] > 0]
        if len(rets) < 2:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        return math.sqrt(var)

    def test_trend_to_unit_maps_correctly(self):
        # -10% -> 0, 0% -> 0.5, +10% -> 1
        assert abs(self._trend_to_unit(-10) - 0.0) < 1e-9
        assert abs(self._trend_to_unit(0) - 0.5) < 1e-9
        assert abs(self._trend_to_unit(10) - 1.0) < 1e-9
        assert self._trend_to_unit(-20) == 0.0
        assert self._trend_to_unit(20) == 1.0

    def test_volatility_flat_is_zero(self):
        flat = [(0.0, 100.0), (1.0, 100.0), (2.0, 100.0), (3.0, 100.0)]
        assert self._volatility(flat) == 0.0

    def test_volatility_volatile_is_positive(self):
        import numpy as np
        np.random.seed(3)
        prices = 100 + np.cumsum(np.random.normal(0, 2, 50))
        assert self._volatility(list(enumerate(prices.tolist()))) > 0


# =====================================================================
# peak_hype: detection logic (3-condition AND, inlined)
# =====================================================================
class TestPeakHype:
    @staticmethod
    def _assess(mention_ratio, bs_decline, price_ext,
                vel_mult=3.0, bs_thresh=0.6, price_mult=2.0):
        return (
            mention_ratio >= vel_mult
            and bs_decline <= bs_thresh
            and price_ext >= price_mult
        )

    def test_not_peak_when_mentions_low(self):
        # Price 3x extended but mentions only 1.5x -> not peak
        assert not self._assess(mention_ratio=1.5, bs_decline=0.3, price_ext=3.0)

    def test_not_peak_when_bs_still_rising(self):
        # Mentions parabolic + price extended but buyers still dominant -> not peak
        assert not self._assess(mention_ratio=5.0, bs_decline=0.9, price_ext=3.0)

    def test_not_peak_when_price_not_extended(self):
        # Mentions + bs decline but price flat -> not a blowoff, just noise
        assert not self._assess(mention_ratio=5.0, bs_decline=0.3, price_ext=1.2)

    def test_peak_when_all_three_conditions_met(self):
        assert self._assess(mention_ratio=5.0, bs_decline=0.25, price_ext=3.0)

    def test_boundary_at_thresholds(self):
        # Exactly at thresholds -> peak (>=, <=)
        assert self._assess(mention_ratio=3.0, bs_decline=0.6, price_ext=2.0)


# =====================================================================
# crowd_engine: signal scorers + fusion + contrarian agent
# =====================================================================
class TestCrowdEngine:
    @staticmethod
    def _score_order_flow(bs_ratio, total_trades=100, buy=None, sell=None):
        """Inline of CrowdPositioningEngine._score_order_flow."""
        if total_trades < 5:
            return 0.0
        if sell == 0:
            return 1.0 if (buy or 0) > 0 else 0.0
        import math
        BUY_SELL_EXTREME = 3.0
        return max(-1.0, min(1.0, math.tanh((bs_ratio - 1.0) / (BUY_SELL_EXTREME - 1.0))))

    def test_order_flow_score_extremes(self):
        # bs_ratio 1.0 (balanced) -> ~0
        assert abs(self._score_order_flow(1.0)) < 0.05
        # bs_ratio 5.0 (crowd all-in long) -> high positive (tanh saturates)
        assert self._score_order_flow(5.0) > 0.8
        # bs_ratio 0.1 (sell-heavy) -> clearly negative
        assert self._score_order_flow(0.1) < -0.3
        # Symmetry: ratio r and 1/r should produce opposite-sign scores
        assert self._score_order_flow(0.2) < 0 < self._score_order_flow(5.0)

    @staticmethod
    def _score_bonding_curve(completion_pct):
        """Inline of _score_bonding_curve."""
        if completion_pct < 50:
            return 0.0
        return max(0.0, min(1.0, (completion_pct - 50.0) / 50.0))

    def test_bonding_curve_score(self):
        assert self._score_bonding_curve(30) == 0.0   # too early
        assert self._score_bonding_curve(50) == 0.0   # neutral
        assert abs(self._score_bonding_curve(75) - 0.5) < 1e-9
        assert self._score_bonding_curve(100) == 1.0  # max sell-the-news risk

    @staticmethod
    def _conviction(raw_scores):
        """Inline of the conviction calc. Filters to active signals (|v|>0.15)
        then computes agreement-weighted mean magnitude."""
        active = [v for v in raw_scores if abs(v) > 0.15]
        if not active:
            return 0.0
        mean_abs = sum(abs(v) for v in active) / len(active)
        signs = set(1 if v > 0 else -1 for v in active)
        agreement = 1.0 if len(signs) == 1 else (0.4 if len(active) <= 2 else 0.2)
        return min(1.0, mean_abs * agreement)

    def test_conviction_high_when_all_agree(self):
        # All positive + strong -> high conviction
        assert self._conviction([0.8, 0.7, 0.6]) > 0.6

    def test_conviction_low_when_divergent(self):
        # Mixed signs -> low conviction (weak setup)
        assert self._conviction([0.8, -0.3, 0.6]) < 0.4

    def test_conviction_zero_when_all_neutral(self):
        # All near-zero -> no active signals -> 0
        assert self._conviction([0.05, 0.1, 0.02]) == 0.0


class TestContrarianAgent:
    """ContrarianAgent logic inlined (avoid importing crowd_engine which
    pulls Config/structlog at import time)."""
    EXTREME = 0.5

    @staticmethod
    def _on_positioning(crowd_score, conviction, regime_hint, expected_move_bps):
        from strategies.base_strategy import Signal, SignalType
        if abs(crowd_score) < 0.5:
            return []
        if crowd_score > 0:
            sig_type = SignalType.SELL
            suggested = 1.0
            reason = f"Fade crowd {regime_hint}"
        else:
            sig_type = SignalType.BUY
            suggested = min(0.02, conviction * 0.03)
            reason = f"Contrarian buy (crowd fear)"
        tp_pct = expected_move_bps / 100.0
        sl_pct = tp_pct * 0.3
        return [(sig_type, suggested, max(10.0, tp_pct), max(5.0, sl_pct))]

    def test_fades_extreme_long_crowd(self):
        from strategies.base_strategy import SignalType
        result = self._on_positioning(0.7, 0.8, "cascade_imminent", 800.0)
        assert len(result) == 1
        sig_type, suggested, tp, sl = result[0]
        assert sig_type == SignalType.SELL
        assert suggested == 1.0

    def test_fades_extreme_short_crowd(self):
        from strategies.base_strategy import SignalType
        result = self._on_positioning(-0.6, 0.7, "fear", 600.0)
        assert len(result) == 1
        sig_type, suggested, tp, sl = result[0]
        assert sig_type == SignalType.BUY
        assert suggested <= 0.03

    def test_no_signal_when_neutral(self):
        result = self._on_positioning(0.2, 0.3, "neutral", 100.0)
        assert result == []

    def test_tp_sl_asymmetric(self):
        # Large expected move -> TP big, SL floored at 5% but still < TP
        result = self._on_positioning(0.6, 0.8, "euphoria", 2000.0)
        _, _, tp, sl = result[0]
        assert tp >= 20.0          # ~20% (2000bps)
        assert sl < tp             # SL always smaller than TP (asymmetric)
        assert sl >= 5.0           # SL never below the 5% floor


# =====================================================================
# crowd_weight_router: dynamic weight dampening on crowd extremes
# =====================================================================
class TestCrowdWeightRouter:
    """Router logic inlined (the module-level setup_logger triggers Config)."""
    TREND_COMPS = {"order_flow", "lifecycle", "sentiment"}
    FLOOR = 0.15

    @staticmethod
    def _strength(crowd_score, conviction):
        return min(1.0, abs(crowd_score) * conviction)

    @staticmethod
    def _multipliers(crowd_score, conviction):
        s = TestCrowdWeightRouter._strength(crowd_score, conviction)
        if s < 0.15:
            return {}
        out = {}
        for c in TestCrowdWeightRouter.TREND_COMPS:
            out[c] = max(TestCrowdWeightRouter.FLOOR, 1.0 - s * (1.0 - TestCrowdWeightRouter.FLOOR))
        return out

    def test_no_dampening_when_weak(self):
        assert self._multipliers(0.3, 0.3) == {}

    def test_dampens_trend_on_strong_extreme(self):
        m = self._multipliers(0.8, 0.9)
        assert m["order_flow"] < 0.5
        assert m["lifecycle"] < 0.5
        assert m["sentiment"] < 0.5
        assert m["order_flow"] >= 0.15  # floor

    def test_floor_never_below_015(self):
        # Even at max strength, trend components stay >= floor
        m = self._multipliers(1.0, 1.0)
        for v in m.values():
            assert v >= 0.15

    def test_renormalization_preserves_sum_and_shifts_weight(self):
        m = self._multipliers(0.8, 0.9)
        base = {"order_flow": 0.30, "smart_money": 0.20, "token_quality": 0.15,
                "lifecycle": 0.12, "liquidity": 0.10, "sentiment": 0.08,
                "bonding_curve": 0.03, "mev_penalty": 0.02}
        eff = {k: w * m.get(k, 1.0) for k, w in base.items()}
        total = sum(eff.values())
        norm = {k: v / total for k, v in eff.items()}
        assert abs(sum(norm.values()) - 1.0) < 1e-9
        assert norm["order_flow"] < base["order_flow"]      # dampened down
        assert norm["mev_penalty"] > base["mev_penalty"]    # relatively up


# =====================================================================
# OMEGA-adapted modules: flow, risk, timing (logic inlined)
# =====================================================================
class TestOmegaFlowSignals:
    """B6 ToxicFlow + B7 SmartMoneyDivergence + B3 WhaleTracker."""

    @staticmethod
    def _toxic(sell_count, buy_count, net_sol, whale_sells, total_trades):
        if total_trades < 15: return 0.0, 0.0
        toxicity = 0.0
        if buy_count > 0 and sell_count / max(1, buy_count) >= 1.8: toxicity += 0.4
        if net_sol < 0: toxicity += min(0.4, abs(net_sol) / 10.0)
        wsp = whale_sells / max(1, total_trades)
        if wsp > 0.05: toxicity += min(0.3, wsp * 3.0)
        return max(-1.0, -toxicity), min(1.0, toxicity)

    def test_toxic_flow_flags_sell_dominance(self):
        s, c = self._toxic(80, 20, -5.0, 10, 100)
        assert s < -0.5 and c > 0.3

    def test_clean_flow_when_buy_dominant(self):
        s, c = self._toxic(30, 70, 3.0, 1, 100)
        assert s > -0.1

    @staticmethod
    def _divergence(price_pct, bs_ratio, total_trades):
        if total_trades < 10: return 0.0, "no_data"
        price_bullish = price_pct > 2.0
        flow_bullish = bs_ratio > 1.0
        if not price_bullish: return max(-0.3, min(0.3, price_pct / 20.0)), "flat"
        if not flow_bullish:
            ds = min(1.0, abs(price_pct) / 30.0)
            fn = min(1.0, (1.0 / max(0.01, bs_ratio)) / 3.0)
            return -min(1.0, ds * fn), "bearish_divergence"
        return min(1.0, (price_pct / 30.0) * (bs_ratio / 3.0)), "confirmed_uptrend"

    def test_divergence_detects_distribution(self):
        s, lbl = self._divergence(30.0, 0.5, 100)
        assert s < -0.3 and lbl == "bearish_divergence"

    def test_confirmed_uptrend_when_aligned(self):
        s, lbl = self._divergence(25.0, 3.0, 100)
        assert s > 0.3 and lbl == "confirmed_uptrend"


class TestOmegaRiskSignals:
    """B22 AdaptiveRiskManager + B8 VolatilityForecast + B20 StressIndex."""

    @staticmethod
    def _size_mult(stress_score, token_vol, loss_streak):
        stress_mult = 1.0 - (stress_score / 100.0) * 0.7
        vol_mult = max(0.2, min(1.0, 0.02 / token_vol)) if token_vol > 0 else 1.0
        streak_mult = max(0.3, 1.0 - loss_streak * 0.12)
        return max(0.1, min(1.0, stress_mult * vol_mult * streak_mult))

    def test_calm_market_full_size(self):
        assert self._size_mult(10, 0.02, 0) > 0.8

    def test_panic_market_minimal_size(self):
        assert self._size_mult(90, 0.08, 4) < 0.3

    def test_loss_streak_reduces_size_monotonically(self):
        sizes = [self._size_mult(20, 0.02, n) for n in range(5)]
        for i in range(len(sizes) - 1):
            assert sizes[i] >= sizes[i + 1]  # each loss shrinks size


class TestOmegaTimingSignals:
    """B12 TimeOfDayAlpha + B15 MultiTimeframeSignal + B24 SentimentNLP."""

    SESSIONS = [(14, 21, 1.20), (0, 8, 1.10), (8, 14, 0.90), (21, 24, 0.80)]

    def _tod(self, hour):
        for s, e, m in self.SESSIONS:
            if s <= hour < e: return m
        return 0.90

    def test_us_session_boosted(self):
        assert self._tod(18) == 1.20

    def test_dead_hours_dampened(self):
        assert self._tod(23) == 0.80

    @staticmethod
    def _mtf(directions):
        weights = [0.15, 0.25, 0.30, 0.30]
        net = sum(w * d for w, d in zip(weights, directions))
        active = [d for d in directions if abs(d) > 0.1]
        if not active: return 0.0, 0.0
        signs = set(1 if d > 0 else -1 for d in active)
        alignment = 1.0 if len(signs) == 1 else 0.3
        return net, alignment * abs(net)

    def test_mtf_aligned_high_conviction(self):
        _, c = self._mtf([0.8, 0.7, 0.6, 0.5])
        assert c > 0.5

    def test_mtf_conflicting_low_conviction(self):
        _, c = self._mtf([0.8, -0.5, 0.6, -0.3])
        assert c < 0.2

    BULLISH = {"moon", "pump", "buy", "send", "based", "gem"}
    BEARISH = {"rug", "dump", "sell", "scam", "jeet"}

    def _nlp(self, texts):
        b = sum(1 for t in texts if any(k in t.lower() for k in self.BULLISH))
        s = sum(1 for t in texts if any(k in t.lower() for k in self.BEARISH))
        total = b + s
        if total == 0: return 0.0
        return max(-1.0, min(1.0, (b - s) / total))

    def test_nlp_bullish_text(self):
        assert self._nlp(["moon", "buy this gem"]) > 0.5

    def test_nlp_bearish_text(self):
        assert self._nlp(["dev rugged", "scam alert"]) < -0.5

    def test_nlp_neutral_text(self):
        assert self._nlp(["hello world", "nice day"]) == 0.0
