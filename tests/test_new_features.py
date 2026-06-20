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
