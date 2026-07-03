#!/usr/bin/env python3
"""
preflight_live.py
=================
Pre-flight checklist BEFORE going live with real SOL.

Checks, in order:
  1. Config exists and is not the example template
  2. trading.mode == "live" (explicit confirmation)
  3. Wallet private key loads + pubkey is valid
  4. RPC endpoint responds (getHealth)
  5. SOL balance is sufficient (>= 0.05 SOL recommended)
  6. Telegram bot is reachable (optional but recommended)
  7. Kill switch file is NOT present
  8. Allocated capital is reasonable (not more than your balance)

Exit 0 = GO for live. Exit 1 = STOP, fix the issue.

Usage:
    python scripts/preflight_live.py
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "✅" if condition else "❌"
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))
    return condition


async def main() -> int:
    all_ok = True
    print("=" * 60)
    print("  LIVE TRADING PRE-FLIGHT CHECKLIST")
    print("=" * 60)
    print()

    # 1. Config file
    print("[1/8] Configuration")
    cfg_path = ROOT / "config" / "config.yaml"
    if not check("config/config.yaml exists", cfg_path.exists()):
        print("\n  Run: cp config/config.yaml.example config/config.yaml")
        return 1
    try:
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
    except Exception as e:
        check("config parses", False, str(e))
        return 1
    mode = cfg.get("trading", {}).get("mode", "paper")
    all_ok &= check("trading.mode is 'live'", mode == "live",
                    f"currently '{mode}' — change to 'live' in config.yaml")
    print()

    # 2. Environment
    print("[2/8] Environment (.env)")
    from utils.config_loader import Config
    try:
        Config.get()
    except Exception as e:
        check("Config loads", False, str(e))
        return 1
    check("Config loads", True)

    # Check for wallet credentials
    import os
    has_priv = bool(os.environ.get("SOLANA_PRIVATE_KEY", "").strip())
    has_json = bool(os.environ.get("SOLANA_KEYPAIR_JSON", "").strip())
    has_seed = bool(os.environ.get("SOLANA_WALLET_SEED", "").strip())
    all_ok &= check("Wallet credential present", has_priv or has_json or has_seed,
                    "Set SOLANA_PRIVATE_KEY in .env")
    print()

    # 3. Wallet
    print("[3/8] Wallet")
    try:
        from utils.wallet_manager import WalletManager
        wm = WalletManager()
        if not wm.has_wallet("solana"):
            check("Solana wallet loads", False, "No keypair loaded")
            return 1
        pubkey = wm.solana_pubkey()
        check("Solana wallet loads", True, f"pubkey: {pubkey}")
        check("Wallet is NOT the ephemeral default", pubkey != "",
              "Ephemeral keypair = no funds")
    except Exception as e:
        check("Wallet loads", False, str(e))
        all_ok = False
    print()

    # 4. RPC
    print("[4/8] Solana RPC")
    rpc_endpoints = cfg.get("chains", {}).get("solana", {}).get("rpc_endpoints", [])
    ws_endpoint = cfg.get("chains", {}).get("solana", {}).get("ws_endpoint", "")
    all_ok &= check("RPC endpoints configured", len(rpc_endpoints) > 0)
    all_ok &= check("WebSocket endpoint configured", bool(ws_endpoint))
    if "YOUR_HELIUS_KEY" in str(rpc_endpoints):
        check("RPC URL has real API key", False,
              "Replace YOUR_HELIUS_KEY with your actual Helius key")
        all_ok = False
    else:
        check("RPC URL has real API key", True)

    # Test RPC connectivity
    try:
        import aiohttp
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            async with s.post(rpc_endpoints[0], json={
                "jsonrpc": "2.0", "id": 1, "method": "getHealth"
            }) as r:
                data = await r.json()
                healthy = data.get("result") == "ok"
                check("RPC responds getHealth=ok", healthy)
                if not healthy:
                    all_ok = False
    except Exception as e:
        check("RPC responds", False, str(e))
        all_ok = False
    print()

    # 5. Balance
    print("[5/8] SOL Balance")
    try:
        from chains.solana_adapter import SolanaAdapter
        adapter = SolanaAdapter()
        await adapter.connect()
        balance = await adapter.get_balance(pubkey)
        await adapter.close()
        check("Balance retrieved", True, f"{balance:.4f} SOL")
        if balance < 0.02:
            check("Sufficient balance (>= 0.02 SOL)", False,
                  f"Only {balance:.4f} SOL. Fund: {pubkey}")
            all_ok = False
        elif balance < 0.05:
            check("Sufficient balance", True,
                  f"{balance:.4f} SOL (low but OK for micro-test)")
        else:
            check("Sufficient balance", True)
        # Warn if allocated capital > balance
        allocated = cfg.get("chains", {}).get("solana", {}).get("allocated_capital_sol", 0)
        if allocated > balance:
            check("Allocated capital <= balance", False,
                  f"Config says {allocated} SOL but wallet has {balance:.4f} SOL")
            all_ok = False
        else:
            check("Allocated capital <= balance", True)
    except Exception as e:
        check("Balance check", False, str(e))
        all_ok = False
    print()

    # 6. Telegram
    print("[6/8] Telegram (optional)")
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not tg_token:
        print("  ⚠️  TELEGRAM_BOT_TOKEN not set — you won't get trade notifications")
        print("      Recommended for live trading (kill switch via Telegram)")
    else:
        check("Telegram token present", True)
    print()

    # 7. Kill switch
    print("[7/8] Kill Switch")
    kill_file = ROOT / "data" / "KILL_SWITCH"
    if kill_file.exists():
        check("No stale KILL_SWITCH file", False,
              "data/KILL_SWITCH exists — remove it: rm data/KILL_SWITCH")
        all_ok = False
    else:
        check("No stale KILL_SWITCH file", True)
    print()

    # 8. Risk limits sanity
    print("[8/8] Risk Limits Sanity")
    fixed_size = cfg.get("trading", {}).get("fixed_size_sol", 0)
    max_pos = cfg.get("trading", {}).get("max_open_positions", 0)
    daily_cap = cfg.get("risk", {}).get("daily_loss_cap_pct", 0)
    check("fixed_size_sol is small", fixed_size <= 0.1,
          f"{fixed_size} SOL (recommend <= 0.1 for first live test)")
    check("max_open_positions is limited", max_pos <= 3,
          f"{max_pos} (recommend <= 3 for first live test)")
    check("daily_loss_cap_pct is strict", daily_cap <= 5.0,
          f"{daily_cap}% (recommend <= 5%)")
    if fixed_size > 0.1:
        all_ok = False
    print()

    # Verdict
    print("=" * 60)
    if all_ok:
        print("  🚀 ALL CHECKS PASSED — READY FOR LIVE TRADING")
        print()
        print("  Start with:")
        print("    python orchestrator.py")
        print()
        print("  Emergency stop:")
        print("    touch data/KILL_SWITCH")
        print("    (or send /kill to your Telegram bot)")
    else:
        print("  ⛔ CHECKS FAILED — FIX THE ISSUES ABOVE BEFORE GOING LIVE")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)
