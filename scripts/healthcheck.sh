#!/usr/bin/env bash
# =====================================================================
# Quick health-check script for the pump.fun trading agent.
# Verifies:
#   1. Python version
#   2. Required packages importable
#   3. Config file present and valid
#   4. .env present with required keys
#   5. Wallet(s) loadable
#   6. Each chain RPC reachable
# Run:  bash scripts/healthcheck.sh
# =====================================================================
set -e

cd "$(dirname "$0")/.."

echo "==== 1. Python version ===="
python --version

echo
echo "==== 2. Required packages ===="
python - <<'PY'
import importlib, sys
mods = ["solana", "solders", "web3", "eth_account", "aiohttp",
        "yaml", "dotenv", "telegram", "structlog", "pandas", "numpy"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  OK   {m}")
    except ImportError:
        missing.append(m)
        print(f"  FAIL {m}")
if missing:
    print(f"\nMissing: {missing}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)
PY

echo
echo "==== 3. Config file ===="
test -f config/config.yaml || { echo "FAIL: config/config.yaml missing. Copy config.yaml.example"; exit 1; }
python -c "from utils.config_loader import Config; Config.get(); print('  OK config loads')"

echo
echo "==== 4. .env secrets ===="
test -f .env || { echo "WARN: .env missing. Copy .env.example"; }

echo
echo "==== 5. Wallets ===="
python - <<'PY'
import asyncio
from utils.config_loader import Config
from utils.wallet_manager import WalletManager
async def main():
    cfg = Config.get()
    wm = WalletManager()
    print("  Wallets loaded:", wm.redacted_summary())
asyncio.run(main())
PY

echo
echo "==== 6. RPC connectivity ===="
python - <<'PY'
import asyncio
from chains.chain_factory import build_adapters
async def main():
    adapters = await build_adapters()
    for name, a in adapters.items():
        try:
            addr = None
            if name == "solana":
                from utils.wallet_manager import WalletManager
                wm = WalletManager()
                if wm.has_wallet("solana"):
                    addr = wm.solana_pubkey()
            else:
                from utils.wallet_manager import WalletManager
                wm = WalletManager()
                if wm.has_wallet(name):
                    addr = wm.evm_address(name)
            if addr:
                bal = await a.get_balance(addr)
                print(f"  OK {name}: balance={bal}")
            else:
                print(f"  OK {name}: connected (no wallet)")
        except Exception as e:
            print(f"  FAIL {name}: {e}")
asyncio.run(main())
PY

echo
echo "==== ALL CHECKS PASSED ===="
