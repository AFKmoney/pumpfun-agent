#!/usr/bin/env python3
"""
check_wallet.py
===============
Verify your Solana wallet is correctly connected to the bot.

Run BEFORE starting the bot to confirm:
  - Your wallet keypair loads without error
  - The pubkey matches what you expect
  - Your SOL balance is sufficient to trade
  - A test signature works (proves the keypair can sign transactions)

Usage:
    python scripts/check_wallet.py

Exit codes:
    0 = wallet OK, ready to trade
    1 = wallet missing or insufficient balance
    2 = configuration error (no config / no env)
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


async def main() -> int:
    # Load config + wallet
    try:
        from utils.wallet_manager import WalletManager
        wm = WalletManager()
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        print("   Run: cp config/config.yaml.example config/config.yaml")
        print("   Then fill in your .env with wallet credentials.")
        return 2

    if not wm.has_wallet("solana"):
        print("❌ No Solana wallet configured.")
        print("\n   To connect your pump.fun wallet, set ONE of these in .env:")
        print("   • SOLANA_PRIVATE_KEY=\"...\"     (base58 key from Phantom → Account → Private Key)")
        print("   • SOLANA_KEYPAIR_JSON=\"[...]\"   (JSON array from Solana CLI / Phantom export)")
        print("   • SOLANA_WALLET_SEED=\"word1 word2 ...\"  (12/24 word mnemonic)")
        return 1

    pubkey = wm.solana_pubkey()
    print(f"✅ Wallet loaded successfully!")
    print(f"   Pubkey: {pubkey}")
    print(f"   View on explorer: https://solscan.io/account/{pubkey}")

    # Check balance
    try:
        from chains.chain_factory import build_adapters
        adapters = await build_adapters()
        solana_adapter = adapters.get("solana")
        if solana_adapter:
            balance = await solana_adapter.get_balance(pubkey)
            print(f"\n💰 Balance: {balance:.4f} SOL")
            if balance < 0.05:
                print(f"   ⚠️  Low balance. Recommend at least 0.05 SOL for trading + fees.")
                print(f"   Fund this address: {pubkey}")
                return 1
            else:
                print(f"   ✅ Sufficient balance for trading.")
    except Exception as e:
        print(f"\n⚠️  Could not check balance (RPC issue?): {e}")
        print(f"   Check your HELIUS_API_KEY and RPC endpoint in config.")

    # Test signature (proves the keypair is valid + can sign)
    try:
        from solders.message import Message
        msg = Message.new_with_blockhash(
            # A harmless SystemProgram transfer of 0 SOL to self
            __import__("solders.system_program", fromlist=["transfer", "TransferParams"])
            .transfer(__import__("solders.system_program", fromlist=["TransferParams"])
                      .TransferParams(from_pubkey=wm.solana_keypair.pubkey(),
                                      to_pubkey=wm.solana_keypair.pubkey(),
                                      lamports=0)),
            wm.solana_keypair.pubkey(),
            __import__("solders.hash", fromlist=["Hash"]).default(),
        )
        from solders.transaction import Transaction
        tx = Transaction.new(msg, [wm.solana_keypair])
        print(f"\n🔐 Signature test: PASS (keypair can sign transactions)")
    except Exception as e:
        print(f"\n❌ Signature test FAILED: {e}")
        return 1

    print(f"\n{'='*50}")
    print(f"🚀 Wallet ready! Run: python orchestrator.py")
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        exit_code = 0
    sys.exit(exit_code)
