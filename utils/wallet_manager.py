"""
wallet_manager.py
=================
Non-custodial wallet manager.
- Generates / loads Solana + EVM wallets from MULTIPLE input formats.
- NEVER exposes private keys to the network or to other modules.
- All signing is performed locally.

Supported Solana wallet connection methods (in priority order):
  1. SOLANA_PRIVATE_KEY — a raw base58 private key (what Phantom exports as
     "Private Key" in Account settings). This is the MOST common way users
     connect their pump.fun wallet.
  2. SOLANA_KEYPAIR_JSON — a JSON array of 64 numbers (Solana CLI keypair
     format, also exportable from Phantom as "Export Private Key JSON").
  3. SOLANA_WALLET_SEED — a 12/24 word mnemonic seed phrase (BIP39).
  4. None of the above — generates an ephemeral keypair (no funds, for testing).

The bot checks env vars in that order and uses the first one it finds.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from utils.config_loader import Config
from utils.logger import setup_logger

log = setup_logger("wallet_manager")


class WalletManager:
    """
    Holds references to chain-specific signer objects.
    Signers themselves stay in memory only; nothing is serialized to disk.
    """

    def __init__(self) -> None:
        self.cfg = Config.get()
        self.solana_keypair = None      # solana.keypair.Keypair
        self.evm_accounts: dict[str, object] = {}  # chain -> eth_account.Account
        self._init_solana()
        self._init_evm()

    # ------------------------------------------------------------------
    # Solana — multi-format key loading
    # ------------------------------------------------------------------
    def _init_solana(self) -> None:
        if "solana" not in self.cfg["chains"]:
            return
        try:
            from solders.keypair import Keypair
            seed_env = self.cfg["chains"]["solana"]["dedicated_wallet_seed_env"]
            seed_phrase = self.cfg.env(seed_env, required=False)

            # Method 1: SOLANA_PRIVATE_KEY (base58 private key — most common for
            # Phantom/pump.fun wallet users)
            priv_key = self.cfg.env("SOLANA_PRIVATE_KEY", required=False)
            if priv_key and priv_key.strip():
                try:
                    import base58
                    raw = base58.b58decode(priv_key.strip())
                    if len(raw) == 64:
                        self.solana_keypair = Keypair.from_bytes(raw)
                        log.info("solana.wallet.loaded_via_private_key",
                                 pubkey=str(self.solana_keypair.pubkey()))
                        return
                except Exception as e:
                    log.warning("solana.wallet.private_key_parse_failed", error=str(e))

            # Method 2: SOLANA_KEYPAIR_JSON (JSON array of 64 numbers)
            keypair_json = self.cfg.env("SOLANA_KEYPAIR_JSON", required=False)
            if keypair_json and keypair_json.strip():
                try:
                    arr = json.loads(keypair_json)
                    if isinstance(arr, list) and len(arr) == 64:
                        self.solana_keypair = Keypair.from_bytes(bytes(arr))
                        log.info("solana.wallet.loaded_via_keypair_json",
                                 pubkey=str(self.solana_keypair.pubkey()))
                        return
                except Exception as e:
                    log.warning("solana.wallet.keypair_json_parse_failed", error=str(e))

            # Method 3: SOLANA_WALLET_SEED (mnemonic phrase)
            if seed_phrase:
                import mnemonic
                m = mnemonic.Mnemonic("english")
                seed = m.to_seed(seed_phrase)
                self.solana_keypair = Keypair.from_seed_and_derivation_path(
                    seed, "m/44'/501'/0'/0'"
                ) if hasattr(Keypair, "from_seed_and_derivation_path") else Keypair.from_seed(seed[:32])
                log.info("solana.wallet.loaded_via_seed_phrase",
                         pubkey=str(self.solana_keypair.pubkey()))
                return

            # No credentials — ephemeral
            log.warning("solana.wallet.seed_missing", msg="No Solana wallet credentials in env; generating ephemeral keypair (no funds).")
            self.solana_keypair = Keypair()
        except Exception as e:
            log.error("solana.wallet.init_failed", error=str(e))
            raise

    def solana_pubkey(self) -> str:
        if self.solana_keypair is None:
            raise RuntimeError("Solana wallet not initialized")
        return str(self.solana_keypair.pubkey())

    # ------------------------------------------------------------------
    # EVM (Base / Ethereum)
    # ------------------------------------------------------------------
    def _init_evm(self) -> None:
        from eth_account import Account
        for chain_name in ("base", "ethereum"):
            if chain_name not in self.cfg["chains"]:
                continue
            env_key = self.cfg["chains"][chain_name]["dedicated_wallet_priv_env"]
            priv = self.cfg.env(env_key, required=False)
            if not priv:
                log.warning(f"{chain_name}.wallet.privkey_missing")
                continue
            acct = Account.from_key(priv)
            self.evm_accounts[chain_name] = acct
            log.info(f"{chain_name}.wallet.loaded", address=acct.address)

    def evm_address(self, chain: str) -> str:
        if chain not in self.evm_accounts:
            raise RuntimeError(f"EVM wallet for chain '{chain}' not initialized")
        return self.evm_accounts[chain].address

    # ------------------------------------------------------------------
    # Safety helpers
    # ------------------------------------------------------------------
    def has_wallet(self, chain: str) -> bool:
        if chain == "solana":
            return self.solana_keypair is not None
        return chain in self.evm_accounts

    def redacted_summary(self) -> dict:
        out: dict = {}
        if self.solana_keypair is not None:
            out["solana"] = {"pubkey": self.solana_pubkey()}
        for chain, acct in self.evm_accounts.items():
            out[chain] = {"address": acct.address}
        return out
