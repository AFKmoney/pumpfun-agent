"""
wallet_manager.py
=================
Non-custodial wallet manager.
- Generates / loads Solana + EVM wallets.
- NEVER exposes private keys to the network or to other modules.
- All signing is performed locally.
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
    # Solana
    # ------------------------------------------------------------------
    def _init_solana(self) -> None:
        if "solana" not in self.cfg["chains"]:
            return
        try:
            from solders.keypair import Keypair
            seed_env = self.cfg["chains"]["solana"]["dedicated_wallet_seed_env"]
            seed_phrase = self.cfg.env(seed_env, required=False)
            if not seed_phrase:
                log.warning("solana.wallet.seed_missing", msg="No Solana seed in env; generating ephemeral keypair (no funds).")
                self.solana_keypair = Keypair()
                return
            # Convert mnemonic -> seed -> keypair
            # NOTE: For production use a hardware wallet or remote signer.
            import mnemonic
            m = mnemonic.Mnemonic("english")
            seed = m.to_seed(seed_phrase)
            self.solana_keypair = Keypair.from_seed_and_derivation_path(
                seed, "m/44'/501'/0'/0'"
            ) if hasattr(Keypair, "from_seed_and_derivation_path") else Keypair.from_seed(seed[:32])
            log.info("solana.wallet.loaded", pubkey=str(self.solana_keypair.pubkey()))
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
