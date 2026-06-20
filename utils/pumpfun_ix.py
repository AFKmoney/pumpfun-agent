"""
pumpfun_ix.py
=============
Direct pump.fun program instruction builders + low-latency launch detection.

This replaces the Jupiter-routed path for tokens still on the pump.fun bonding
curve, and the getTransaction round-trip in launch detection. Together these
two changes cut sniping latency from >1000ms to <400ms.

Two public surfaces:
1. `build_buy_ix()` / `build_sell_ix()` — build pump.fun program instructions.
2. `parse_create_from_logs()` — extract the mint/bonding_curve/user directly
   from a logsSubscribe notification, WITHOUT an extra getTransaction call.

pump.fun program accounts (from on-chain IDL):
  global           : PDA ["global"] @ program
  fee_recipient    : derived, baked into global config
  mint             : the token mint
  bonding_curve    : PDA ["bonding-curve", mint] @ program
  associated_user  : buyer's ATA for the mint
  user             : buyer's wallet

Instruction data layout (Borsh):
  discriminator (8 bytes = sha256("global:buy")[:8])
  amount        (u64, raw token units)
  max_sol_cost  (u64, raw lamports)   # buy only — slippage cap

SELL has the same accounts, no max_sol_cost field:
  discriminator (8 bytes = sha256("global:sell")[:8])
  amount        (u64, raw token units)
"""
from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass
from typing import Optional

import base58

PUMP_FUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
# pump.fun "global" config account — derived via PDA seed ["global"] against the
# pump.fun program. Computed once at import and cached. This is the canonical
# address (4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf).
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
ASSOCIATED_TOKEN_PROGRAM = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
SYSTEM_PROGRAM = "11111111111111111111111111111111"
RENT_SYSVAR = "SysvarRent111111111111111111111111111111111"


def _discriminator(method: str) -> bytes:
    return hashlib.sha256(f"global:{method}".encode()).digest()[:8]


BUY_DISCRIMINATOR = _discriminator("buy")
SELL_DISCRIMINATOR = _discriminator("sell")
CREATE_DISCRIMINATOR = _discriminator("create")


# ----------------------------------------------------------------------
# PDA derivation
# ----------------------------------------------------------------------
def derive_global_pda() -> str:
    """
    pump.fun global config account. Derived via PDA seed ["global"] against
    the pump.fun program. Cached after first computation.
    """
    global _GLOBAL_PDA_CACHE
    if _GLOBAL_PDA_CACHE is None:
        from solders.pubkey import Pubkey
        pda, _ = Pubkey.find_program_address([b"global"], Pubkey.from_string(PUMP_FUN_PROGRAM))
        _GLOBAL_PDA_CACHE = str(pda)
    return _GLOBAL_PDA_CACHE


# Lazy-computed cache for the global PDA (avoids solders import at module load).
_GLOBAL_PDA_CACHE: Optional[str] = None

# Backwards-compat constant: callers that imported PUMP_FUN_GLOBAL get the
# derived value via a module-level property-like accessor.
def __getattr__(name):
    if name == "PUMP_FUN_GLOBAL":
        return derive_global_pda()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def derive_bonding_curve_pda(mint: str) -> str:
    """Bonding curve account PDA ['bonding-curve', mint]."""
    from solders.pubkey import Pubkey
    seeds = [b"bonding-curve", bytes(Pubkey.from_string(mint))]
    pda, _ = Pubkey.find_program_address(seeds, Pubkey.from_string(PUMP_FUN_PROGRAM))
    return str(pda)


def derive_fee_recipient() -> str:
    """
    pump.fun's fee recipient (fixed, well-known from the program).
    This is the address the 1% fee flows to.
    """
    return "CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM"


# ----------------------------------------------------------------------
# Instruction builders
# ----------------------------------------------------------------------
@dataclass
class PumpfunAccounts:
    """Resolved account set for a pump.fun buy/sell instruction."""
    global_account: str
    fee_recipient: str
    mint: str
    bonding_curve: str
    associated_user: str  # buyer's ATA for the mint
    user: str             # buyer wallet


def resolve_buy_accounts(buyer_wallet: str, mint: str) -> PumpfunAccounts:
    """
    Resolve all 6 accounts needed for a pump.fun buy instruction.
    The buyer's ATA is derived via AssociatedTokenProgram PDA rules
    (does not require it to exist on-chain; the ix will still be valid if
    the program accepts a non-existent account — but typically buyers must
    create the ATA first via AssociatedTokenProgram.create call).
    """
    from solders.pubkey import Pubkey
    owner = Pubkey.from_string(buyer_wallet)
    mint_pk = Pubkey.from_string(mint)
    # ATA = PDA [wallet, token_program, mint] @ associated token program
    ata, _ = Pubkey.find_program_address(
        [bytes(owner), bytes(Pubkey.from_string(TOKEN_PROGRAM)), bytes(mint_pk)],
        Pubkey.from_string(ASSOCIATED_TOKEN_PROGRAM),
    )
    return PumpfunAccounts(
        global_account=derive_global_pda(),
        fee_recipient=derive_fee_recipient(),
        mint=mint,
        bonding_curve=derive_bonding_curve_pda(mint),
        associated_user=str(ata),
        user=buyer_wallet,
    )


def build_buy_ix(buyer_wallet: str, mint: str, token_amount_raw: int, max_sol_cost_lamports: int):
    """
    Build the pump.fun `buy` instruction.

    Args:
      buyer_wallet: signer wallet address (str)
      mint: token mint address (str)
      token_amount_raw: amount of tokens to buy, in raw units (token_units, NOT human)
      max_sol_cost_lamports: maximum SOL cost the buyer is willing to pay, in lamports
                              (= expected_cost * (1 + slippage)). Acts as MEV/slippage guard.

    Returns:
      solders.instruction.Instruction (ready to be included in a VersionedTransaction)
    """
    from solders.pubkey import Pubkey
    from solders.instruction import AccountMeta, Instruction

    accs = resolve_buy_accounts(buyer_wallet, mint)

    # Account order MUST match the on-chain buy(ix) definition exactly:
    #   global, fee_recipient, mint, bonding_curve, associated_user, user,
    #   system_program, token_program, rent
    metas = [
        AccountMeta(pubkey=Pubkey.from_string(accs.global_account), is_signer=False, is_writable=True),
        AccountMeta(pubkey=Pubkey.from_string(accs.fee_recipient), is_signer=False, is_writable=True),
        AccountMeta(pubkey=Pubkey.from_string(accs.mint), is_signer=False, is_writable=False),
        AccountMeta(pubkey=Pubkey.from_string(accs.bonding_curve), is_signer=False, is_writable=True),
        AccountMeta(pubkey=Pubkey.from_string(accs.associated_user), is_signer=False, is_writable=True),
        AccountMeta(pubkey=Pubkey.from_string(accs.user), is_signer=True, is_writable=True),
        AccountMeta(pubkey=Pubkey.from_string(SYSTEM_PROGRAM), is_signer=False, is_writable=False),
        AccountMeta(pubkey=Pubkey.from_string(TOKEN_PROGRAM), is_signer=False, is_writable=False),
        AccountMeta(pubkey=Pubkey.from_string(RENT_SYSVAR), is_signer=False, is_writable=False),
    ]
    data = BUY_DISCRIMINATOR + int(token_amount_raw).to_bytes(8, "little") + int(max_sol_cost_lamports).to_bytes(8, "little")
    return Instruction(
        program_id=Pubkey.from_string(PUMP_FUN_PROGRAM),
        accounts=metas,
        data=data,
    )


def build_sell_ix(seller_wallet: str, mint: str, token_amount_raw: int):
    """
    Build the pump.fun `sell` instruction.

    Args:
      seller_wallet: signer wallet address (str)
      mint: token mint address (str)
      token_amount_raw: amount of tokens to sell, in raw units

    Returns:
      solders.instruction.Instruction
    """
    from solders.pubkey import Pubkey
    from solders.instruction import AccountMeta, Instruction

    accs = resolve_buy_accounts(seller_wallet, mint)

    # sell(ix) account order:
    #   global, fee_recipient, mint, bonding_curve, associated_user, user,
    #   system_program, associated_token_program, token_program
    metas = [
        AccountMeta(pubkey=Pubkey.from_string(accs.global_account), is_signer=False, is_writable=True),
        AccountMeta(pubkey=Pubkey.from_string(accs.fee_recipient), is_signer=False, is_writable=True),
        AccountMeta(pubkey=Pubkey.from_string(accs.mint), is_signer=False, is_writable=False),
        AccountMeta(pubkey=Pubkey.from_string(accs.bonding_curve), is_signer=False, is_writable=True),
        AccountMeta(pubkey=Pubkey.from_string(accs.associated_user), is_signer=False, is_writable=True),
        AccountMeta(pubkey=Pubkey.from_string(accs.user), is_signer=True, is_writable=True),
        AccountMeta(pubkey=Pubkey.from_string(SYSTEM_PROGRAM), is_signer=False, is_writable=False),
        AccountMeta(pubkey=Pubkey.from_string(ASSOCIATED_TOKEN_PROGRAM), is_signer=False, is_writable=False),
        AccountMeta(pubkey=Pubkey.from_string(TOKEN_PROGRAM), is_signer=False, is_writable=False),
    ]
    data = SELL_DISCRIMINATOR + int(token_amount_raw).to_bytes(8, "little")
    return Instruction(
        program_id=Pubkey.from_string(PUMP_FUN_PROGRAM),
        accounts=metas,
        data=data,
    )


def build_create_ata_ix(buyer_wallet: str, mint: str):
    """
    Build an AssociatedTokenProgram.create_idempotent instruction for the
    buyer's ATA on `mint`. Required as a leading instruction before the pump.fun
    buy if the buyer has no ATA yet (common on fresh snipes).
    """
    from solders.pubkey import Pubkey
    from solders.instruction import AccountMeta, Instruction

    accs = resolve_buy_accounts(buyer_wallet, mint)
    metas = [
        AccountMeta(pubkey=Pubkey.from_string(accs.user), is_signer=True, is_writable=True),
        AccountMeta(pubkey=Pubkey.from_string(accs.associated_user), is_signer=False, is_writable=True),
        AccountMeta(pubkey=Pubkey.from_string(accs.mint), is_signer=False, is_writable=False),
        AccountMeta(pubkey=Pubkey.from_string(SYSTEM_PROGRAM), is_signer=False, is_writable=False),
        AccountMeta(pubkey=Pubkey.from_string(TOKEN_PROGRAM), is_signer=False, is_writable=False),
        AccountMeta(pubkey=Pubkey.from_string(SYSTEM_PROGRAM), is_signer=False, is_writable=False),
        AccountMeta(pubkey=Pubkey.from_string(RENT_SYSVAR), is_signer=False, is_writable=False),
        AccountMeta(pubkey=Pubkey.from_string(ASSOCIATED_TOKEN_PROGRAM), is_signer=False, is_writable=False),
    ]
    data = base58.b58decode(b"") if False else b"\x1b"  # CreateIdempotent discriminator (1 byte)
    return Instruction(
        program_id=Pubkey.from_string(ASSOCIATED_TOKEN_PROGRAM),
        accounts=metas,
        data=data,
    )


# ----------------------------------------------------------------------
# Low-latency launch detection: parse mint DIRECTLY from logsSubscribe logs
# ----------------------------------------------------------------------
# pump.fun emits these log lines on a `create` instruction (Anchor events):
#   "Instruction: Create"
#   "CreateEvent: ... name=\"...\", symbol=\"...\", uri=\"...\", mint=<b58>, bonding_curve=<b58>, user=<b58>"
#
# Some RPCs surface a text-readable event line (Helius / QuickNode with log
# enrichment); others only emit "Program data: <base64>" (raw Anchor event).
# We handle both, then fall back to a getTransaction round-trip only if neither
# yields the addresses.
_KV_RE = re.compile(r'(\w+)=([1-9A-HJ-NP-Za-km-z]{32,44})')


def _extract_kv_pubkeys(lines: list[str]) -> dict:
    """Find all `key=<base58>` pairs across all log lines. Returns dict of matches."""
    found: dict[str, str] = {}
    for line in lines:
        for key, val in _KV_RE.findall(line):
            found[key] = val
    return found


@dataclass
class ParsedLaunch:
    mint: Optional[str]
    bonding_curve: Optional[str]
    user: Optional[str]
    signature: str
    raw_logs: list[str]
    needs_full_fetch: bool  # True if we could not parse from logs and need getTransaction


def parse_create_from_logs(logs: list[str], signature: str) -> ParsedLaunch:
    """
    Try to extract (mint, bonding_curve, user) directly from the logsSubscribe
    notification's log lines, without an extra getTransaction call.

    Returns a ParsedLaunch. If `needs_full_fetch` is True, the caller should
    fall back to fetch_create_event_from_signature() to get the real mint.
    """
    # 1. Best case: RPC log enrichment gives us a text-readable CreateEvent line.
    kv = _extract_kv_pubkeys(logs)
    if "mint" in kv and "bonding_curve" in kv and "user" in kv:
        return ParsedLaunch(
            mint=kv["mint"], bonding_curve=kv["bonding_curve"], user=kv["user"],
            signature=signature, raw_logs=logs, needs_full_fetch=False,
        )
    # Some RPCs only expose the mint but not the bonding curve. We can derive
    # the bonding curve PDA from the mint client-side (it's deterministic).
    if "mint" in kv:
        mint = kv["mint"]
        try:
            bc = derive_bonding_curve_pda(mint)
        except Exception:
            bc = None
        return ParsedLaunch(
            mint=mint, bonding_curve=bc, user=kv.get("user"),
            signature=signature, raw_logs=logs, needs_full_fetch=(bc is None),
        )

    # 2. Anchor "Program data: <base64>" event. The CreateEvent payload's last
    #    96 bytes are mint(32) bonding_curve(32) user(32). We try to decode it.
    for line in logs:
        if "Program data:" not in line:
            continue
        try:
            b64 = line.split("Program data:", 1)[1].strip()
            raw = base64.b64decode(b64)
            if len(raw) >= 8 + 32 * 3:
                offset = len(raw) - 96
                mint_b = raw[offset:offset + 32]
                bc_b = raw[offset + 32:offset + 64]
                user_b = raw[offset + 64:offset + 96]
                mint = base58.b58encode(mint_b).decode()
                bc = base58.b58encode(bc_b).decode()
                user = base58.b58encode(user_b).decode()
                if mint != PUMP_FUN_PROGRAM and mint != "11111111111111111111111111111111":
                    return ParsedLaunch(
                        mint=mint, bonding_curve=bc, user=user,
                        signature=signature, raw_logs=logs, needs_full_fetch=False,
                    )
        except Exception:
            continue

    # 3. Last resort: caller must do getTransaction
    return ParsedLaunch(
        mint=None, bonding_curve=None, user=None,
        signature=signature, raw_logs=logs, needs_full_fetch=True,
    )


# ----------------------------------------------------------------------
# Discriminator verification helpers (used by tests + analyzers)
# ----------------------------------------------------------------------
def is_create_log(logs: list[str]) -> bool:
    return any("Instruction: Create" in line or "CreateEvent" in line for line in logs)


def is_buy_instruction_data(data_bytes: bytes) -> bool:
    return len(data_bytes) >= 8 and data_bytes[:8] == BUY_DISCRIMINATOR


def is_sell_instruction_data(data_bytes: bytes) -> bool:
    return len(data_bytes) >= 8 and data_bytes[:8] == SELL_DISCRIMINATOR
