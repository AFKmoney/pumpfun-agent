"""
pumpfun_parser.py
=================
Parses pump.fun program instructions to extract new token launches.

Pump.fun program ID: 6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P
Instruction discriminator for "create" (sha256("global:create")[0..8]):
    -> 0x18 0x1e 0xc9 0x0d 0xa2 0xb8 0x12 0x5e

Create instruction layout (after 8-byte discriminator):
    name:        string  (4-byte LE length + UTF-8 bytes)
    symbol:      string
    uri:         string
    mint:        Pubkey  (32 bytes)
    bonding_curve: Pubkey (32 bytes)
    user:        Pubkey  (32 bytes)

We use this both for:
- Parsing logsSubscribe events (need to fetch the tx afterwards)
- Decoding getTransaction parsed instructions
"""
from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Optional

import base58

PUMP_FUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"


def _discriminator(method: str) -> bytes:
    """Anchor discriminator = first 8 bytes of sha256('global:<method>')."""
    h = hashlib.sha256(f"global:{method}".encode()).digest()
    return h[:8]


CREATE_DISCRIMINATOR = _discriminator("create")
BUY_DISCRIMINATOR = _discriminator("buy")
SELL_DISCRIMINATOR = _discriminator("sell")


@dataclass
class PumpCreateEvent:
    name: str
    symbol: str
    uri: str
    mint: str
    bonding_curve: str
    user: str
    raw_metadata: Optional[dict] = None


def _read_string(buf: bytes, offset: int) -> tuple[str, int]:
    """Read a Borsh-style string (4-byte LE length + UTF-8). Returns (str, new_offset)."""
    if offset + 4 > len(buf):
        raise ValueError("Buffer too short for string length")
    length = int.from_bytes(buf[offset:offset + 4], "little")
    offset += 4
    if offset + length > len(buf):
        raise ValueError("Buffer too short for string data")
    s = buf[offset:offset + length].decode("utf-8", errors="replace")
    return s, offset + length


def _read_pubkey(buf: bytes, offset: int) -> tuple[str, int]:
    if offset + 32 > len(buf):
        raise ValueError("Buffer too short for pubkey")
    pk = base58.b58encode(buf[offset:offset + 32]).decode("ascii")
    return pk, offset + 32


def parse_create_instruction(data_b64_or_bytes) -> Optional[PumpCreateEvent]:
    """
    data: base64 string (as returned by Solana RPC) OR raw bytes.
    Returns PumpCreateEvent or None if not a create instruction.
    """
    if isinstance(data_b64_or_bytes, str):
        try:
            buf = base64.b64decode(data_b64_or_bytes)
        except Exception:
            return None
    elif isinstance(data_b64_or_bytes, (bytes, bytearray)):
        buf = bytes(data_b64_or_bytes)
    else:
        return None

    if len(buf) < 8:
        return None
    if buf[:8] != CREATE_DISCRIMINATOR:
        return None

    try:
        offset = 8
        name, offset = _read_string(buf, offset)
        symbol, offset = _read_string(buf, offset)
        uri, offset = _read_string(buf, offset)
        mint, offset = _read_pubkey(buf, offset)
        bonding_curve, offset = _read_pubkey(buf, offset)
        user, offset = _read_pubkey(buf, offset)
        return PumpCreateEvent(
            name=name, symbol=symbol, uri=uri,
            mint=mint, bonding_curve=bonding_curve, user=user,
        )
    except Exception:
        return None


async def fetch_create_event_from_signature(adapter, signature: str) -> Optional[PumpCreateEvent]:
    """
    Given a tx signature, fetch the parsed tx and find the pump.fun create instruction.
    Returns PumpCreateEvent if found, else None.
    """
    resp = await adapter._rpc("getTransaction", [
        signature,
        {"maxSupportedTransactionVersion": 0, "encoding": "base64"},
    ])
    tx = resp.get("result")
    if not tx:
        return None
    msg = tx.get("transaction", [None, None])[0] if isinstance(tx.get("transaction"), list) else None
    # Prefer the parsed meta if available
    instructions = []
    meta = tx.get("meta", {}) or {}
    inner = meta.get("innerInstructions", []) or []
    for ix in (tx.get("transaction", {}).get("message", {}).get("instructions", []) or []):
        instructions.append(ix)
    for inner_group in inner:
        for ix in inner_group.get("instructions", []):
            instructions.append(ix)

    for ix in instructions:
        prog_id = ix.get("programId") or ix.get("programIdIndex")
        # Resolve programIdIndex to actual key if needed
        if isinstance(prog_id, int):
            account_keys = tx.get("transaction", {}).get("message", {}).get("accountKeys", [])
            if prog_id < len(account_keys):
                prog_id = account_keys[prog_id]
            else:
                continue
        if prog_id != PUMP_FUN_PROGRAM:
            continue
        data = ix.get("data", "")
        ev = parse_create_instruction(data)
        if ev:
            ev.raw_metadata = {
                "signature": signature,
                "slot": tx.get("slot"),
                "block_time": tx.get("blockTime"),
            }
            return ev
    return None
