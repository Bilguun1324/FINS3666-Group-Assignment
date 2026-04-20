from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests


SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
RAYDIUM_CPMM_PROGRAM = "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C"
SOLANA_RPC_ENDPOINTS = (
    "https://solana-rpc.publicnode.com",
    "https://api.mainnet-beta.solana.com",
)
DEFAULT_WINDOW_DAYS = 7
STATE_FREQUENCY = "1s"
STALE_AFTER_MINUTES = 10
RAYDIUM_SCAN_PAGES = 30
RAYDIUM_PAGE_SIZE = 100
METEORA_DAMM_V2_PAGE_SIZE = 50


@dataclass(frozen=True)
class TokenSpec:
    symbol: str
    address: str
    decimals: int


@dataclass(frozen=True)
class SolanaPoolSpec:
    dex: str
    source: str
    pool_address: str
    pool_name: str
    pool_family: str
    pool_type: str
    base_token: TokenSpec
    quote_token: TokenSpec
    vault_base: str | None
    vault_quote: str | None
    tvl_usd: float | None
    volume_24h_usd: float | None
    fee_rate_floor: float | None
    fee_model: str
    reconstructable: bool
    reconstructable_reason: str
    metadata: dict[str, Any]

    @property
    def slug(self) -> str:
        return f"{self.dex}_{self.pool_address}"


@dataclass(frozen=True)
class SolanaMarketSpec:
    slug: str = "solana_research"
    chain_name: str = "Solana"
    pair_label: str = "SOL/USDC"
    base_token: TokenSpec = TokenSpec("SOL", SOL_MINT, 9)
    quote_token: TokenSpec = TokenSpec("USDC", USDC_MINT, 6)
    sample_window_days: int = DEFAULT_WINDOW_DAYS
    frequency: str = STATE_FREQUENCY
    stale_after_minutes: int = STALE_AFTER_MINUTES
    trade_sizes_base: tuple[float, ...] = (0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01)
    primary_trade_size_base: float = 0.001


MARKET_SPEC = SolanaMarketSpec()


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "notebooks").exists() and (candidate / "data").exists():
            return candidate
    return current


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(parsed) else parsed


def _safe_int(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ensure_utc_timestamp(value: datetime | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


class SolanaRpcClient:
    def __init__(self, endpoints: Iterable[str] = SOLANA_RPC_ENDPOINTS) -> None:
        self.endpoints = list(endpoints)
        self._index = 0
        self.session = requests.Session()

    def _endpoint(self) -> str:
        return self.endpoints[self._index % len(self.endpoints)]

    def _rotate(self) -> None:
        self._index = (self._index + 1) % len(self.endpoints)

    def call(
        self,
        method: str,
        params: list[Any],
        *,
        timeout: int = 120,
        max_retries: int = 8,
    ) -> Any:
        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(max_retries):
            endpoint = self._endpoint()
            try:
                response = self.session.post(
                    endpoint,
                    json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                    timeout=timeout,
                )
                if response.status_code == 429:
                    raise requests.HTTPError(f"429 from {endpoint}", response=response)
                response.raise_for_status()
                payload = response.json()
                if "error" in payload:
                    code = payload["error"].get("code")
                    if code == 429:
                        raise RuntimeError(f"RPC 429 from {endpoint}: {payload['error']}")
                    raise RuntimeError(f"RPC error from {endpoint}: {payload['error']}")
                return payload["result"]
            except Exception as exc:  # pragma: no cover - network failures are non-deterministic
                last_error = exc
                if attempt == max_retries - 1:
                    break
                self._rotate()
                time.sleep(delay)
                delay = min(delay * 1.75, 15.0)
        raise RuntimeError(f"Solana RPC call failed for {method}") from last_error

    def batch_get_block(self, slots: list[int]) -> list[dict[str, Any]]:
        if not slots:
            return []
        batch = [
            {
                "jsonrpc": "2.0",
                "id": index + 1,
                "method": "getBlock",
                "params": [
                    slot,
                    {
                        "encoding": "jsonParsed",
                        "transactionDetails": "full",
                        "rewards": False,
                        "maxSupportedTransactionVersion": 0,
                    },
                ],
            }
            for index, slot in enumerate(slots)
        ]
        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(8):
            endpoint = self._endpoint()
            try:
                response = self.session.post(endpoint, json=batch, timeout=180)
                if response.status_code == 429:
                    raise requests.HTTPError(f"429 from {endpoint}", response=response)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise RuntimeError(f"Unexpected batch payload from {endpoint}: {payload}")
                errors = [item for item in payload if item.get("error")]
                if errors:
                    rate_limited = any(item["error"].get("code") == 429 for item in errors)
                    if rate_limited:
                        raise RuntimeError(f"RPC batch 429 from {endpoint}")
                    raise RuntimeError(f"RPC batch error from {endpoint}: {errors[0]['error']}")
                return sorted(payload, key=lambda item: item["id"])
            except Exception as exc:  # pragma: no cover - network failures are non-deterministic
                last_error = exc
                if attempt == 7:
                    break
                self._rotate()
                time.sleep(delay)
                delay = min(delay * 1.75, 15.0)
        if len(slots) == 1:
            slot = slots[0]
            result = self.call(
                "getBlock",
                [
                    slot,
                    {
                        "encoding": "jsonParsed",
                        "transactionDetails": "full",
                        "rewards": False,
                        "maxSupportedTransactionVersion": 0,
                    },
                ],
                timeout=180,
                max_retries=12,
            )
            return [{"id": 1, "result": result}]

        # Fall back to per-slot requests when batch limits are exhausted.
        serial_payload: list[dict[str, Any]] = []
        for index, slot in enumerate(slots, start=1):
            result = self.call(
                "getBlock",
                [
                    slot,
                    {
                        "encoding": "jsonParsed",
                        "transactionDetails": "full",
                        "rewards": False,
                        "maxSupportedTransactionVersion": 0,
                    },
                ],
                timeout=180,
                max_retries=12,
            )
            serial_payload.append({"id": index, "result": result})
            time.sleep(0.15)
        return serial_payload


def _requests_get_json(url: str, *, params: dict[str, Any] | list[tuple[str, Any]] | None = None) -> Any:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def _token_from_payload(payload: dict[str, Any]) -> TokenSpec:
    return TokenSpec(
        symbol=payload.get("symbol") or payload.get("name") or payload["address"],
        address=payload["address"],
        decimals=int(payload["decimals"]),
    )


def _raydium_pair_rows() -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for page in range(1, RAYDIUM_SCAN_PAGES + 1):
        payload = _requests_get_json(
            "https://api-v3.raydium.io/pools/info/list",
            params={
                "poolType": "all",
                "poolSortField": "default",
                "sortType": "desc",
                "pageSize": RAYDIUM_PAGE_SIZE,
                "page": page,
            },
        )
        rows = payload["data"]["data"]
        if not rows:
            break
        for row in rows:
            pair = {row.get("mintA", {}).get("address"), row.get("mintB", {}).get("address")}
            if pair == {SOL_MINT, USDC_MINT}:
                matches.append(row)
    return matches


def discover_raydium_candidates() -> list[SolanaPoolSpec]:
    rows = _raydium_pair_rows()
    candidates: list[SolanaPoolSpec] = []
    detail_cache: dict[str, dict[str, Any] | None] = {}
    for row in rows:
        pool_id = row["id"]
        if pool_id not in detail_cache:
            detail_payload = _requests_get_json(f"https://api-v3.raydium.io/pools/key/ids?ids={pool_id}")
            detail_cache[pool_id] = detail_payload["data"][0]
        detail = detail_cache[pool_id]
        pooltype = row.get("pooltype") or []
        reconstructable = False
        fee_model = "blocked"
        reason = "Unknown Raydium pool type."
        vault_base = None
        vault_quote = None
        fee_rate_floor = _safe_float(row.get("feeRate"))
        if detail is not None and detail.get("vault"):
            mint_a_address = detail["mintA"]["address"]
            vault_a = detail["vault"]["A"]
            vault_b = detail["vault"]["B"]
            if mint_a_address == SOL_MINT:
                vault_base, vault_quote = vault_a, vault_b
            else:
                vault_base, vault_quote = vault_b, vault_a
        if "Cpmm" in pooltype or row.get("programId") == RAYDIUM_CPMM_PROGRAM:
            reconstructable = True
            fee_model = "exact_fixed_fee"
            reason = "Raydium CPMM exposes direct pool vaults and fixed trade fee metadata."
        elif "OpenBookMarket" in pooltype:
            reason = "Raydium Standard AMM pools share inventory with OpenBook, so vault balances alone are incomplete."
        elif "Clmm" in pooltype or row.get("type") == "Concentrated":
            reason = "Raydium concentrated liquidity pools require concentrated-liquidity state, not just token vault balances."
        candidates.append(
            SolanaPoolSpec(
                dex=f"raydium_{'cpmm' if reconstructable else row.get('type', 'pool').lower()}",
                source="raydium_v3_api",
                pool_address=pool_id,
                pool_name=f"Raydium {row.get('type', 'Pool')} SOL/USDC",
                pool_family="raydium",
                pool_type="|".join(pooltype) if pooltype else str(row.get("type")),
                base_token=TokenSpec("SOL", SOL_MINT, 9),
                quote_token=TokenSpec("USDC", USDC_MINT, 6),
                vault_base=vault_base,
                vault_quote=vault_quote,
                tvl_usd=_safe_float(row.get("tvl")),
                volume_24h_usd=_safe_float((row.get("day") or {}).get("volumeQuote")),
                fee_rate_floor=fee_rate_floor,
                fee_model=fee_model,
                reconstructable=reconstructable,
                reconstructable_reason=reason,
                metadata={
                    "pooltype": pooltype,
                    "programId": row.get("programId"),
                    "detail": detail,
                },
            )
        )
    return candidates


def discover_orca_candidates() -> list[SolanaPoolSpec]:
    payload = _requests_get_json("https://api.orca.so/v2/solana/pools/search", params={"q": "SOL-USDC"})
    candidates: list[SolanaPoolSpec] = []
    for row in payload["data"]:
        pair = {row.get("tokenMintA"), row.get("tokenMintB")}
        if pair != {SOL_MINT, USDC_MINT}:
            continue
        if row.get("tokenMintA") == SOL_MINT:
            vault_base, vault_quote = row.get("tokenVaultA"), row.get("tokenVaultB")
        else:
            vault_base, vault_quote = row.get("tokenVaultB"), row.get("tokenVaultA")
        candidates.append(
            SolanaPoolSpec(
                dex="orca_whirlpool",
                source="orca_public_api",
                pool_address=row["address"],
                pool_name=f"Orca {row.get('poolType', 'pool')} SOL/USDC",
                pool_family="orca",
                pool_type=row.get("poolType", "whirlpool"),
                base_token=TokenSpec("SOL", SOL_MINT, 9),
                quote_token=TokenSpec("USDC", USDC_MINT, 6),
                vault_base=vault_base,
                vault_quote=vault_quote,
                tvl_usd=_safe_float(row.get("tvl")),
                volume_24h_usd=_safe_float(row.get("volume24h")),
                fee_rate_floor=None,
                fee_model="blocked",
                reconstructable=False,
                reconstructable_reason="Orca SOL/USDC pools are Whirlpool/Splash-style CLMM pools, not notebook-02-style reserve pools.",
                metadata=row,
            )
        )
    return candidates


def discover_meteora_damm_v1_candidates() -> list[SolanaPoolSpec]:
    pair = f"{USDC_MINT}-{SOL_MINT}"
    payload = _requests_get_json(
        "https://damm-api.meteora.ag/pools/search",
        params={
            "page": 0,
            "size": 50,
            "include_pool_token_pairs": pair,
            "sort_key": "volume",
            "order_by": "desc",
        },
    )
    candidates: list[SolanaPoolSpec] = []
    seen_vault_sets: dict[tuple[str | None, str | None], int] = {}
    for row in payload["data"]:
        mints = row.get("pool_token_mints") or []
        if set(mints) != {SOL_MINT, USDC_MINT}:
            continue
        vaults = row.get("vaults") or [None, None]
        if len(vaults) < 2:
            vaults = [None, None]
        vault_key = tuple(vaults[:2])
        seen_vault_sets[vault_key] = seen_vault_sets.get(vault_key, 0) + 1
        candidates.append(
            SolanaPoolSpec(
                dex="meteora_damm_v1",
                source="meteora_damm_v1_api",
                pool_address=row["pool_address"],
                pool_name=row.get("pool_name") or "Meteora DAMM v1 SOL/USDC",
                pool_family="meteora_damm_v1",
                pool_type=row.get("pool_type", "unknown"),
                base_token=TokenSpec("SOL", SOL_MINT, 9),
                quote_token=TokenSpec("USDC", USDC_MINT, 6),
                vault_base=vaults[1] if mints and mints[0] == USDC_MINT else vaults[0],
                vault_quote=vaults[0] if mints and mints[0] == USDC_MINT else vaults[1],
                tvl_usd=_safe_float(row.get("pool_tvl")),
                volume_24h_usd=_safe_float(row.get("trading_volume")),
                fee_rate_floor=_safe_float(row.get("total_fee_pct")) / 100.0 if _safe_float(row.get("total_fee_pct")) is not None else None,
                fee_model="blocked_shared_dynamic_vaults",
                reconstructable=False,
                reconstructable_reason="Meteora DAMM v1 routes pool assets through shared Dynamic Vaults, so vault balances are not per-pool reserves.",
                metadata=row,
            )
        )
    repeated = {vaults for vaults, count in seen_vault_sets.items() if count > 1}
    for candidate in candidates:
        candidate.metadata["shared_vault_pair_reused"] = (candidate.vault_base, candidate.vault_quote) in repeated
    return candidates


def discover_meteora_damm_v2_candidates() -> list[SolanaPoolSpec]:
    payload = _requests_get_json(
        "https://damm-v2.datapi.meteora.ag/pools",
        params={
            "page": 1,
            "page_size": METEORA_DAMM_V2_PAGE_SIZE,
            "query": "SOL USDC",
            "sort_by": "tvl:desc",
        },
    )
    candidates: list[SolanaPoolSpec] = []
    for row in payload["data"]:
        pair = {row.get("token_x", {}).get("address"), row.get("token_y", {}).get("address")}
        if pair != {SOL_MINT, USDC_MINT}:
            continue
        token_x = row["token_x"]
        token_y = row["token_y"]
        if token_x["address"] == SOL_MINT:
            vault_base, vault_quote = row.get("vault_x"), row.get("vault_y")
        else:
            vault_base, vault_quote = row.get("vault_y"), row.get("vault_x")
        pool_config = row.get("pool_config") or {}
        concentrated = bool(pool_config.get("concentrated_liquidity"))
        dynamic_fee_initialized = bool(pool_config.get("dynamic_fee_initialized"))
        reconstructable = not concentrated
        fee_model = "lower_bound_base_fee" if reconstructable and dynamic_fee_initialized else "exact_base_fee"
        reason = (
            "Meteora DAMM v2 exposes direct pool vaults and uses a constant-product full-range pool."
            if reconstructable
            else "Meteora DAMM v2 concentrated pools require ranged-liquidity state, not just vault balances."
        )
        candidates.append(
            SolanaPoolSpec(
                dex="meteora_damm_v2",
                source="meteora_damm_v2_api",
                pool_address=row["address"],
                pool_name=row.get("name") or "Meteora DAMM v2 SOL/USDC",
                pool_family="meteora_damm_v2",
                pool_type="concentrated" if concentrated else "full_range",
                base_token=_token_from_payload(row["token_x"]) if row["token_x"]["address"] == SOL_MINT else _token_from_payload(row["token_y"]),
                quote_token=_token_from_payload(row["token_y"]) if row["token_x"]["address"] == SOL_MINT else _token_from_payload(row["token_x"]),
                vault_base=vault_base,
                vault_quote=vault_quote,
                tvl_usd=_safe_float(row.get("tvl")),
                volume_24h_usd=_safe_float((row.get("volume") or {}).get("24h")),
                fee_rate_floor=(_safe_float(pool_config.get("base_fee_pct")) or 0.0) / 100.0,
                fee_model=fee_model,
                reconstructable=reconstructable,
                reconstructable_reason=reason,
                metadata=row,
            )
        )
    return candidates


def discover_official_venue_catalog() -> pd.DataFrame:
    candidates = (
        discover_raydium_candidates()
        + discover_orca_candidates()
        + discover_meteora_damm_v1_candidates()
        + discover_meteora_damm_v2_candidates()
    )
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rows.append(
            {
                "dex": candidate.dex,
                "source": candidate.source,
                "pool_address": candidate.pool_address,
                "pool_name": candidate.pool_name,
                "pool_family": candidate.pool_family,
                "pool_type": candidate.pool_type,
                "tvl_usd": candidate.tvl_usd,
                "volume_24h_usd": candidate.volume_24h_usd,
                "fee_rate_floor": candidate.fee_rate_floor,
                "fee_model": candidate.fee_model,
                "reconstructable": candidate.reconstructable,
                "reconstructable_reason": candidate.reconstructable_reason,
                "vault_base": candidate.vault_base,
                "vault_quote": candidate.vault_quote,
                "metadata": json.dumps(candidate.metadata, default=str),
            }
        )
    return pd.DataFrame(rows).sort_values(["reconstructable", "volume_24h_usd", "tvl_usd"], ascending=[False, False, False]).reset_index(drop=True)


def select_reconstructable_pools(catalog: pd.DataFrame) -> list[SolanaPoolSpec]:
    if catalog.empty:
        return []
    rows = catalog.to_dict(orient="records")
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["reconstructable"]:
            by_family.setdefault(row["pool_family"], []).append(row)

    selected_rows: list[dict[str, Any]] = []
    if by_family.get("raydium"):
        selected_rows.append(sorted(by_family["raydium"], key=lambda row: (row["volume_24h_usd"] or 0.0, row["tvl_usd"] or 0.0), reverse=True)[0])
    if by_family.get("meteora_damm_v2"):
        selected_rows.append(sorted(by_family["meteora_damm_v2"], key=lambda row: (row["volume_24h_usd"] or 0.0, row["tvl_usd"] or 0.0), reverse=True)[0])

    selected_specs: list[SolanaPoolSpec] = []
    for row in selected_rows:
        metadata = json.loads(row["metadata"])
        selected_specs.append(
            SolanaPoolSpec(
                dex=row["dex"],
                source=row["source"],
                pool_address=row["pool_address"],
                pool_name=row["pool_name"],
                pool_family=row["pool_family"],
                pool_type=row["pool_type"],
                base_token=MARKET_SPEC.base_token,
                quote_token=MARKET_SPEC.quote_token,
                vault_base=row["vault_base"],
                vault_quote=row["vault_quote"],
                tvl_usd=_safe_float(row["tvl_usd"]),
                volume_24h_usd=_safe_float(row["volume_24h_usd"]),
                fee_rate_floor=_safe_float(row["fee_rate_floor"]),
                fee_model=row["fee_model"],
                reconstructable=bool(row["reconstructable"]),
                reconstructable_reason=row["reconstructable_reason"],
                metadata=metadata,
            )
        )
    return selected_specs


def _collect_signatures_for_address(
    rpc: SolanaRpcClient,
    address: str,
    *,
    window_start: pd.Timestamp,
) -> list[dict[str, Any]]:
    window_start_unix = int(window_start.timestamp())
    before: str | None = None
    collected: list[dict[str, Any]] = []
    older_seed: dict[str, Any] | None = None
    while True:
        params: list[Any] = [address, {"limit": 1000}]
        if before:
            params[1]["before"] = before
        rows = rpc.call("getSignaturesForAddress", params)
        if not rows:
            break
        stop = False
        for row in rows:
            block_time = row.get("blockTime")
            if block_time is None:
                continue
            if block_time >= window_start_unix:
                collected.append(row)
            else:
                older_seed = row
                stop = True
                break
        if stop:
            break
        before = rows[-1]["signature"]
        time.sleep(0.15)
    if older_seed is not None:
        collected.append(older_seed)
    return collected


def collect_pool_signature_rows(
    rpc: SolanaRpcClient,
    pool: SolanaPoolSpec,
    *,
    window_start: pd.Timestamp,
) -> pd.DataFrame:
    signature_rows: list[dict[str, Any]] = []
    for address in [pool.vault_base, pool.vault_quote]:
        if not address:
            continue
        signature_rows.extend(_collect_signatures_for_address(rpc, address, window_start=window_start))
    if not signature_rows:
        return pd.DataFrame(columns=["signature", "slot", "blockTime", "err", "memo", "confirmationStatus"])
    frame = (
        pd.DataFrame(signature_rows)
        .drop_duplicates("signature")
        .sort_values(["slot", "blockTime"], ascending=[True, True])
        .reset_index(drop=True)
    )
    frame["timestamp"] = pd.to_datetime(frame["blockTime"], unit="s", utc=True)
    keep_columns = ["signature", "slot", "blockTime", "timestamp", "confirmationStatus"]
    return frame[[column for column in keep_columns if column in frame.columns]].copy()


def _post_balance_lookup(tx: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    account_keys = tx["transaction"]["message"]["accountKeys"]
    keys: list[str] = []
    for entry in account_keys:
        keys.append(entry["pubkey"] if isinstance(entry, dict) else entry)
    post = (tx.get("meta") or {}).get("postTokenBalances") or []
    return keys, post


def extract_pool_events_from_slots(
    rpc: SolanaRpcClient,
    pool: SolanaPoolSpec,
    slot_rows: pd.DataFrame,
) -> pd.DataFrame:
    if slot_rows.empty:
        return pd.DataFrame()
    slots = slot_rows["slot"].dropna().astype(int).sort_values().unique().tolist()
    events: list[dict[str, Any]] = []
    batch_size = 10 if pool.dex == "raydium_cpmm" else 3
    for start in range(0, len(slots), batch_size):
        batch_slots = slots[start : start + batch_size]
        payload = rpc.batch_get_block(batch_slots)
        for item in payload:
            block = item["result"]
            if not block or block.get("blockTime") is None:
                continue
            block_time = pd.to_datetime(block["blockTime"], unit="s", utc=True)
            block_slot = int(block["parentSlot"]) + 1 if block.get("parentSlot") is not None else batch_slots[item["id"] - 1]
            for tx in block["transactions"]:
                meta = tx.get("meta") or {}
                if meta.get("err") is not None:
                    continue
                keys, post_token_balances = _post_balance_lookup(tx)
                balances: dict[str, dict[str, Any]] = {}
                for balance in post_token_balances:
                    index = balance.get("accountIndex")
                    if index is None or index >= len(keys):
                        continue
                    pubkey = keys[index]
                    if pubkey not in (pool.vault_base, pool.vault_quote):
                        continue
                    balances[pubkey] = balance
                if pool.vault_base not in balances or pool.vault_quote not in balances:
                    continue
                base_amount_raw = balances[pool.vault_base]["uiTokenAmount"]["amount"]
                quote_amount_raw = balances[pool.vault_quote]["uiTokenAmount"]["amount"]
                base_amount = int(base_amount_raw) / (10 ** pool.base_token.decimals)
                quote_amount = int(quote_amount_raw) / (10 ** pool.quote_token.decimals)
                if base_amount <= 0 or quote_amount <= 0:
                    continue
                signature = tx["transaction"]["signatures"][0]
                events.append(
                    {
                        "timestamp": block_time,
                        "slot": block_slot,
                        "signature": signature,
                        "dex": pool.dex,
                        "pool_address": pool.pool_address,
                        "reserve_base_raw": int(base_amount_raw),
                        "reserve_quote_raw": int(quote_amount_raw),
                        "reserve_base": base_amount,
                        "reserve_quote": quote_amount,
                        "mid_price_quote_per_base": quote_amount / base_amount,
                        "network_fee_lamports": int(meta.get("fee", 0)),
                        "fee_rate_floor": pool.fee_rate_floor,
                        "fee_model": pool.fee_model,
                    }
                )
        if start == 0 or (start // batch_size + 1) % 20 == 0 or start + batch_size >= len(slots):
            print(
                f"[solana] {pool.dex}: processed {min(start + batch_size, len(slots))}/{len(slots)} slots",
                flush=True,
            )
        time.sleep(0.05 if pool.dex == "raydium_cpmm" else 0.12)
    if not events:
        return pd.DataFrame()
    out = pd.DataFrame(events).drop_duplicates(["signature", "dex"]).sort_values(["timestamp", "slot", "signature"]).reset_index(drop=True)
    return out


def build_pool_state(events_df: pd.DataFrame, *, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for (dex, pool_address), group in events_df.groupby(["dex", "pool_address"], as_index=False):
        state = group[
            [
                "timestamp",
                "slot",
                "signature",
                "dex",
                "pool_address",
                "reserve_base",
                "reserve_quote",
                "mid_price_quote_per_base",
                "network_fee_lamports",
                "fee_rate_floor",
                "fee_model",
            ]
        ].copy()
        state["last_event_timestamp"] = state["timestamp"]
        state = (
            state.set_index("timestamp")
            .resample(MARKET_SPEC.frequency)
            .last()
            .ffill()
            .rename_axis("timestamp")
            .reset_index()
        )
        state = state[(state["timestamp"] >= window_start) & (state["timestamp"] <= window_end)].copy()
        state["dex"] = dex
        state["pool_address"] = pool_address
        state["stale_state"] = (state["timestamp"] - state["last_event_timestamp"]) > pd.Timedelta(minutes=MARKET_SPEC.stale_after_minutes)
        frames.append(state)
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "dex"]).reset_index(drop=True)


def amount_out_for_exact_in(amount_in: float, reserve_in: np.ndarray, reserve_out: np.ndarray, fee_rate: float) -> np.ndarray:
    reserve_in = np.asarray(reserve_in, dtype=float)
    reserve_out = np.asarray(reserve_out, dtype=float)
    result = np.full(reserve_in.shape, np.nan, dtype=float)
    if amount_in <= 0:
        return result
    valid = (reserve_in > 0) & (reserve_out > 0)
    effective_in = amount_in * (1.0 - fee_rate)
    result[valid] = reserve_out[valid] * effective_in / (reserve_in[valid] + effective_in)
    return result


def amount_in_for_exact_out(amount_out: float, reserve_in: np.ndarray, reserve_out: np.ndarray, fee_rate: float) -> np.ndarray:
    reserve_in = np.asarray(reserve_in, dtype=float)
    reserve_out = np.asarray(reserve_out, dtype=float)
    result = np.full(reserve_in.shape, np.nan, dtype=float)
    if amount_out <= 0 or fee_rate >= 1.0:
        return result
    valid = (reserve_in > 0) & (reserve_out > amount_out)
    raw_required = reserve_in[valid] * amount_out / (reserve_out[valid] - amount_out)
    result[valid] = raw_required / (1.0 - fee_rate)
    return result


def build_fee_bounds(pool_events_df: pd.DataFrame) -> dict[str, Any]:
    nonzero = pool_events_df["network_fee_lamports"][pool_events_df["network_fee_lamports"] > 0]
    if nonzero.empty:
        return {
            "network_fee_lamports_floor": 5000,
            "network_fee_lamports_p95": 5000,
        }
    return {
        "network_fee_lamports_floor": int(max(5000, nonzero.min())),
        "network_fee_lamports_p95": int(np.nanpercentile(nonzero, 95)),
    }


def _evaluate_direction(
    *,
    buy_reserve_base: np.ndarray,
    buy_reserve_quote: np.ndarray,
    sell_reserve_base: np.ndarray,
    sell_reserve_quote: np.ndarray,
    buy_fee_rate_floor: float,
    sell_fee_rate_floor: float,
    trade_size_base: float,
    network_fee_quote_floor: np.ndarray,
    network_fee_quote_p95: np.ndarray,
) -> dict[str, np.ndarray]:
    buy_cost_no_fee = amount_in_for_exact_out(
        trade_size_base,
        reserve_in=buy_reserve_quote,
        reserve_out=buy_reserve_base,
        fee_rate=0.0,
    )
    buy_cost_with_fee_floor = amount_in_for_exact_out(
        trade_size_base,
        reserve_in=buy_reserve_quote,
        reserve_out=buy_reserve_base,
        fee_rate=buy_fee_rate_floor,
    )
    sell_return_no_fee = amount_out_for_exact_in(
        trade_size_base,
        reserve_in=sell_reserve_base,
        reserve_out=sell_reserve_quote,
        fee_rate=0.0,
    )
    sell_return_with_fee_floor = amount_out_for_exact_in(
        trade_size_base,
        reserve_in=sell_reserve_base,
        reserve_out=sell_reserve_quote,
        fee_rate=sell_fee_rate_floor,
    )

    gross_edge_quote = sell_return_no_fee - buy_cost_no_fee
    pool_fee_cost_quote_floor = (buy_cost_with_fee_floor - buy_cost_no_fee) + (sell_return_no_fee - sell_return_with_fee_floor)
    edge_after_pool_fees_quote_floor = sell_return_with_fee_floor - buy_cost_with_fee_floor
    net_edge_quote_floor = edge_after_pool_fees_quote_floor - network_fee_quote_floor
    net_edge_quote_p95 = edge_after_pool_fees_quote_floor - network_fee_quote_p95

    denominator = buy_cost_with_fee_floor
    valid = (
        np.isfinite(buy_cost_no_fee)
        & np.isfinite(buy_cost_with_fee_floor)
        & np.isfinite(sell_return_no_fee)
        & np.isfinite(sell_return_with_fee_floor)
        & np.isfinite(network_fee_quote_floor)
        & np.isfinite(network_fee_quote_p95)
        & (denominator > 0)
    )

    def to_bps(values: np.ndarray) -> np.ndarray:
        out = np.full(denominator.shape, np.nan, dtype=float)
        out[valid] = values[valid] / denominator[valid] * 10_000.0
        return out

    return {
        "gross_edge_quote": gross_edge_quote,
        "gross_edge_bps": to_bps(gross_edge_quote),
        "pool_fee_cost_quote_floor": pool_fee_cost_quote_floor,
        "pool_fee_cost_bps_floor": to_bps(pool_fee_cost_quote_floor),
        "edge_after_pool_fees_quote_floor": edge_after_pool_fees_quote_floor,
        "edge_after_pool_fees_bps_floor": to_bps(edge_after_pool_fees_quote_floor),
        "network_fee_quote_floor": network_fee_quote_floor,
        "network_fee_bps_floor": to_bps(network_fee_quote_floor),
        "network_fee_quote_p95": network_fee_quote_p95,
        "network_fee_bps_p95": to_bps(network_fee_quote_p95),
        "net_edge_quote_floor": net_edge_quote_floor,
        "net_edge_bps_floor": to_bps(net_edge_quote_floor),
        "net_edge_quote_p95": net_edge_quote_p95,
        "net_edge_bps_p95": to_bps(net_edge_quote_p95),
        "valid": valid,
    }


def build_arb_labels(
    pool_state_df: pd.DataFrame,
    selected_pools: list[SolanaPoolSpec],
    fee_bounds: dict[str, Any],
    *,
    trade_size_base: float,
) -> pd.DataFrame:
    if pool_state_df.empty or len(selected_pools) != 2:
        return pd.DataFrame()

    fee_lookup = {pool.dex: float(pool.fee_rate_floor or 0.0) for pool in selected_pools}
    dexes = sorted(pool.dex for pool in selected_pools)
    wide = (
        pool_state_df.pivot(index="timestamp", columns="dex", values=["reserve_base", "reserve_quote", "stale_state", "mid_price_quote_per_base"])
        .sort_index()
    )
    market_price_quote = wide["mid_price_quote_per_base"].mean(axis=1).to_numpy(dtype=float)
    network_fee_quote_floor = fee_bounds["network_fee_lamports_floor"] / 1e9 * market_price_quote
    network_fee_quote_p95 = fee_bounds["network_fee_lamports_p95"] / 1e9 * market_price_quote

    dex_a, dex_b = dexes
    stale_a = wide[("stale_state", dex_a)].astype("boolean").fillna(True).to_numpy(dtype=bool)
    stale_b = wide[("stale_state", dex_b)].astype("boolean").fillna(True).to_numpy(dtype=bool)
    stale_any = stale_a | stale_b

    direction_ab = _evaluate_direction(
        buy_reserve_base=wide[("reserve_base", dex_a)].to_numpy(dtype=float),
        buy_reserve_quote=wide[("reserve_quote", dex_a)].to_numpy(dtype=float),
        sell_reserve_base=wide[("reserve_base", dex_b)].to_numpy(dtype=float),
        sell_reserve_quote=wide[("reserve_quote", dex_b)].to_numpy(dtype=float),
        buy_fee_rate_floor=fee_lookup[dex_a],
        sell_fee_rate_floor=fee_lookup[dex_b],
        trade_size_base=trade_size_base,
        network_fee_quote_floor=network_fee_quote_floor,
        network_fee_quote_p95=network_fee_quote_p95,
    )
    direction_ba = _evaluate_direction(
        buy_reserve_base=wide[("reserve_base", dex_b)].to_numpy(dtype=float),
        buy_reserve_quote=wide[("reserve_quote", dex_b)].to_numpy(dtype=float),
        sell_reserve_base=wide[("reserve_base", dex_a)].to_numpy(dtype=float),
        sell_reserve_quote=wide[("reserve_quote", dex_a)].to_numpy(dtype=float),
        buy_fee_rate_floor=fee_lookup[dex_b],
        sell_fee_rate_floor=fee_lookup[dex_a],
        trade_size_base=trade_size_base,
        network_fee_quote_floor=network_fee_quote_floor,
        network_fee_quote_p95=network_fee_quote_p95,
    )
    score_ab = np.where(direction_ab["valid"], direction_ab["net_edge_quote_floor"], -np.inf)
    score_ba = np.where(direction_ba["valid"], direction_ba["net_edge_quote_floor"], -np.inf)
    use_ab = score_ab >= score_ba
    valid_any = direction_ab["valid"] | direction_ba["valid"]

    def choose(metric: str) -> np.ndarray:
        return np.where(use_ab, direction_ab[metric], direction_ba[metric])

    out = pd.DataFrame(
        {
            "timestamp": wide.index.to_numpy(),
            "buy_dex": np.where(use_ab, dex_a, dex_b),
            "sell_dex": np.where(use_ab, dex_b, dex_a),
            "trade_size_base": trade_size_base,
            "market_price_quote_per_base": market_price_quote,
            "gross_edge_quote": choose("gross_edge_quote"),
            "gross_edge_bps": choose("gross_edge_bps"),
            "pool_fee_cost_quote_floor": choose("pool_fee_cost_quote_floor"),
            "pool_fee_cost_bps_floor": choose("pool_fee_cost_bps_floor"),
            "edge_after_pool_fees_quote_floor": choose("edge_after_pool_fees_quote_floor"),
            "edge_after_pool_fees_bps_floor": choose("edge_after_pool_fees_bps_floor"),
            "network_fee_quote_floor": choose("network_fee_quote_floor"),
            "network_fee_bps_floor": choose("network_fee_bps_floor"),
            "network_fee_quote_p95": choose("network_fee_quote_p95"),
            "network_fee_bps_p95": choose("network_fee_bps_p95"),
            "net_edge_quote_floor": choose("net_edge_quote_floor"),
            "net_edge_bps_floor": choose("net_edge_bps_floor"),
            "net_edge_quote_p95": choose("net_edge_quote_p95"),
            "net_edge_bps_p95": choose("net_edge_bps_p95"),
            "stale_state": stale_any,
        }
    )
    out = out[valid_any].copy()
    out["positive_after_pool_fees_floor"] = (out["edge_after_pool_fees_quote_floor"] > 0.0) & (~out["stale_state"])
    out["positive_after_network_floor"] = (out["net_edge_quote_floor"] > 0.0) & (~out["stale_state"])
    out["positive_after_network_p95"] = (out["net_edge_quote_p95"] > 0.0) & (~out["stale_state"])
    return out.sort_values("timestamp").reset_index(drop=True)


def build_opportunity_windows(arb_df: pd.DataFrame, *, flag_column: str) -> pd.DataFrame:
    positive = arb_df[arb_df[flag_column]].copy().sort_values("timestamp")
    if positive.empty:
        return pd.DataFrame(
            columns=[
                "start_timestamp",
                "end_timestamp",
                "seconds",
                "buy_dex",
                "sell_dex",
                "max_net_edge_bps_floor",
                "mean_net_edge_bps_floor",
                "max_net_profit_quote_floor",
                "mean_net_profit_quote_floor",
            ]
        )
    positive["new_window"] = (
        positive["timestamp"].diff().ne(pd.Timedelta(seconds=1))
        | positive["buy_dex"].ne(positive["buy_dex"].shift())
        | positive["sell_dex"].ne(positive["sell_dex"].shift())
    )
    positive["window_id"] = positive["new_window"].cumsum()
    return (
        positive.groupby("window_id", as_index=False)
        .agg(
            start_timestamp=("timestamp", "min"),
            end_timestamp=("timestamp", "max"),
            seconds=("timestamp", "count"),
            buy_dex=("buy_dex", "first"),
            sell_dex=("sell_dex", "first"),
            max_net_edge_bps_floor=("net_edge_bps_floor", "max"),
            mean_net_edge_bps_floor=("net_edge_bps_floor", "mean"),
            max_net_profit_quote_floor=("net_edge_quote_floor", "max"),
            mean_net_profit_quote_floor=("net_edge_quote_floor", "mean"),
        )
        .sort_values(["max_net_edge_bps_floor", "seconds"], ascending=[False, False])
        .reset_index(drop=True)
    )


def build_size_sensitivity(
    pool_state_df: pd.DataFrame,
    selected_pools: list[SolanaPoolSpec],
    fee_bounds: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    primary_labels = pd.DataFrame()
    for trade_size in MARKET_SPEC.trade_sizes_base:
        labels = build_arb_labels(pool_state_df, selected_pools, fee_bounds, trade_size_base=trade_size)
        if trade_size == MARKET_SPEC.primary_trade_size_base:
            primary_labels = labels.copy()
        if labels.empty:
            summary_rows.append(
                {
                    "trade_size_base": trade_size,
                    "observed_seconds": 0,
                    "gross_positive_seconds": 0,
                    "pool_fee_floor_positive_seconds": 0,
                    "network_floor_positive_seconds": 0,
                    "network_p95_positive_seconds": 0,
                    "max_net_edge_bps_floor": np.nan,
                    "mean_net_edge_bps_floor": np.nan,
                }
            )
            continue
        summary_rows.append(
            {
                "trade_size_base": trade_size,
                "observed_seconds": int(len(labels)),
                "gross_positive_seconds": int((labels["gross_edge_quote"] > 0).sum()),
                "pool_fee_floor_positive_seconds": int(labels["positive_after_pool_fees_floor"].sum()),
                "network_floor_positive_seconds": int(labels["positive_after_network_floor"].sum()),
                "network_p95_positive_seconds": int(labels["positive_after_network_p95"].sum()),
                "max_net_edge_bps_floor": float(labels["net_edge_bps_floor"].max()),
                "mean_net_edge_bps_floor": float(labels["net_edge_bps_floor"].mean()),
            }
        )
    return pd.DataFrame(summary_rows), primary_labels


def run_qc(pool_events_df: pd.DataFrame, pool_state_df: pd.DataFrame, arb_labels_df: pd.DataFrame) -> dict[str, Any]:
    report = {
        "duplicate_events": int(pool_events_df.duplicated(["dex", "signature"]).sum()) if not pool_events_df.empty else 0,
        "non_positive_reserves": int(((pool_events_df["reserve_base"] <= 0) | (pool_events_df["reserve_quote"] <= 0)).sum()) if not pool_events_df.empty else 0,
        "pool_state_nulls": int(pool_state_df[["reserve_base", "reserve_quote"]].isna().sum().sum()) if not pool_state_df.empty else 0,
        "arb_label_nulls": int(arb_labels_df[["net_edge_quote_floor", "net_edge_bps_floor"]].isna().sum().sum()) if not arb_labels_df.empty else 0,
    }
    report["passed"] = all(value == 0 for value in report.values() if isinstance(value, int))
    return report


def dataset_summary_frame(dataset: dict[str, Any]) -> pd.DataFrame:
    summary_rows = [
        {"field": "status", "value": dataset["status"]},
        {"field": "window_start_utc", "value": dataset["window_start"]},
        {"field": "window_end_utc", "value": dataset["window_end"]},
        {"field": "selected_pool_count", "value": len(dataset["selected_pools"])},
        {"field": "venue_catalog_rows", "value": len(dataset["venue_catalog"])},
        {"field": "pool_event_rows", "value": len(dataset["pool_events"])},
        {"field": "pool_state_rows", "value": len(dataset["pool_state"])},
        {"field": "arb_label_rows", "value": len(dataset["arb_labels"])},
    ]
    if dataset.get("fee_bounds"):
        summary_rows.extend(
            [
                {"field": "network_fee_lamports_floor", "value": dataset["fee_bounds"]["network_fee_lamports_floor"]},
                {"field": "network_fee_lamports_p95", "value": dataset["fee_bounds"]["network_fee_lamports_p95"]},
            ]
        )
    if not dataset["arb_labels"].empty:
        summary_rows.append({"field": "max_net_edge_bps_floor", "value": float(dataset["arb_labels"]["net_edge_bps_floor"].max())})
    return pd.DataFrame(summary_rows)


def _paths(project_root: Path) -> dict[str, Path]:
    data_root = project_root / "data" / MARKET_SPEC.slug
    output_root = project_root / "outputs" / MARKET_SPEC.slug
    return {
        "raw_dir": data_root / "raw",
        "curated_dir": data_root / "curated",
        "metadata_dir": output_root / "metadata",
        "report_dir": output_root / "report",
        "venue_catalog_raw": data_root / "raw" / "official_venue_catalog.csv",
        "selected_pools_raw": data_root / "raw" / "selected_reconstructable_pools.csv",
        "signatures_raw": data_root / "raw" / "pool_signatures.parquet",
        "events_curated": data_root / "curated" / "pool_events_1s_source.parquet",
        "events_curated_preview": data_root / "curated" / "pool_events_1s_source_preview.csv",
        "pool_state": data_root / "curated" / "pool_state_1s.parquet",
        "pool_state_preview": data_root / "curated" / "pool_state_1s_preview.csv",
        "arb_labels": data_root / "curated" / "arb_labels_1s.parquet",
        "arb_labels_preview": data_root / "curated" / "arb_labels_1s_preview.csv",
        "qc_report": data_root / "curated" / "qc_report_1s.json",
        "manifest": output_root / "metadata" / "dataset_manifest.json",
        "size_sensitivity": output_root / "report" / "opportunity_size_sensitivity_1s.csv",
        "opportunity_windows": output_root / "report" / "opportunity_windows_1s.csv",
    }


def _write_preview_csv(path: Path, frame: pd.DataFrame, limit: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.head(limit).to_csv(path, index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _market_manifest(
    *,
    status: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    selected_pools: list[SolanaPoolSpec],
    fee_bounds: dict[str, Any],
    venue_catalog: pd.DataFrame,
    pool_events_df: pd.DataFrame,
    pool_state_df: pd.DataFrame,
    arb_labels_df: pd.DataFrame,
    size_sensitivity_df: pd.DataFrame,
    opportunity_windows_df: pd.DataFrame,
    qc_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": status,
        "market_slug": MARKET_SPEC.slug,
        "chain_name": MARKET_SPEC.chain_name,
        "pair_label": MARKET_SPEC.pair_label,
        "base_token": asdict(MARKET_SPEC.base_token),
        "quote_token": asdict(MARKET_SPEC.quote_token),
        "sample_window_days": MARKET_SPEC.sample_window_days,
        "state_frequency": MARKET_SPEC.frequency,
        "window_start_utc": window_start.isoformat(),
        "window_end_utc": window_end.isoformat(),
        "selected_pools": [asdict(pool) for pool in selected_pools],
        "network_fee_bounds": fee_bounds,
        "venue_catalog_rows": int(len(venue_catalog)),
        "pool_event_rows": int(len(pool_events_df)),
        "pool_state_rows": int(len(pool_state_df)),
        "arb_label_rows": int(len(arb_labels_df)),
        "size_sensitivity_rows": int(len(size_sensitivity_df)),
        "opportunity_window_rows": int(len(opportunity_windows_df)),
        "max_net_edge_bps_floor": float(arb_labels_df["net_edge_bps_floor"].max()) if not arb_labels_df.empty else None,
        "positive_network_floor_seconds": int(arb_labels_df["positive_after_network_floor"].sum()) if not arb_labels_df.empty else 0,
        "positive_network_p95_seconds": int(arb_labels_df["positive_after_network_p95"].sum()) if not arb_labels_df.empty else 0,
        "limitations": [
            "Orca SOL/USDC pools are Whirlpool/Splash-style CLMM pools and are not reconstructed here.",
            "Raydium Standard SOL/USDC pools include OpenBook inventory and are excluded from reserve-only reconstruction.",
            "Meteora DAMM v1 SOL/USDC pools reuse shared Dynamic Vault pairs, so direct vault-balance history is not per-pool state.",
            "Meteora DAMM v2 full-range SOL/USDC uses a dynamic fee toggle; arbitrage labels therefore use the current base fee as a lower-bound pool fee.",
            "Network-fee outputs are shown as lower-bound and p95 fee bands derived from observed pool-touching transaction fees, not a claimed exact arbitrage transaction cost.",
        ],
        "qc_report": qc_report,
    }


def load_or_build_solana_dataset(
    *,
    project_root: Path | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    root = find_project_root(project_root or Path.cwd())
    paths = _paths(root)
    if not refresh and all(
        path.exists()
        for path in [
            paths["venue_catalog_raw"],
            paths["selected_pools_raw"],
            paths["events_curated"],
            paths["pool_state"],
            paths["arb_labels"],
            paths["qc_report"],
            paths["size_sensitivity"],
            paths["opportunity_windows"],
            paths["manifest"],
        ]
    ):
        venue_catalog = pd.read_csv(paths["venue_catalog_raw"])
        selected_pools_df = pd.read_csv(paths["selected_pools_raw"])
        pool_events = pd.read_parquet(paths["events_curated"])
        pool_state = pd.read_parquet(paths["pool_state"])
        arb_labels = pd.read_parquet(paths["arb_labels"])
        qc_report = json.loads(paths["qc_report"].read_text())
        size_sensitivity = pd.read_csv(paths["size_sensitivity"])
        opportunity_windows = pd.read_csv(paths["opportunity_windows"], parse_dates=["start_timestamp", "end_timestamp"])
        manifest = json.loads(paths["manifest"].read_text())
        fee_bounds = manifest.get("network_fee_bounds", {})
        selected_pools = selected_pools_df.to_dict(orient="records")
        return {
            "status": manifest["status"],
            "window_start": manifest["window_start_utc"],
            "window_end": manifest["window_end_utc"],
            "venue_catalog": venue_catalog,
            "selected_pools": selected_pools,
            "pool_events": pool_events,
            "pool_state": pool_state,
            "arb_labels": arb_labels,
            "qc_report": qc_report,
            "size_sensitivity": size_sensitivity,
            "opportunity_windows": opportunity_windows,
            "manifest": manifest,
            "fee_bounds": fee_bounds,
        }

    for key in ["raw_dir", "curated_dir", "metadata_dir", "report_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    venue_catalog = discover_official_venue_catalog()
    selected_pools = select_reconstructable_pools(venue_catalog)
    if len(selected_pools) != 2:
        raise RuntimeError("Solana investigation did not find two reconstructable SOL/USDC venues.")

    venue_catalog.to_csv(paths["venue_catalog_raw"], index=False)
    pd.DataFrame(
        [
            {
                "dex": pool.dex,
                "pool_address": pool.pool_address,
                "pool_name": pool.pool_name,
                "pool_type": pool.pool_type,
                "tvl_usd": pool.tvl_usd,
                "volume_24h_usd": pool.volume_24h_usd,
                "fee_rate_floor": pool.fee_rate_floor,
                "fee_model": pool.fee_model,
                "vault_base": pool.vault_base,
                "vault_quote": pool.vault_quote,
            }
            for pool in selected_pools
        ]
    ).to_csv(paths["selected_pools_raw"], index=False)

    window_end = pd.Timestamp.now(tz="UTC").floor("s")
    window_start = window_end - pd.Timedelta(days=MARKET_SPEC.sample_window_days)

    rpc = SolanaRpcClient()
    existing_signatures = pd.read_parquet(paths["signatures_raw"]) if paths["signatures_raw"].exists() else pd.DataFrame()
    existing_events = pd.read_parquet(paths["events_curated"]) if paths["events_curated"].exists() else pd.DataFrame()
    signature_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    for pool in selected_pools:
        if not existing_signatures.empty and "dex" in existing_signatures.columns and pool.dex in existing_signatures["dex"].unique():
            signatures = existing_signatures[existing_signatures["dex"] == pool.dex].copy()
            print(f"[solana] reusing signatures for {pool.dex}: {len(signatures)} rows", flush=True)
        else:
            print(f"[solana] collecting signatures for {pool.dex} {pool.pool_address}", flush=True)
            signatures = collect_pool_signature_rows(rpc, pool, window_start=window_start)
            signatures["dex"] = pool.dex
            signatures["pool_address"] = pool.pool_address
        signature_frames.append(signatures)
        pd.concat(signature_frames, ignore_index=True).to_parquet(paths["signatures_raw"], index=False)
        print(f"[solana] {pool.dex}: collected {len(signatures)} signatures", flush=True)

        if not existing_events.empty and "dex" in existing_events.columns and pool.dex in existing_events["dex"].unique():
            events = existing_events[existing_events["dex"] == pool.dex].copy()
            print(f"[solana] reusing reserve events for {pool.dex}: {len(events)} rows", flush=True)
        else:
            print(f"[solana] fetching blocks for {pool.dex}", flush=True)
            events = extract_pool_events_from_slots(rpc, pool, signatures)
        event_frames.append(events)
        pd.concat(event_frames, ignore_index=True).to_parquet(paths["events_curated"], index=False)
        _write_preview_csv(paths["events_curated_preview"], pd.concat(event_frames, ignore_index=True))
        print(f"[solana] {pool.dex}: parsed {len(events)} reserve events", flush=True)

    signature_df = pd.concat(signature_frames, ignore_index=True).sort_values(["timestamp", "dex"]).reset_index(drop=True)
    pool_events_df = pd.concat(event_frames, ignore_index=True).sort_values(["timestamp", "dex"]).reset_index(drop=True)
    pool_state_df = build_pool_state(pool_events_df, window_start=window_start, window_end=window_end)
    fee_bounds = build_fee_bounds(pool_events_df)
    size_sensitivity_df, arb_labels_df = build_size_sensitivity(pool_state_df, selected_pools, fee_bounds)
    opportunity_windows_df = build_opportunity_windows(arb_labels_df, flag_column="positive_after_network_floor")
    qc_report = run_qc(pool_events_df, pool_state_df, arb_labels_df)
    manifest = _market_manifest(
        status="supported_with_limitations",
        window_start=window_start,
        window_end=window_end,
        selected_pools=selected_pools,
        fee_bounds=fee_bounds,
        venue_catalog=venue_catalog,
        pool_events_df=pool_events_df,
        pool_state_df=pool_state_df,
        arb_labels_df=arb_labels_df,
        size_sensitivity_df=size_sensitivity_df,
        opportunity_windows_df=opportunity_windows_df,
        qc_report=qc_report,
    )

    signature_df.to_parquet(paths["signatures_raw"], index=False)
    pool_events_df.to_parquet(paths["events_curated"], index=False)
    _write_preview_csv(paths["events_curated_preview"], pool_events_df)
    pool_state_df.to_parquet(paths["pool_state"], index=False)
    _write_preview_csv(paths["pool_state_preview"], pool_state_df)
    arb_labels_df.to_parquet(paths["arb_labels"], index=False)
    _write_preview_csv(paths["arb_labels_preview"], arb_labels_df)
    _write_json(paths["qc_report"], qc_report)
    size_sensitivity_df.to_csv(paths["size_sensitivity"], index=False)
    opportunity_windows_df.to_csv(paths["opportunity_windows"], index=False)
    _write_json(paths["manifest"], manifest)

    return {
        "status": manifest["status"],
        "window_start": manifest["window_start_utc"],
        "window_end": manifest["window_end_utc"],
        "venue_catalog": venue_catalog,
        "selected_pools": [
            {
                "dex": pool.dex,
                "pool_address": pool.pool_address,
                "pool_name": pool.pool_name,
                "pool_type": pool.pool_type,
                "tvl_usd": pool.tvl_usd,
                "volume_24h_usd": pool.volume_24h_usd,
                "fee_rate_floor": pool.fee_rate_floor,
                "fee_model": pool.fee_model,
                "reconstructable_reason": pool.reconstructable_reason,
            }
            for pool in selected_pools
        ],
        "pool_events": pool_events_df,
        "pool_state": pool_state_df,
        "arb_labels": arb_labels_df,
        "qc_report": qc_report,
        "size_sensitivity": size_sensitivity_df,
        "opportunity_windows": opportunity_windows_df,
        "manifest": manifest,
        "fee_bounds": fee_bounds,
    }
