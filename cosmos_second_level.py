from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from second_level_markets import (
    FEES_256_TOPIC,
    PairSpec,
    RpcClient,
    SYNC_256_TOPIC,
    TOPIC_TO_EVENT_NAME,
    TokenSpec,
    UNISWAP_V2_SWAP_TOPIC,
    _collect_logs_adaptive,
    _read_json,
    _write_json,
    _write_preview_csv,
    build_base_fee,
    build_pair_metadata,
    build_pool_state,
    build_swaps_raw,
    fetch_block_headers,
    market_paths,
    normalize_event_logs,
    run_qc,
    select_rpc_urls,
)


MARKET_KEY = "11_cosmos_research"
MARKET_SLUG = "cosmos_research"
CHAIN_KEY = "kava"
CHAIN_NAME = "Cosmos Proxy (Kava EVM)"
PAIR_LABEL = "ATOM stablecoin triangle"
STATE_FREQUENCY = "1s"
SAMPLE_WINDOW_DAYS = 3
STALE_AFTER_MINUTES = 15
BLOCK_CHUNK_SIZE = 10_000
ANCHOR_LOOKBACK_BLOCKS = 20_000
PRIORITY_FEE_GWEI = 0.0

KAVA_WKAVA = TokenSpec("WKAVA", "0xc86c7c0efbd6a49b35e8714c5f59d99de09a225b", 18)
KAVA_ATOM = TokenSpec("ATOM", "0x15932e26f5bd4923d46a2b205191c4b5d5f43fe3", 6)
KAVA_USDT = TokenSpec("USDt", "0x919c1c267bc06a7039e03fcc2ef738525769109c", 6)
KAVA_AXLUSDC = TokenSpec("axlUSDC", "0xeb466342c4d449bc9f53a865d5cb90586f405215", 6)

ATOM_USDT_DEX = "equilibre_atom_usdt"
ATOM_AXLUSDC_DEX = "equilibre_atom_axlusdc"
USDT_AXLUSDC_DEX = "equilibre_usdt_axlusdc"
GAS_REFERENCE_DEX = "equilibre_wkava_usdt_gas_reference"

ROUTE_PAIRS: tuple[PairSpec, ...] = (
    PairSpec(
        dex=ATOM_USDT_DEX,
        pair_address="0xd39c219b018207f44799d93433c9ba23a01efba2",
        fee_bps=0.0,
        event_style="equilibre_v2",
        base_token=KAVA_ATOM,
        quote_token=KAVA_USDT,
    ),
    PairSpec(
        dex=ATOM_AXLUSDC_DEX,
        pair_address="0xa82bf6cb717a00ccf2fea79b4b3821b9108e8a66",
        fee_bps=0.0,
        event_style="equilibre_v2",
        base_token=KAVA_ATOM,
        quote_token=KAVA_AXLUSDC,
    ),
    PairSpec(
        dex=USDT_AXLUSDC_DEX,
        pair_address="0x4a18f16b6a4f695639b0d1390263def2e91fc60f",
        fee_bps=0.0,
        event_style="equilibre_v2",
        base_token=KAVA_USDT,
        quote_token=KAVA_AXLUSDC,
    ),
)

GAS_REFERENCE_PAIR = PairSpec(
    dex=GAS_REFERENCE_DEX,
    pair_address="0xbe87f2e81aa16238445651fabf62ae097498c200",
    fee_bps=0.0,
    event_style="equilibre_v2",
    base_token=KAVA_WKAVA,
    quote_token=KAVA_USDT,
    role="gas_reference",
)

PRIMARY_TRADE_SIZE_QUOTE = 0.5
TRADE_SIZES_QUOTE: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)


def cosmos_paths(project_root: Path) -> dict[str, Path]:
    paths = market_paths(project_root, MARKET_SLUG)
    paths.update(
        {
            "fee_summary": paths["raw_dir"] / "fee_inference_summary.csv",
            "route_catalog": paths["raw_dir"] / "route_pool_catalog.csv",
            "tx_receipts": paths["raw_dir"] / "swap_transaction_receipts.parquet",
            "tx_receipts_preview": paths["raw_dir"] / "swap_transaction_receipts_preview.csv",
        }
    )
    return paths


def _block_timestamp(rpc: RpcClient, block_number: int) -> datetime:
    raw = rpc.call("eth_getBlockByNumber", [hex(int(block_number)), False], use_cache=True)
    if not raw:
        raise ValueError(f"Block {block_number} is unavailable from the configured RPC")
    return datetime.fromtimestamp(int(raw["timestamp"], 16), tz=UTC)


def _resolve_block_window(rpc: RpcClient) -> tuple[int, int, datetime, datetime]:
    end_block = rpc.get_latest_block_number()
    end_time = _block_timestamp(rpc, end_block)
    start_time = end_time - timedelta(days=SAMPLE_WINDOW_DAYS)

    sample_back = max(end_block - 1_000, 1)
    sample_time = _block_timestamp(rpc, sample_back)
    sampled_seconds = max((end_time - sample_time).total_seconds(), 1.0)
    sampled_blocks = max(end_block - sample_back, 1)
    avg_block_seconds = sampled_seconds / sampled_blocks
    estimated_lookback = int((end_time - start_time).total_seconds() / avg_block_seconds)

    low = max(1, end_block - estimated_lookback - 20_000)
    high = end_block
    while low > 1 and _block_timestamp(rpc, low) > start_time:
        high = low
        low = max(1, low - max(estimated_lookback // 2, 20_000))

    while low < high:
        mid = (low + high) // 2
        if _block_timestamp(rpc, mid) < start_time:
            low = mid + 1
        else:
            high = mid
    start_block = low
    return start_block, end_block, start_time, end_time


def _amount_out_exact_in(amount_in: float | np.ndarray, reserve_in: np.ndarray, reserve_out: np.ndarray, fee_rate: float) -> np.ndarray:
    amount_in_arr, reserve_in_arr, reserve_out_arr = np.broadcast_arrays(
        np.asarray(amount_in, dtype=float),
        np.asarray(reserve_in, dtype=float),
        np.asarray(reserve_out, dtype=float),
    )
    result = np.full(amount_in_arr.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(amount_in_arr)
        & np.isfinite(reserve_in_arr)
        & np.isfinite(reserve_out_arr)
        & (amount_in_arr > 0)
        & (reserve_in_arr > 0)
        & (reserve_out_arr > 0)
    )
    effective_in = amount_in_arr * (1.0 - fee_rate)
    result[valid] = reserve_out_arr[valid] * effective_in[valid] / (reserve_in_arr[valid] + effective_in[valid])
    return result


def _parquet_safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    safe = df.copy()
    int64_min = np.iinfo(np.int64).min
    int64_max = np.iinfo(np.int64).max

    for column in safe.columns:
        series = safe[column]

        if (
            pd.api.types.is_datetime64_any_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_integer_dtype(series)
            or pd.api.types.is_float_dtype(series)
        ):
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue

        if not non_null.map(lambda value: isinstance(value, (int, np.integer)) and not isinstance(value, bool)).all():
            continue

        max_abs = non_null.map(lambda value: abs(int(value))).max()
        if max_abs > int64_max:
            safe[column] = series.map(lambda value: None if pd.isna(value) else str(value))
        else:
            safe[column] = pd.array(
                [pd.NA if pd.isna(value) else int(value) for value in series],
                dtype="Int64",
            )

    return safe


def _find_anchor_sync_log(
    rpc: RpcClient,
    pair: PairSpec,
    *,
    start_block: int,
    lookback_blocks: int,
) -> dict[str, Any] | None:
    if start_block <= 1:
        return None
    search_end = start_block - 1
    search_start_limit = max(1, start_block - lookback_blocks)
    anchor_chunk = 10_000
    while search_end >= search_start_limit:
        chunk_start = max(search_start_limit, search_end - anchor_chunk + 1)
        logs = _collect_logs_adaptive(
            rpc,
            address=pair.pair_address,
            from_block=chunk_start,
            to_block=search_end,
            topics=[[SYNC_256_TOPIC]],
            min_chunk_size=25,
        )
        if logs:
            return sorted(logs, key=lambda item: (int(item["blockNumber"], 16), int(item["logIndex"], 16)))[-1]
        search_end = chunk_start - 1
    return None


def collect_equilibre_pair_logs(
    rpc: RpcClient,
    pair: PairSpec,
    *,
    start_block: int,
    end_block: int,
    chunk_size: int,
    lookback_blocks: int,
) -> list[dict[str, Any]]:
    raw_logs: list[dict[str, Any]] = []
    anchor = _find_anchor_sync_log(rpc, pair, start_block=start_block, lookback_blocks=lookback_blocks)
    if anchor is not None:
        raw_logs.append(anchor)

    topic_list = [SYNC_256_TOPIC, UNISWAP_V2_SWAP_TOPIC, FEES_256_TOPIC]
    for chunk_start in range(start_block, end_block + 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size - 1, end_block)
        for topic0 in topic_list:
            raw_logs.extend(
                _collect_logs_adaptive(
                    rpc,
                    address=pair.pair_address,
                    from_block=chunk_start,
                    to_block=chunk_end,
                    topics=[[topic0]],
                    min_chunk_size=25,
                )
            )

    seen: set[tuple[str, int]] = set()
    rows: list[dict[str, Any]] = []
    for log in sorted(raw_logs, key=lambda item: (int(item["blockNumber"], 16), int(item["logIndex"], 16))):
        key = (log["transactionHash"], int(log["logIndex"], 16))
        if key in seen:
            continue
        seen.add(key)
        topic0 = log["topics"][0]
        rows.append(
            {
                "role": pair.role,
                "dex": pair.dex,
                "pair_address": pair.pair_address,
                "block_number": int(log["blockNumber"], 16),
                "transaction_hash": log["transactionHash"],
                "log_index": int(log["logIndex"], 16),
                "event_name": TOPIC_TO_EVENT_NAME.get(topic0, "Unknown"),
                "topic0": topic0,
                "topic1": log["topics"][1] if len(log["topics"]) > 1 else None,
                "topic2": log["topics"][2] if len(log["topics"]) > 2 else None,
                "topic3": log["topics"][3] if len(log["topics"]) > 3 else None,
                "data": log["data"],
                "removed": bool(log.get("removed", False)),
            }
        )
    return rows


def _summarize_selected_pools(pair_metadata_df: pd.DataFrame) -> pd.DataFrame:
    labels = {
        ATOM_USDT_DEX: "Route pool: buy or sell ATOM against USDt",
        ATOM_AXLUSDC_DEX: "Route pool: buy or sell ATOM against axlUSDC",
        USDT_AXLUSDC_DEX: "Route pool: explicit stablecoin conversion, avoids parity assumptions",
        GAS_REFERENCE_DEX: "Gas reference pool: converts WKAVA gas cost into USDt",
    }
    out = pair_metadata_df.copy()
    out["selected_role_note"] = out["dex"].map(labels)
    return out[
        [
            "role",
            "dex",
            "pair_address",
            "event_style",
            "base_symbol",
            "quote_symbol",
            "fee_bps",
            "fee_rate",
            "selected_role_note",
        ]
    ].sort_values(["role", "dex"]).reset_index(drop=True)


def _infer_fee_rates(events_df: pd.DataFrame, pair_metadata_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    swaps = events_df[events_df["event_name"] == "Swap"].copy()
    fees = events_df[events_df["event_name"] == "Fees"].copy()
    if swaps.empty or fees.empty:
        raise ValueError("Equilibre fee inference requires both Swap and Fees events")

    merged_frames: list[pd.DataFrame] = []
    group_cols = ["dex", "pair_address", "transaction_hash"]
    fee_keep = ["log_index", "fee0_raw", "fee1_raw"]
    swap_keep = [
        "role",
        "dex",
        "pair_address",
        "transaction_hash",
        "block_number",
        "timestamp",
        "log_index",
        "amount0_in_raw",
        "amount1_in_raw",
        "amount0_out_raw",
        "amount1_out_raw",
    ]

    for group_key, swap_group in swaps[swap_keep].groupby(group_cols, sort=False):
        fee_group = fees.loc[
            (fees["dex"] == group_key[0])
            & (fees["pair_address"] == group_key[1])
            & (fees["transaction_hash"] == group_key[2]),
            fee_keep,
        ].copy()
        if fee_group.empty:
            merged_frames.append(swap_group.assign(fee_log_index=np.nan, fee0_raw=np.nan, fee1_raw=np.nan))
            continue

        merged = pd.merge_asof(
            swap_group.sort_values("log_index"),
            fee_group.sort_values("log_index").rename(columns={"log_index": "fee_log_index"}),
            left_on="log_index",
            right_on="fee_log_index",
            direction="backward",
        )
        merged_frames.append(merged)

    inferred = pd.concat(merged_frames, ignore_index=True)
    inferred["input_side"] = np.select(
        [
            (inferred["amount0_in_raw"] > 0) & (inferred["amount1_in_raw"] == 0),
            (inferred["amount1_in_raw"] > 0) & (inferred["amount0_in_raw"] == 0),
        ],
        ["token0", "token1"],
        default="mixed",
    )
    inferred["fee_rate"] = np.nan

    token0_mask = (
        inferred["input_side"].eq("token0")
        & inferred["amount0_in_raw"].gt(0)
        & inferred["fee0_raw"].notna()
    )
    inferred.loc[token0_mask, "fee_rate"] = (
        inferred.loc[token0_mask, "fee0_raw"].astype(float)
        / inferred.loc[token0_mask, "amount0_in_raw"].astype(float)
    )

    token1_mask = (
        inferred["input_side"].eq("token1")
        & inferred["amount1_in_raw"].gt(0)
        & inferred["fee1_raw"].notna()
    )
    inferred.loc[token1_mask, "fee_rate"] = (
        inferred.loc[token1_mask, "fee1_raw"].astype(float)
        / inferred.loc[token1_mask, "amount1_in_raw"].astype(float)
    )
    inferred.loc[~np.isfinite(inferred["fee_rate"]), "fee_rate"] = np.nan
    inferred["fee_bps_inferred"] = inferred["fee_rate"] * 10_000.0

    fee_summary = (
        inferred.groupby(["role", "dex", "pair_address"], as_index=False)
        .agg(
            swaps_observed=("transaction_hash", "count"),
            fee_matches=("fee_rate", lambda s: int(np.isfinite(s).sum())),
            median_fee_rate=("fee_rate", "median"),
            min_fee_rate=("fee_rate", "min"),
            max_fee_rate=("fee_rate", "max"),
            median_fee_bps=("fee_bps_inferred", "median"),
            min_fee_bps=("fee_bps_inferred", "min"),
            max_fee_bps=("fee_bps_inferred", "max"),
        )
        .sort_values(["role", "dex"])
        .reset_index(drop=True)
    )
    fee_summary["fee_rate_range_bps"] = fee_summary["max_fee_bps"] - fee_summary["min_fee_bps"]

    updated = pair_metadata_df.merge(
        fee_summary[["dex", "median_fee_rate", "median_fee_bps"]],
        on="dex",
        how="left",
    )
    updated["fee_rate"] = updated["median_fee_rate"].fillna(updated["fee_rate"])
    updated["fee_bps"] = updated["median_fee_bps"].fillna(updated["fee_bps"])
    updated = updated.drop(columns=["median_fee_rate", "median_fee_bps"])
    return fee_summary, updated


def _fetch_transaction_receipts(rpc: RpcClient, tx_hashes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    unique_hashes = sorted(set(tx_hashes))
    for start in range(0, len(unique_hashes), 50):
        chunk = unique_hashes[start : start + 50]
        calls = [("eth_getTransactionReceipt", [tx_hash]) for tx_hash in chunk]
        raw_receipts = rpc.batch_call(calls, use_cache=True)
        for tx_hash, receipt in zip(chunk, raw_receipts):
            if not receipt:
                continue
            rows.append(
                {
                    "transaction_hash": tx_hash,
                    "block_number": int(receipt["blockNumber"], 16),
                    "gas_used": int(receipt["gasUsed"], 16),
                    "effective_gas_price_wei": int(receipt.get("effectiveGasPrice", "0x0"), 16),
                    "status": int(receipt.get("status", "0x1"), 16),
                    "to_address": receipt.get("to"),
                    "from_address": receipt.get("from"),
                }
            )
    return pd.DataFrame(rows).sort_values(["block_number", "transaction_hash"]).reset_index(drop=True)


def _estimate_gas_units(events_df: pd.DataFrame, receipts_df: pd.DataFrame) -> tuple[int, dict[str, Any]]:
    route_swaps = events_df[(events_df["role"] == "arb_pair") & (events_df["event_name"] == "Swap")].copy()
    route_touch = (
        route_swaps.groupby("transaction_hash", as_index=False)
        .agg(
            route_pool_count=("pair_address", "nunique"),
            route_swap_events=("pair_address", "count"),
        )
    )
    gas_frame = route_touch.merge(receipts_df, on="transaction_hash", how="left")

    one_pool = gas_frame.loc[gas_frame["route_pool_count"] == 1, "gas_used"].dropna()
    two_pool = gas_frame.loc[gas_frame["route_pool_count"] == 2, "gas_used"].dropna()
    three_pool = gas_frame.loc[gas_frame["route_pool_count"] >= 3, "gas_used"].dropna()

    if not three_pool.empty:
        gas_units = int(round(float(three_pool.median())))
        source = "observed_three_pool_swap_receipts"
    elif not two_pool.empty and not one_pool.empty:
        one_median = float(one_pool.median())
        two_median = float(two_pool.median())
        per_extra_pool = max(two_median - one_median, 0.0)
        gas_units = int(round(two_median + per_extra_pool))
        source = "extrapolated_from_observed_two_pool_swap_receipts"
    elif not two_pool.empty:
        gas_units = int(round(float(two_pool.quantile(0.9)) * 1.5))
        source = "scaled_from_observed_two_pool_swap_receipts"
    elif not one_pool.empty:
        gas_units = int(round(float(one_pool.quantile(0.9)) * 2.5))
        source = "scaled_from_observed_one_pool_swap_receipts"
    else:
        gas_units = 350_000
        source = "fallback_no_receipts"

    diagnostics = {
        "gas_units": gas_units,
        "gas_estimation_source": source,
        "tx_count_route_pool_1": int((gas_frame["route_pool_count"] == 1).sum()),
        "tx_count_route_pool_2": int((gas_frame["route_pool_count"] == 2).sum()),
        "tx_count_route_pool_3_plus": int((gas_frame["route_pool_count"] >= 3).sum()),
        "median_single_pool_gas": float(one_pool.median()) if not one_pool.empty else None,
        "median_two_pool_gas": float(two_pool.median()) if not two_pool.empty else None,
        "median_three_pool_gas": float(three_pool.median()) if not three_pool.empty else None,
    }
    return gas_units, diagnostics


def _build_gas_quote_series(gas_reference_state_df: pd.DataFrame) -> pd.DataFrame:
    out = gas_reference_state_df[["timestamp", "mid_price_quote_per_base", "stale_state"]].copy()
    return out.rename(
        columns={
            "mid_price_quote_per_base": "gas_quote_price",
            "stale_state": "gas_reference_stale_state",
        }
    )


def _evaluate_route(
    *,
    route_name: str,
    buy_dex: str,
    sell_dex: str,
    conversion_dex: str,
    trade_size_quote: float,
    gas_cost_quote: np.ndarray,
    step1_reserve_in: np.ndarray,
    step1_reserve_out: np.ndarray,
    step2_reserve_in: np.ndarray,
    step2_reserve_out: np.ndarray,
    step3_reserve_in: np.ndarray,
    step3_reserve_out: np.ndarray,
    step1_fee_rate: float,
    step2_fee_rate: float,
    step3_fee_rate: float,
) -> dict[str, Any]:
    step1_no_fee = _amount_out_exact_in(trade_size_quote, step1_reserve_in, step1_reserve_out, 0.0)
    step1_with_fee = _amount_out_exact_in(trade_size_quote, step1_reserve_in, step1_reserve_out, step1_fee_rate)
    step2_no_fee = _amount_out_exact_in(step1_no_fee, step2_reserve_in, step2_reserve_out, 0.0)
    step2_with_fee = _amount_out_exact_in(step1_with_fee, step2_reserve_in, step2_reserve_out, step2_fee_rate)
    final_no_fee = _amount_out_exact_in(step2_no_fee, step3_reserve_in, step3_reserve_out, 0.0)
    final_with_fee = _amount_out_exact_in(step2_with_fee, step3_reserve_in, step3_reserve_out, step3_fee_rate)

    gross_edge_quote = final_no_fee - trade_size_quote
    fee_cost_quote = final_no_fee - final_with_fee
    net_edge_quote = final_with_fee - trade_size_quote - gas_cost_quote

    valid = (
        np.isfinite(final_no_fee)
        & np.isfinite(final_with_fee)
        & np.isfinite(gas_cost_quote)
        & (trade_size_quote > 0)
    )
    gross_edge_bps = np.full(final_no_fee.shape, np.nan, dtype=float)
    fee_cost_bps = np.full(final_no_fee.shape, np.nan, dtype=float)
    gas_cost_bps = np.full(final_no_fee.shape, np.nan, dtype=float)
    net_edge_bps = np.full(final_no_fee.shape, np.nan, dtype=float)
    gross_edge_bps[valid] = gross_edge_quote[valid] / trade_size_quote * 10_000.0
    fee_cost_bps[valid] = fee_cost_quote[valid] / trade_size_quote * 10_000.0
    gas_cost_bps[valid] = gas_cost_quote[valid] / trade_size_quote * 10_000.0
    net_edge_bps[valid] = net_edge_quote[valid] / trade_size_quote * 10_000.0

    return {
        "route_name": route_name,
        "buy_dex": buy_dex,
        "sell_dex": sell_dex,
        "conversion_dex": conversion_dex,
        "final_quote_no_fee": final_no_fee,
        "final_quote_with_fee": final_with_fee,
        "gross_edge_quote": gross_edge_quote,
        "gross_edge_bps": gross_edge_bps,
        "fee_cost_quote": fee_cost_quote,
        "fee_cost_bps": fee_cost_bps,
        "gas_cost_quote": gas_cost_quote,
        "gas_cost_bps": gas_cost_bps,
        "net_edge_quote": net_edge_quote,
        "net_edge_bps": net_edge_bps,
        "valid": valid,
    }


def build_triangle_arb_labels(
    pool_state_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    pair_metadata_df: pd.DataFrame,
    gas_reference_state_df: pd.DataFrame,
    *,
    trade_size_quote: float,
    gas_units: int,
) -> pd.DataFrame:
    if pool_state_df.empty:
        return pd.DataFrame()

    wide = (
        pool_state_df.pivot(index="timestamp", columns="dex", values=["reserve_base", "reserve_quote", "stale_state"])
        .sort_index()
    )
    gas_frame = build_base_fee(blocks_df, frequency=STATE_FREQUENCY).set_index("timestamp")
    gas_frame = gas_frame.join(_build_gas_quote_series(gas_reference_state_df).set_index("timestamp"), how="outer")
    gas_frame["base_fee_per_gas_wei"] = gas_frame["base_fee_per_gas_wei"].ffill().fillna(0.0)
    gas_frame["gas_quote_price"] = gas_frame["gas_quote_price"].ffill()
    gas_frame["gas_reference_stale_state"] = gas_frame["gas_reference_stale_state"].astype("boolean").ffill().fillna(True)
    gas_frame.columns = pd.MultiIndex.from_tuples([("__meta__", column) for column in gas_frame.columns])
    wide = wide.join(gas_frame, how="left")

    base_fee = wide[("__meta__", "base_fee_per_gas_wei")].ffill().fillna(0.0).to_numpy(dtype=float)
    gas_quote_price = wide[("__meta__", "gas_quote_price")].ffill().to_numpy(dtype=float)
    gas_ref_stale = wide[("__meta__", "gas_reference_stale_state")].astype("boolean").fillna(True).to_numpy(dtype=bool)
    gas_cost_native = gas_units * (base_fee + PRIORITY_FEE_GWEI * 1e9) / 1e18
    gas_cost_quote = gas_cost_native * gas_quote_price

    fee_lookup = pair_metadata_df.set_index("dex")["fee_rate"].to_dict()
    stale_any = (
        wide[("stale_state", ATOM_USDT_DEX)].astype("boolean").fillna(True).to_numpy(dtype=bool)
        | wide[("stale_state", ATOM_AXLUSDC_DEX)].astype("boolean").fillna(True).to_numpy(dtype=bool)
        | wide[("stale_state", USDT_AXLUSDC_DEX)].astype("boolean").fillna(True).to_numpy(dtype=bool)
        | gas_ref_stale
    )

    route_a = _evaluate_route(
        route_name="usdt_atom_axlusdc_usdt",
        buy_dex=ATOM_USDT_DEX,
        sell_dex=ATOM_AXLUSDC_DEX,
        conversion_dex=USDT_AXLUSDC_DEX,
        trade_size_quote=trade_size_quote,
        gas_cost_quote=gas_cost_quote,
        step1_reserve_in=wide[("reserve_quote", ATOM_USDT_DEX)].to_numpy(dtype=float),
        step1_reserve_out=wide[("reserve_base", ATOM_USDT_DEX)].to_numpy(dtype=float),
        step2_reserve_in=wide[("reserve_base", ATOM_AXLUSDC_DEX)].to_numpy(dtype=float),
        step2_reserve_out=wide[("reserve_quote", ATOM_AXLUSDC_DEX)].to_numpy(dtype=float),
        step3_reserve_in=wide[("reserve_quote", USDT_AXLUSDC_DEX)].to_numpy(dtype=float),
        step3_reserve_out=wide[("reserve_base", USDT_AXLUSDC_DEX)].to_numpy(dtype=float),
        step1_fee_rate=float(fee_lookup[ATOM_USDT_DEX]),
        step2_fee_rate=float(fee_lookup[ATOM_AXLUSDC_DEX]),
        step3_fee_rate=float(fee_lookup[USDT_AXLUSDC_DEX]),
    )
    route_b = _evaluate_route(
        route_name="usdt_axlusdc_atom_usdt",
        buy_dex=ATOM_AXLUSDC_DEX,
        sell_dex=ATOM_USDT_DEX,
        conversion_dex=USDT_AXLUSDC_DEX,
        trade_size_quote=trade_size_quote,
        gas_cost_quote=gas_cost_quote,
        step1_reserve_in=wide[("reserve_base", USDT_AXLUSDC_DEX)].to_numpy(dtype=float),
        step1_reserve_out=wide[("reserve_quote", USDT_AXLUSDC_DEX)].to_numpy(dtype=float),
        step2_reserve_in=wide[("reserve_quote", ATOM_AXLUSDC_DEX)].to_numpy(dtype=float),
        step2_reserve_out=wide[("reserve_base", ATOM_AXLUSDC_DEX)].to_numpy(dtype=float),
        step3_reserve_in=wide[("reserve_base", ATOM_USDT_DEX)].to_numpy(dtype=float),
        step3_reserve_out=wide[("reserve_quote", ATOM_USDT_DEX)].to_numpy(dtype=float),
        step1_fee_rate=float(fee_lookup[USDT_AXLUSDC_DEX]),
        step2_fee_rate=float(fee_lookup[ATOM_AXLUSDC_DEX]),
        step3_fee_rate=float(fee_lookup[ATOM_USDT_DEX]),
    )

    score_a = np.where(route_a["valid"], route_a["net_edge_quote"], -np.inf)
    score_b = np.where(route_b["valid"], route_b["net_edge_quote"], -np.inf)
    use_a = score_a >= score_b
    valid_any = route_a["valid"] | route_b["valid"]

    def choose(key: str) -> np.ndarray:
        return np.where(use_a, route_a[key], route_b[key])

    out = pd.DataFrame(
        {
            "timestamp": wide.index.to_numpy(),
            "buy_dex": choose("buy_dex"),
            "sell_dex": choose("sell_dex"),
            "conversion_dex": choose("conversion_dex"),
            "route_name": choose("route_name"),
            "trade_size_quote": trade_size_quote,
            "gross_edge_quote": choose("gross_edge_quote"),
            "gross_edge_bps": choose("gross_edge_bps"),
            "fee_cost_quote": choose("fee_cost_quote"),
            "fee_cost_bps": choose("fee_cost_bps"),
            "gas_cost_quote": choose("gas_cost_quote"),
            "gas_cost_bps": choose("gas_cost_bps"),
            "net_edge_quote": choose("net_edge_quote"),
            "net_edge_bps": choose("net_edge_bps"),
            "final_quote_no_fee": choose("final_quote_no_fee"),
            "final_quote_with_fee": choose("final_quote_with_fee"),
            "base_fee_per_gas_wei": base_fee,
            "gas_quote_price": gas_quote_price,
            "stale_state": stale_any,
        }
    )
    out = out[valid_any].copy()
    out["opportunity_flag"] = (out["net_edge_quote"] > 0.0) & (~out["stale_state"])
    return out.sort_values("timestamp").reset_index(drop=True)


def build_opportunity_windows(arb_df: pd.DataFrame) -> pd.DataFrame:
    positive = arb_df[arb_df["opportunity_flag"]].copy().sort_values("timestamp")
    if positive.empty:
        return pd.DataFrame(
            columns=[
                "start_timestamp",
                "end_timestamp",
                "seconds",
                "route_name",
                "buy_dex",
                "sell_dex",
                "conversion_dex",
                "max_net_edge_bps",
                "mean_net_edge_bps",
                "max_net_profit_quote",
                "mean_net_profit_quote",
            ]
        )

    positive["new_window"] = (
        positive["timestamp"].diff().ne(pd.Timedelta(seconds=1))
        | positive["route_name"].ne(positive["route_name"].shift())
    )
    positive["window_id"] = positive["new_window"].cumsum()
    return (
        positive.groupby("window_id", as_index=False)
        .agg(
            start_timestamp=("timestamp", "min"),
            end_timestamp=("timestamp", "max"),
            seconds=("timestamp", "count"),
            route_name=("route_name", "first"),
            buy_dex=("buy_dex", "first"),
            sell_dex=("sell_dex", "first"),
            conversion_dex=("conversion_dex", "first"),
            max_net_edge_bps=("net_edge_bps", "max"),
            mean_net_edge_bps=("net_edge_bps", "mean"),
            max_net_profit_quote=("net_edge_quote", "max"),
            mean_net_profit_quote=("net_edge_quote", "mean"),
        )
        .sort_values(["max_net_edge_bps", "seconds"], ascending=[False, False])
        .reset_index(drop=True)
    )


def build_size_sensitivity(
    pool_state_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    pair_metadata_df: pd.DataFrame,
    gas_reference_state_df: pd.DataFrame,
    *,
    gas_units: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    primary_labels = pd.DataFrame()
    for trade_size_quote in TRADE_SIZES_QUOTE:
        labels = build_triangle_arb_labels(
            pool_state_df,
            blocks_df,
            pair_metadata_df,
            gas_reference_state_df,
            trade_size_quote=trade_size_quote,
            gas_units=gas_units,
        )
        if trade_size_quote == PRIMARY_TRADE_SIZE_QUOTE:
            primary_labels = labels.copy()
        if labels.empty:
            summary_rows.append(
                {
                    "trade_size_quote": trade_size_quote,
                    "observed_seconds": 0,
                    "non_stale_seconds": 0,
                    "gross_positive_seconds": 0,
                    "net_positive_seconds": 0,
                    "max_net_profit_quote": np.nan,
                    "mean_net_profit_quote": np.nan,
                    "max_net_edge_bps": np.nan,
                    "mean_net_edge_bps": np.nan,
                }
            )
            continue

        summary_rows.append(
            {
                "trade_size_quote": trade_size_quote,
                "observed_seconds": int(len(labels)),
                "non_stale_seconds": int((~labels["stale_state"]).sum()),
                "gross_positive_seconds": int((labels["gross_edge_quote"] > 0.0).sum()),
                "net_positive_seconds": int(labels["opportunity_flag"].sum()),
                "max_net_profit_quote": float(labels["net_edge_quote"].max()),
                "mean_net_profit_quote": float(labels["net_edge_quote"].mean()),
                "max_net_edge_bps": float(labels["net_edge_bps"].max()),
                "mean_net_edge_bps": float(labels["net_edge_bps"].mean()),
            }
        )
    return pd.DataFrame(summary_rows), primary_labels


def _cached_dataset_exists(paths: dict[str, Path]) -> bool:
    required = [
        paths["raw_logs"],
        paths["raw_blocks"],
        paths["raw_pair_metadata"],
        paths["events_curated"],
        paths["swaps_raw"],
        paths["pool_state"],
        paths["gas_reference_state"],
        paths["arb_labels"],
        paths["size_sensitivity"],
        paths["opportunity_windows"],
        paths["dataset_manifest"],
        paths["fee_summary"],
    ]
    return all(path.exists() for path in required)


def _load_cached_dataset(paths: dict[str, Path]) -> dict[str, Any]:
    dataset_manifest = _read_json(paths["dataset_manifest"])
    return {
        "status": dataset_manifest["status"],
        "pair_metadata": pd.read_parquet(paths["raw_pair_metadata"]),
        "selected_pairs": pd.read_csv(paths["route_catalog"]),
        "events_curated": pd.read_parquet(paths["events_curated"]),
        "swaps_raw": pd.read_parquet(paths["swaps_raw"]),
        "pool_state": pd.read_parquet(paths["pool_state"]),
        "gas_reference_state": pd.read_parquet(paths["gas_reference_state"]),
        "arb_labels": pd.read_parquet(paths["arb_labels"]),
        "size_sensitivity": pd.read_csv(paths["size_sensitivity"]),
        "opportunity_windows": pd.read_csv(paths["opportunity_windows"], parse_dates=["start_timestamp", "end_timestamp"]) if paths["opportunity_windows"].exists() else pd.DataFrame(),
        "qc_report": _read_json(paths["qc_report"]),
        "dataset_manifest": dataset_manifest,
        "raw_manifest": _read_json(paths["raw_manifest"]),
        "fee_summary": pd.read_csv(paths["fee_summary"]),
        "tx_receipts": pd.read_parquet(paths["tx_receipts"]) if paths["tx_receipts"].exists() else pd.DataFrame(),
    }


def _write_outputs(
    paths: dict[str, Path],
    *,
    logs_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    pair_metadata_df: pd.DataFrame,
    selected_pairs_df: pd.DataFrame,
    events_df: pd.DataFrame,
    swaps_df: pd.DataFrame,
    pool_state_df: pd.DataFrame,
    gas_reference_state_df: pd.DataFrame,
    arb_labels_df: pd.DataFrame,
    size_sensitivity_df: pd.DataFrame,
    opportunity_windows_df: pd.DataFrame,
    qc_report: dict[str, Any],
    raw_manifest: dict[str, Any],
    dataset_manifest: dict[str, Any],
    fee_summary_df: pd.DataFrame,
    tx_receipts_df: pd.DataFrame,
) -> None:
    for key in ["raw_dir", "curated_dir", "report_dir", "metadata_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    logs_write = _parquet_safe_frame(logs_df)
    blocks_write = _parquet_safe_frame(blocks_df)
    pair_metadata_write = _parquet_safe_frame(pair_metadata_df)
    events_write = _parquet_safe_frame(events_df)
    swaps_write = _parquet_safe_frame(swaps_df)
    pool_state_write = _parquet_safe_frame(pool_state_df)
    gas_reference_write = _parquet_safe_frame(gas_reference_state_df)
    arb_labels_write = _parquet_safe_frame(arb_labels_df)
    tx_receipts_write = _parquet_safe_frame(tx_receipts_df)

    logs_write.to_parquet(paths["raw_logs"], index=False)
    _write_preview_csv(paths["raw_logs_preview"], logs_write, ["role", "dex", "pair_address", "block_number", "transaction_hash", "log_index", "event_name", "topic0"])
    blocks_write.to_parquet(paths["raw_blocks"], index=False)
    _write_preview_csv(paths["raw_blocks_preview"], blocks_write, ["block_number", "block_timestamp", "base_fee_per_gas_wei", "gas_used", "gas_limit"])
    pair_metadata_write.to_parquet(paths["raw_pair_metadata"], index=False)
    _write_preview_csv(paths["raw_pair_metadata_preview"], pair_metadata_write, ["role", "dex", "pair_address", "event_style", "base_symbol", "quote_symbol", "fee_bps"])
    selected_pairs_df.to_csv(paths["route_catalog"], index=False)
    fee_summary_df.to_csv(paths["fee_summary"], index=False)
    if not tx_receipts_df.empty:
        tx_receipts_write.to_parquet(paths["tx_receipts"], index=False)
        _write_preview_csv(paths["tx_receipts_preview"], tx_receipts_write, ["transaction_hash", "block_number", "gas_used", "effective_gas_price_wei", "status"])
    _write_json(paths["raw_manifest"], raw_manifest)

    events_write.to_parquet(paths["events_curated"], index=False)
    _write_preview_csv(paths["events_curated_preview"], events_write, ["timestamp", "role", "dex", "pair_address", "block_number", "transaction_hash", "log_index", "event_name", "reserve_base_post", "reserve_quote_post", "mid_price_quote_per_base"])
    swaps_write.to_parquet(paths["swaps_raw"], index=False)
    _write_preview_csv(paths["swaps_raw_preview"], swaps_write, ["timestamp", "role", "dex", "pair_address", "block_number", "transaction_hash", "log_index", "amount_base", "amount_quote", "volume_base", "volume_quote", "trade_direction"])
    pool_state_write.to_parquet(paths["pool_state"], index=False)
    _write_preview_csv(paths["pool_state_preview"], pool_state_write, ["timestamp", "role", "dex", "pair_address", "mid_price_quote_per_base", "reserve_base", "reserve_quote", "swap_count", "volume_base", "volume_quote", "stale_state"])
    gas_reference_write.to_parquet(paths["gas_reference_state"], index=False)
    _write_preview_csv(paths["gas_reference_state_preview"], gas_reference_write, ["timestamp", "role", "dex", "pair_address", "mid_price_quote_per_base", "reserve_base", "reserve_quote", "stale_state"])
    arb_labels_write.to_parquet(paths["arb_labels"], index=False)
    _write_preview_csv(paths["arb_labels_preview"], arb_labels_write, ["timestamp", "buy_dex", "sell_dex", "conversion_dex", "trade_size_quote", "gross_edge_bps", "fee_cost_bps", "gas_cost_quote", "net_edge_quote", "net_edge_bps", "opportunity_flag", "stale_state"])
    _write_json(paths["qc_report"], qc_report)
    size_sensitivity_df.to_csv(paths["size_sensitivity"], index=False)
    opportunity_windows_df.to_csv(paths["opportunity_windows"], index=False)
    _write_json(paths["dataset_manifest"], dataset_manifest)


def load_or_build_cosmos_dataset(
    project_root: str | Path | None = None,
    *,
    refresh: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    project_root = Path(project_root or Path.cwd()).resolve()
    paths = cosmos_paths(project_root)
    if not refresh and _cached_dataset_exists(paths):
        return _load_cached_dataset(paths)

    log = print if verbose else (lambda *args, **kwargs: None)
    rpc_urls, env_var, rpc_source = select_rpc_urls(CHAIN_KEY)
    rpc = RpcClient(rpc_urls, paths["rpc_cache_dir"])
    start_block, end_block, start_time, end_time = _resolve_block_window(rpc)
    log(f"[cosmos_second_level] window {start_time.isoformat()} -> {end_time.isoformat()} ({start_block} -> {end_block})")

    pair_specs = [*ROUTE_PAIRS, GAS_REFERENCE_PAIR]
    pair_metadata_df = build_pair_metadata(rpc, pair_specs)
    log(f"[cosmos_second_level] pair metadata rows: {len(pair_metadata_df)}")

    log_frames: list[pd.DataFrame] = []
    for pair in pair_specs:
        log(f"[cosmos_second_level] collecting logs for {pair.dex}")
        rows = collect_equilibre_pair_logs(
            rpc,
            pair,
            start_block=start_block,
            end_block=end_block,
            chunk_size=BLOCK_CHUNK_SIZE,
            lookback_blocks=ANCHOR_LOOKBACK_BLOCKS,
        )
        log(f"[cosmos_second_level] collected {len(rows)} logs for {pair.dex}")
        if rows:
            log_frames.append(pd.DataFrame(rows))

    logs_df = pd.concat(log_frames, ignore_index=True).sort_values(["block_number", "log_index"]).reset_index(drop=True)
    if logs_df.empty:
        raise RuntimeError("No Equilibre logs were collected for the Cosmos proxy notebook")
    log(f"[cosmos_second_level] log rows: {len(logs_df)}")

    blocks_df = fetch_block_headers(rpc, logs_df["block_number"].drop_duplicates().tolist())
    events_df = normalize_event_logs(logs_df, blocks_df, pair_metadata_df)
    swaps_df = build_swaps_raw(events_df)
    fee_summary_df, pair_metadata_df = _infer_fee_rates(events_df, pair_metadata_df)
    selected_pairs_df = _summarize_selected_pools(pair_metadata_df)
    log(f"[cosmos_second_level] events={len(events_df)} swaps={len(swaps_df)}")

    pool_state_df = build_pool_state(
        events_df,
        swaps_df,
        role="arb_pair",
        stale_after_minutes=STALE_AFTER_MINUTES,
        frequency=STATE_FREQUENCY,
        window_start=start_time,
        window_end=end_time,
    )
    log(f"[cosmos_second_level] route pool state rows: {len(pool_state_df)}")
    gas_reference_state_df = build_pool_state(
        events_df,
        swaps_df,
        role="gas_reference",
        stale_after_minutes=STALE_AFTER_MINUTES,
        frequency=STATE_FREQUENCY,
        window_start=start_time,
        window_end=end_time,
    )
    log(f"[cosmos_second_level] gas reference state rows: {len(gas_reference_state_df)}")
    tx_receipts_df = _fetch_transaction_receipts(rpc, swaps_df["transaction_hash"].dropna().tolist())
    gas_units, gas_diagnostics = _estimate_gas_units(events_df, tx_receipts_df)
    log(f"[cosmos_second_level] inferred gas units: {gas_units} ({gas_diagnostics['gas_estimation_source']})")

    size_sensitivity_df, arb_labels_df = build_size_sensitivity(
        pool_state_df,
        blocks_df,
        pair_metadata_df,
        gas_reference_state_df,
        gas_units=gas_units,
    )
    opportunity_windows_df = build_opportunity_windows(arb_labels_df)
    qc_report = run_qc(swaps_df, pool_state_df, arb_labels_df, frequency=STATE_FREQUENCY)
    log(
        "[cosmos_second_level] size_sensitivity_rows="
        f"{len(size_sensitivity_df)} primary_positive_seconds="
        f"{int(arb_labels_df['opportunity_flag'].sum()) if not arb_labels_df.empty else 0}"
    )

    raw_manifest = {
        "status": "supported",
        "market_key": MARKET_KEY,
        "market_slug": MARKET_SLUG,
        "chain_key": CHAIN_KEY,
        "chain_name": CHAIN_NAME,
        "pair_label": PAIR_LABEL,
        "sample_window_days": SAMPLE_WINDOW_DAYS,
        "state_frequency": STATE_FREQUENCY,
        "window_start_utc": start_time.isoformat(),
        "window_end_utc": end_time.isoformat(),
        "start_block": start_block,
        "end_block": end_block,
        "rpc_env_var": env_var,
        "rpc_source": rpc_source,
        "rpc_urls": rpc_urls,
        "selected_pairs": selected_pairs_df.to_dict("records"),
        "notes": [
            "This notebook remains a Cosmos proxy via Kava EVM rather than a direct Osmosis backtest.",
            "The route uses three reserve-based Equilibre pools and an explicit USDt/axlUSDC conversion pool.",
            "Wagmi and Kinetix ATOM/USDt pools were excluded because they expose CLMM surfaces rather than notebook-02-style reserve state.",
        ],
    }
    dataset_manifest = {
        "status": "supported",
        "market_key": MARKET_KEY,
        "market_slug": MARKET_SLUG,
        "chain_key": CHAIN_KEY,
        "chain_name": CHAIN_NAME,
        "pair_label": PAIR_LABEL,
        "sample_window_days": SAMPLE_WINDOW_DAYS,
        "state_frequency": STATE_FREQUENCY,
        "window_start_utc": start_time.isoformat(),
        "window_end_utc": end_time.isoformat(),
        "primary_trade_size_quote": PRIMARY_TRADE_SIZE_QUOTE,
        "trade_sizes_quote": list(TRADE_SIZES_QUOTE),
        "stale_after_minutes": STALE_AFTER_MINUTES,
        "gas_units": gas_units,
        "priority_fee_gwei": PRIORITY_FEE_GWEI,
        "gas_estimation": gas_diagnostics,
        "selected_pairs": selected_pairs_df.to_dict("records"),
        "fee_summary": fee_summary_df.to_dict("records"),
        "pair_metadata_rows": int(len(pair_metadata_df)),
        "event_log_rows": int(len(logs_df)),
        "events_curated_rows": int(len(events_df)),
        "swaps_raw_rows": int(len(swaps_df)),
        "pool_state_rows": int(len(pool_state_df)),
        "gas_reference_state_rows": int(len(gas_reference_state_df)),
        "arb_label_rows": int(len(arb_labels_df)),
        "size_sensitivity_rows": int(len(size_sensitivity_df)),
        "positive_seconds": int(arb_labels_df["opportunity_flag"].sum()) if not arb_labels_df.empty else 0,
        "positive_windows": int(len(opportunity_windows_df)),
        "notes": [
            "Real data only: Kava EVM RPC logs, blocks, receipts, and on-chain fee events.",
            "No stablecoin parity assumption is used; USDt/axlUSDC conversion is modeled through its own pool.",
            "This is a three-pool route screen aligned to notebook 02 outputs rather than a two-pool same-quote backtest.",
        ],
    }

    _write_outputs(
        paths,
        logs_df=logs_df,
        blocks_df=blocks_df,
        pair_metadata_df=pair_metadata_df,
        selected_pairs_df=selected_pairs_df,
        events_df=events_df,
        swaps_df=swaps_df,
        pool_state_df=pool_state_df,
        gas_reference_state_df=gas_reference_state_df,
        arb_labels_df=arb_labels_df,
        size_sensitivity_df=size_sensitivity_df,
        opportunity_windows_df=opportunity_windows_df,
        qc_report=qc_report,
        raw_manifest=raw_manifest,
        dataset_manifest=dataset_manifest,
        fee_summary_df=fee_summary_df,
        tx_receipts_df=tx_receipts_df,
    )

    return {
        "status": "supported",
        "pair_metadata": pair_metadata_df,
        "selected_pairs": selected_pairs_df,
        "events_curated": events_df,
        "swaps_raw": swaps_df,
        "pool_state": pool_state_df,
        "gas_reference_state": gas_reference_state_df,
        "arb_labels": arb_labels_df,
        "size_sensitivity": size_sensitivity_df,
        "opportunity_windows": opportunity_windows_df,
        "qc_report": qc_report,
        "dataset_manifest": dataset_manifest,
        "raw_manifest": raw_manifest,
        "fee_summary": fee_summary_df,
        "tx_receipts": tx_receipts_df,
    }
