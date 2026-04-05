"""Normalization and decoding of raw Ethereum logs into research tables."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from mev_dataset.config import MarketConfig


def _decode_words(data: str) -> list[int]:
    payload = data[2:] if data.startswith("0x") else data
    if not payload:
        return []
    return [int(payload[i : i + 64], 16) for i in range(0, len(payload), 64)]


def _scale_amount(raw_amount: int | float | None, decimals: int) -> float:
    if raw_amount is None or pd.isna(raw_amount):
        return float("nan")
    return float(raw_amount) / float(10**decimals)


def _decode_event_fields(event_name: str, data: str) -> dict[str, float]:
    words = _decode_words(data)
    if event_name == "Sync":
        return {
            "reserve0_raw": words[0],
            "reserve1_raw": words[1],
        }
    if event_name == "Swap":
        return {
            "amount0_in_raw": words[0],
            "amount1_in_raw": words[1],
            "amount0_out_raw": words[2],
            "amount1_out_raw": words[3],
        }
    if event_name in {"Mint", "Burn"}:
        return {
            "amount0_raw": words[0],
            "amount1_raw": words[1],
        }
    return {}


def normalize_event_logs(
    logs_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    pair_metadata_df: pd.DataFrame,
    config: MarketConfig,
) -> pd.DataFrame:
    if logs_df.empty:
        return pd.DataFrame()

    merged = logs_df.merge(blocks_df, on="block_number", how="left").merge(
        pair_metadata_df,
        on=["dex", "pair_address"],
        how="left",
        suffixes=("", "_meta"),
    )
    decoded = pd.DataFrame([_decode_event_fields(row.event_name, row.data) for row in merged.itertuples()])
    events = pd.concat([merged.reset_index(drop=True), decoded], axis=1)
    events = events.sort_values(["pair_address", "block_number", "log_index"]).reset_index(drop=True)
    wbtc_decimals = config.tokens["wbtc"].decimals
    weth_decimals = config.tokens["weth"].decimals

    events["reserve_wbtc_post"] = np.where(
        events["wbtc_is_token0"],
        events.get("reserve0_raw"),
        events.get("reserve1_raw"),
    )
    events["reserve_weth_post"] = np.where(
        events["wbtc_is_token0"],
        events.get("reserve1_raw"),
        events.get("reserve0_raw"),
    )
    events["reserve_wbtc_post"] = events.groupby("pair_address")["reserve_wbtc_post"].ffill()
    events["reserve_weth_post"] = events.groupby("pair_address")["reserve_weth_post"].ffill()
    events["reserve_wbtc_post"] = events["reserve_wbtc_post"].map(lambda x: _scale_amount(x, wbtc_decimals))
    events["reserve_weth_post"] = events["reserve_weth_post"].map(lambda x: _scale_amount(x, weth_decimals))
    events["mid_price_eth_per_btc"] = events["reserve_weth_post"] / events["reserve_wbtc_post"]
    events = events.rename(columns={"block_timestamp": "timestamp"})

    return events


def build_swaps_raw(events_df: pd.DataFrame, config: MarketConfig) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    swaps = events_df[events_df["event_name"] == "Swap"].copy()
    if swaps.empty:
        return swaps

    wbtc_decimals = config.tokens["wbtc"].decimals
    weth_decimals = config.tokens["weth"].decimals

    swaps["amount0_net_raw"] = swaps["amount0_in_raw"] - swaps["amount0_out_raw"]
    swaps["amount1_net_raw"] = swaps["amount1_in_raw"] - swaps["amount1_out_raw"]
    swaps["amount_wbtc"] = np.where(swaps["wbtc_is_token0"], swaps["amount0_net_raw"], swaps["amount1_net_raw"])
    swaps["amount_weth"] = np.where(swaps["wbtc_is_token0"], swaps["amount1_net_raw"], swaps["amount0_net_raw"])
    swaps["amount_wbtc"] = swaps["amount_wbtc"].map(lambda x: _scale_amount(x, wbtc_decimals))
    swaps["amount_weth"] = swaps["amount_weth"].map(lambda x: _scale_amount(x, weth_decimals))
    swaps["volume_wbtc"] = swaps["amount_wbtc"].abs()
    swaps["volume_weth"] = swaps["amount_weth"].abs()
    swaps["trade_direction"] = np.where(swaps["amount_wbtc"] < 0, "buy_wbtc", "sell_wbtc")

    columns = [
        "timestamp",
        "block_number",
        "dex",
        "pair_address",
        "transaction_hash",
        "log_index",
        "amount_wbtc",
        "amount_weth",
        "volume_wbtc",
        "volume_weth",
        "reserve_wbtc_post",
        "reserve_weth_post",
        "mid_price_eth_per_btc",
        "trade_direction",
    ]
    return swaps[columns].sort_values(["block_number", "log_index"]).reset_index(drop=True)
