"""Feature engineering and arbitrage labeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mev_dataset.config import GasConfig


@dataclass(frozen=True)
class ArbDirectionResult:
    buy_dex: str
    sell_dex: str
    gross_edge_weth: float
    gross_edge_bps: float
    fee_cost_weth: float
    fee_cost_bps: float
    gas_cost_weth: float
    net_edge_weth: float
    net_edge_bps: float


def amount_out_for_exact_in(amount_in: float, reserve_in: float, reserve_out: float, fee_rate: float = 0.0) -> float:
    if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
        return 0.0
    effective_in = amount_in * (1.0 - fee_rate)
    return reserve_out * effective_in / (reserve_in + effective_in)


def amount_in_for_exact_out(amount_out: float, reserve_in: float, reserve_out: float, fee_rate: float = 0.0) -> float:
    if amount_out <= 0 or reserve_in <= 0 or reserve_out <= amount_out:
        raise ValueError("invalid reserves or target amount_out for constant-product calculation")
    raw_required = reserve_in * amount_out / (reserve_out - amount_out)
    if fee_rate >= 1.0:
        raise ValueError("fee_rate must be less than 1.0")
    return raw_required / (1.0 - fee_rate)


def gas_cost_weth(base_fee_per_gas_wei: float, gas: GasConfig) -> float:
    total_gas_price_wei = float(base_fee_per_gas_wei) + gas.priority_fee_gwei * 1e9
    return gas.gas_units * total_gas_price_wei / 1e18


def gas_cost_bps(cost_weth: float, reference_weth: float) -> float:
    if reference_weth <= 0:
        return float("nan")
    return cost_weth / reference_weth * 10_000.0


def build_pool_state_1m(events_df: pd.DataFrame, swaps_df: pd.DataFrame, stale_after_minutes: int = 15) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    syncs = events_df[events_df["event_name"] == "Sync"].copy()
    if syncs.empty:
        raise ValueError("No Sync events available. Pool state cannot be reconstructed.")

    swap_stats = (
        swaps_df.assign(timestamp=swaps_df["timestamp"].dt.floor("1min"))
        .groupby(["dex", "pair_address", "timestamp"], as_index=False)
        .agg(
            swap_count=("transaction_hash", "count"),
            volume_wbtc=("volume_wbtc", "sum"),
            volume_weth=("volume_weth", "sum"),
        )
    )

    frames: list[pd.DataFrame] = []
    for (dex, pair_address), group in syncs.groupby(["dex", "pair_address"]):
        state = group[["timestamp", "reserve_wbtc_post", "reserve_weth_post"]].copy()
        state["timestamp"] = pd.to_datetime(state["timestamp"], utc=True)
        state["last_sync_timestamp"] = state["timestamp"]
        state = (
            state.set_index("timestamp")
            .resample("1min")
            .last()
            .ffill()
            .rename_axis("timestamp")
            .reset_index()
        )
        state["dex"] = dex
        state["pair_address"] = pair_address
        stats = swap_stats[(swap_stats["dex"] == dex) & (swap_stats["pair_address"] == pair_address)]
        state = state.merge(stats, on=["dex", "pair_address", "timestamp"], how="left")
        state[["swap_count", "volume_wbtc", "volume_weth"]] = state[["swap_count", "volume_wbtc", "volume_weth"]].fillna(0.0)
        state["swap_count"] = state["swap_count"].astype(int)
        state["mid_price_eth_per_btc"] = state["reserve_weth_post"] / state["reserve_wbtc_post"]
        state["stale_state"] = (
            state["timestamp"] - state["last_sync_timestamp"]
        ) > pd.Timedelta(minutes=stale_after_minutes)
        state = state.rename(
            columns={
                "reserve_wbtc_post": "reserve_wbtc",
                "reserve_weth_post": "reserve_weth",
            }
        )
        frames.append(state)

    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "dex"]).reset_index(drop=True)


def build_base_fee_1m(blocks_df: pd.DataFrame) -> pd.DataFrame:
    blocks = blocks_df.copy()
    blocks["block_timestamp"] = pd.to_datetime(blocks["block_timestamp"], utc=True)
    return (
        blocks.set_index("block_timestamp")[["base_fee_per_gas_wei"]]
        .resample("1min")
        .last()
        .ffill()
        .rename_axis("timestamp")
        .reset_index()
    )


def evaluate_arbitrage_direction(
    buy_dex: str,
    sell_dex: str,
    buy_reserve_wbtc: float,
    buy_reserve_weth: float,
    sell_reserve_wbtc: float,
    sell_reserve_weth: float,
    buy_fee_rate: float,
    sell_fee_rate: float,
    trade_size_wbtc: float,
    gas_cost_in_weth: float,
) -> ArbDirectionResult:
    buy_cost_no_fee = amount_in_for_exact_out(
        trade_size_wbtc,
        reserve_in=buy_reserve_weth,
        reserve_out=buy_reserve_wbtc,
        fee_rate=0.0,
    )
    buy_cost_with_fee = amount_in_for_exact_out(
        trade_size_wbtc,
        reserve_in=buy_reserve_weth,
        reserve_out=buy_reserve_wbtc,
        fee_rate=buy_fee_rate,
    )
    sell_return_no_fee = amount_out_for_exact_in(
        trade_size_wbtc,
        reserve_in=sell_reserve_wbtc,
        reserve_out=sell_reserve_weth,
        fee_rate=0.0,
    )
    sell_return_with_fee = amount_out_for_exact_in(
        trade_size_wbtc,
        reserve_in=sell_reserve_wbtc,
        reserve_out=sell_reserve_weth,
        fee_rate=sell_fee_rate,
    )
    gross_edge_weth = sell_return_no_fee - buy_cost_no_fee
    fee_cost_weth = (buy_cost_with_fee - buy_cost_no_fee) + (sell_return_no_fee - sell_return_with_fee)
    net_edge_weth = sell_return_with_fee - buy_cost_with_fee - gas_cost_in_weth
    denominator = buy_cost_with_fee
    return ArbDirectionResult(
        buy_dex=buy_dex,
        sell_dex=sell_dex,
        gross_edge_weth=gross_edge_weth,
        gross_edge_bps=(gross_edge_weth / denominator) * 10_000.0,
        fee_cost_weth=fee_cost_weth,
        fee_cost_bps=(fee_cost_weth / denominator) * 10_000.0,
        gas_cost_weth=gas_cost_in_weth,
        net_edge_weth=net_edge_weth,
        net_edge_bps=(net_edge_weth / denominator) * 10_000.0,
    )


def build_arb_labels_1m(
    pool_state_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    pair_metadata_df: pd.DataFrame,
    notional_wbtc: float,
    gas: GasConfig,
) -> pd.DataFrame:
    if pool_state_df.empty:
        return pd.DataFrame()

    fee_lookup = pair_metadata_df.set_index("dex")["fee_rate"].to_dict()
    gas_series = build_base_fee_1m(blocks_df)
    dexes = sorted(pool_state_df["dex"].dropna().unique().tolist())
    if len(dexes) != 2:
        raise ValueError("Arbitrage labeling expects exactly two DEX venues.")

    wide = pool_state_df.pivot(index="timestamp", columns="dex", values=["reserve_wbtc", "reserve_weth", "stale_state"])
    gas_wide = gas_series.set_index("timestamp")
    gas_wide.columns = pd.MultiIndex.from_tuples([("__meta__", column) for column in gas_wide.columns])
    wide = wide.join(gas_wide, how="left")
    wide[("__meta__", "base_fee_per_gas_wei")] = wide[("__meta__", "base_fee_per_gas_wei")].ffill()

    records: list[dict[str, Any]] = []
    for timestamp, row in wide.iterrows():
        base_fee = float(row.get(("__meta__", "base_fee_per_gas_wei"), 0.0) or 0.0)
        gas_weth = gas_cost_weth(base_fee, gas)
        candidates: list[ArbDirectionResult] = []
        for buy_dex, sell_dex in [(dexes[0], dexes[1]), (dexes[1], dexes[0])]:
            buy_reserve_wbtc = row[("reserve_wbtc", buy_dex)]
            buy_reserve_weth = row[("reserve_weth", buy_dex)]
            sell_reserve_wbtc = row[("reserve_wbtc", sell_dex)]
            sell_reserve_weth = row[("reserve_weth", sell_dex)]
            if any(pd.isna(value) for value in [buy_reserve_wbtc, buy_reserve_weth, sell_reserve_wbtc, sell_reserve_weth]):
                continue
            try:
                candidate = evaluate_arbitrage_direction(
                    buy_dex=buy_dex,
                    sell_dex=sell_dex,
                    buy_reserve_wbtc=float(buy_reserve_wbtc),
                    buy_reserve_weth=float(buy_reserve_weth),
                    sell_reserve_wbtc=float(sell_reserve_wbtc),
                    sell_reserve_weth=float(sell_reserve_weth),
                    buy_fee_rate=float(fee_lookup[buy_dex]),
                    sell_fee_rate=float(fee_lookup[sell_dex]),
                    trade_size_wbtc=notional_wbtc,
                    gas_cost_in_weth=gas_weth,
                )
            except ValueError:
                continue
            candidates.append(candidate)
        if not candidates:
            continue
        best = max(candidates, key=lambda item: item.net_edge_weth)
        is_stale = bool(row[("stale_state", best.buy_dex)] or row[("stale_state", best.sell_dex)])
        records.append(
            {
                "timestamp": timestamp,
                "buy_dex": best.buy_dex,
                "sell_dex": best.sell_dex,
                "gross_edge_weth": best.gross_edge_weth,
                "gross_edge_bps": best.gross_edge_bps,
                "fee_cost_weth": best.fee_cost_weth,
                "fee_cost_bps": best.fee_cost_bps,
                "gas_cost_weth": best.gas_cost_weth,
                "gas_cost_bps": gas_cost_bps(best.gas_cost_weth, best.gross_edge_weth + best.fee_cost_weth + 1e-12),
                "net_edge_weth": best.net_edge_weth,
                "net_edge_bps": best.net_edge_bps,
                "base_fee_per_gas_wei": base_fee,
                "opportunity_flag": bool(best.net_edge_weth > 0 and not is_stale),
                "stale_state": is_stale,
            }
        )
    return pd.DataFrame.from_records(records).sort_values("timestamp").reset_index(drop=True)
