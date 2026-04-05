"""Quality-control checks for curated datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "swaps_raw": [
        "timestamp",
        "block_number",
        "dex",
        "pair_address",
        "transaction_hash",
        "log_index",
        "amount_wbtc",
        "amount_weth",
        "reserve_wbtc_post",
        "reserve_weth_post",
    ],
    "pool_state_1m": [
        "timestamp",
        "dex",
        "pair_address",
        "mid_price_eth_per_btc",
        "reserve_wbtc",
        "reserve_weth",
        "swap_count",
        "volume_wbtc",
        "volume_weth",
        "stale_state",
    ],
    "arb_labels_1m": [
        "timestamp",
        "buy_dex",
        "sell_dex",
        "gross_edge_bps",
        "fee_cost_bps",
        "gas_cost_weth",
        "net_edge_bps",
        "opportunity_flag",
    ],
}


def _schema_errors(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    return [column for column in required_columns if column not in df.columns]


def run_qc(swaps_raw: pd.DataFrame, pool_state_1m: pd.DataFrame, arb_labels_1m: pd.DataFrame) -> dict[str, Any]:
    gaps = {}
    if not pool_state_1m.empty:
        for dex, group in pool_state_1m.groupby("dex"):
            expected = pd.date_range(group["timestamp"].min(), group["timestamp"].max(), freq="1min", tz="UTC")
            gaps[dex] = int(len(expected.difference(pd.DatetimeIndex(group["timestamp"]))))

    monotonic_violations = 0
    if not swaps_raw.empty:
        ordered = swaps_raw.sort_values(["block_number", "log_index"]).copy()
        monotonic_violations = int((ordered["timestamp"].diff().dropna() < pd.Timedelta(0)).sum())

    report = {
        "schema_errors": {
            name: _schema_errors(dataset, REQUIRED_COLUMNS[name])
            for name, dataset in {
                "swaps_raw": swaps_raw,
                "pool_state_1m": pool_state_1m,
                "arb_labels_1m": arb_labels_1m,
            }.items()
        },
        "duplicate_swap_events": int(swaps_raw.duplicated(["transaction_hash", "log_index"]).sum()) if not swaps_raw.empty else 0,
        "monotonic_timestamp_violations": monotonic_violations,
        "non_positive_reserves": int(
            ((pool_state_1m["reserve_wbtc"] <= 0) | (pool_state_1m["reserve_weth"] <= 0)).sum()
        ) if not pool_state_1m.empty else 0,
        "pool_state_gaps_1m": gaps,
        "stale_state_nulls": int(pool_state_1m["stale_state"].isna().sum()) if not pool_state_1m.empty else 0,
    }
    report["passed"] = (
        all(len(errors) == 0 for errors in report["schema_errors"].values())
        and report["duplicate_swap_events"] == 0
        and report["monotonic_timestamp_violations"] == 0
        and report["non_positive_reserves"] == 0
        and all(value == 0 for value in report["pool_state_gaps_1m"].values())
        and report["stale_state_nulls"] == 0
    )
    return report


def write_qc_report(output_dir: str | Path, report: dict[str, Any]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "qc_report.json").write_text(json.dumps(report, indent=2, default=str))
