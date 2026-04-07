"""Raw log and block extraction for the WBTC/WETH market."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mev_dataset.config import MarketConfig
from mev_dataset.constants import PAIR_EVENT_TOPICS, TOPIC_TO_EVENT
from mev_dataset.discovery import PairMetadata, pair_metadata_frame
from mev_dataset.rpc import RpcClient


RAW_LOG_COLUMNS = [
    "dex",
    "pair_address",
    "block_number",
    "transaction_hash",
    "log_index",
    "event_name",
    "topic0",
    "topic1",
    "topic2",
    "topic3",
    "data",
    "removed",
]

PREVIEW_ROW_LIMIT = 5000
RAW_PREVIEW_COLUMNS = {
    "event_logs": [
        "dex",
        "pair_address",
        "block_number",
        "transaction_hash",
        "log_index",
        "event_name",
        "topic0",
        "removed",
    ],
    "block_headers": [
        "block_number",
        "block_timestamp",
        "base_fee_per_gas_wei",
        "gas_used",
        "gas_limit",
    ],
    "pair_metadata": [
        "dex",
        "factory_address",
        "pair_address",
        "token0",
        "token1",
        "wbtc_is_token0",
        "fee_rate",
    ],
}


def _empty_raw_logs() -> pd.DataFrame:
    return pd.DataFrame(columns=RAW_LOG_COLUMNS)


def _preview_frame(df: pd.DataFrame, preferred_columns: list[str], max_rows: int = PREVIEW_ROW_LIMIT) -> pd.DataFrame:
    columns = [column for column in preferred_columns if column in df.columns]
    preview = df[columns].head(max_rows) if columns else df.head(max_rows)
    return preview


def resolve_block_window(
    config: MarketConfig,
    rpc: RpcClient,
    start_block: int | None = None,
    end_block: int | None = None,
) -> tuple[int, int, datetime, datetime]:
    start_time, end_time = config.resolve_window()
    resolved_start_block = start_block or rpc.find_block_by_timestamp(start_time, direction="after")
    resolved_end_block = end_block or rpc.find_block_by_timestamp(end_time, direction="before")
    return resolved_start_block, resolved_end_block, start_time, end_time


def collect_pair_logs(
    rpc: RpcClient,
    pair: PairMetadata,
    start_block: int,
    end_block: int,
    chunk_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk_start in range(start_block, end_block + 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size - 1, end_block)
        logs = rpc.eth_get_logs(
            address=pair.pair_address,
            from_block=chunk_start,
            to_block=chunk_end,
            topics=[PAIR_EVENT_TOPICS],
            use_cache=True,
        )
        for log in logs:
            topic0 = log["topics"][0]
            rows.append(
                {
                    "dex": pair.dex,
                    "pair_address": pair.pair_address,
                    "block_number": int(log["blockNumber"], 16),
                    "transaction_hash": log["transactionHash"],
                    "log_index": int(log["logIndex"], 16),
                    "event_name": TOPIC_TO_EVENT.get(topic0, "Unknown"),
                    "topic0": topic0,
                    "topic1": log["topics"][1] if len(log["topics"]) > 1 else None,
                    "topic2": log["topics"][2] if len(log["topics"]) > 2 else None,
                    "topic3": log["topics"][3] if len(log["topics"]) > 3 else None,
                    "data": log["data"],
                    "removed": bool(log.get("removed", False)),
                }
            )
    return rows


def fetch_block_headers(rpc: RpcClient, block_numbers: list[int]) -> pd.DataFrame:
    headers = rpc.get_blocks_by_number(sorted(set(block_numbers)), use_cache=True, batch_size=100)
    return pd.DataFrame(
        {
            "block_number": [header.block_number for header in headers],
            "block_hash": [header.block_hash for header in headers],
            "parent_hash": [header.parent_hash for header in headers],
            "block_timestamp": [header.block_timestamp for header in headers],
            "base_fee_per_gas_wei": [header.base_fee_per_gas_wei for header in headers],
            "gas_used": [header.gas_used for header in headers],
            "gas_limit": [header.gas_limit for header in headers],
        }
    )


def collect_chain_data(
    config: MarketConfig,
    rpc: RpcClient,
    pairs: list[PairMetadata],
    start_block: int | None = None,
    end_block: int | None = None,
    chunk_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    resolved_start_block, resolved_end_block, start_time, end_time = resolve_block_window(
        config,
        rpc,
        start_block=start_block,
        end_block=end_block,
    )
    all_rows: list[dict[str, Any]] = []
    pair_frame = pair_metadata_frame(pairs)
    for pair in pairs:
        all_rows.extend(
            collect_pair_logs(
                rpc=rpc,
                pair=pair,
                start_block=resolved_start_block,
                end_block=resolved_end_block,
                chunk_size=chunk_size or config.block_chunk_size,
            )
        )
    logs_df = pd.DataFrame(all_rows) if all_rows else _empty_raw_logs()
    if not logs_df.empty:
        logs_df = logs_df.sort_values(["block_number", "log_index", "dex"]).reset_index(drop=True)
        block_numbers = logs_df["block_number"].drop_duplicates().tolist()
    else:
        block_numbers = [resolved_start_block, resolved_end_block]
    blocks_df = fetch_block_headers(rpc, block_numbers)
    manifest = {
        "market": config.market,
        "chain_id": config.chain_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "window": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "start_block": resolved_start_block,
            "end_block": resolved_end_block,
        },
        "pairs": pair_frame.to_dict(orient="records"),
    }
    return logs_df, blocks_df, pair_frame, manifest


def write_raw_outputs(
    config: MarketConfig,
    logs_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    pair_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    raw_dir = Path(config.raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    logs_df.to_parquet(raw_dir / "event_logs.parquet", index=False)
    logs_df.to_csv(raw_dir / "event_logs.csv", index=False)
    _preview_frame(logs_df, RAW_PREVIEW_COLUMNS["event_logs"]).to_csv(raw_dir / "event_logs_preview.csv", index=False)
    blocks_df.to_parquet(raw_dir / "block_headers.parquet", index=False)
    blocks_df.to_csv(raw_dir / "block_headers.csv", index=False)
    _preview_frame(blocks_df, RAW_PREVIEW_COLUMNS["block_headers"]).to_csv(raw_dir / "block_headers_preview.csv", index=False)
    pair_frame.to_parquet(raw_dir / "pair_metadata.parquet", index=False)
    pair_frame.to_csv(raw_dir / "pair_metadata.csv", index=False)
    _preview_frame(pair_frame, RAW_PREVIEW_COLUMNS["pair_metadata"]).to_csv(raw_dir / "pair_metadata_preview.csv", index=False)
    (raw_dir / "source_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
