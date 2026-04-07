"""Typer CLI for building the WBTC/WETH dataset."""

from __future__ import annotations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import typer

from mev_dataset.config import MarketConfig, load_market_config
from mev_dataset.discovery import discover_pairs
from mev_dataset.extract import collect_chain_data, write_raw_outputs
from mev_dataset.features import build_arb_labels_1m, build_pool_state_1m
from mev_dataset.normalize import build_swaps_raw, normalize_event_logs
from mev_dataset.qc import run_qc, write_qc_report
from mev_dataset.rpc import RpcClient
from mev_dataset.split import assign_splits, build_split_assignments, write_split_manifest

app = typer.Typer(add_completion=False, no_args_is_help=True)

PREVIEW_ROW_LIMIT = 5000
CURATED_PREVIEW_COLUMNS = {
    "events_curated": [
        "timestamp",
        "dex",
        "pair_address",
        "block_number",
        "transaction_hash",
        "log_index",
        "event_name",
        "reserve_wbtc_post",
        "reserve_weth_post",
        "mid_price_eth_per_btc",
    ],
    "swaps_raw": [
        "timestamp",
        "dex",
        "block_number",
        "transaction_hash",
        "log_index",
        "amount_wbtc",
        "amount_weth",
        "volume_wbtc",
        "volume_weth",
        "mid_price_eth_per_btc",
        "trade_direction",
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
        "split",
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
        "stale_state",
        "split",
    ],
    "split_assignments": [
        "timestamp",
        "split",
    ],
}


DATASET_DICTIONARY = [
    ("swaps_raw", "timestamp", "UTC timestamp of the block containing the swap event"),
    ("swaps_raw", "block_number", "Ethereum block number"),
    ("swaps_raw", "dex", "Source DEX venue"),
    ("swaps_raw", "pair_address", "Pool smart contract address"),
    ("swaps_raw", "transaction_hash", "Transaction hash for the event"),
    ("swaps_raw", "log_index", "Log index within the block"),
    ("swaps_raw", "amount_wbtc", "Signed WBTC reserve delta from the pool perspective"),
    ("swaps_raw", "amount_weth", "Signed WETH reserve delta from the pool perspective"),
    ("swaps_raw", "reserve_wbtc_post", "Post-event WBTC reserves in the pool"),
    ("swaps_raw", "reserve_weth_post", "Post-event WETH reserves in the pool"),
    ("pool_state_1m", "timestamp", "1-minute UTC bar timestamp"),
    ("pool_state_1m", "dex", "DEX venue for the state row"),
    ("pool_state_1m", "mid_price_eth_per_btc", "Implied pool mid price in ETH per BTC"),
    ("pool_state_1m", "reserve_wbtc", "Forward-filled WBTC reserves for the minute"),
    ("pool_state_1m", "reserve_weth", "Forward-filled WETH reserves for the minute"),
    ("pool_state_1m", "swap_count", "Number of swaps observed in the minute"),
    ("pool_state_1m", "volume_wbtc", "Absolute WBTC volume observed in the minute"),
    ("pool_state_1m", "volume_weth", "Absolute WETH volume observed in the minute"),
    ("pool_state_1m", "stale_state", "True when the last Sync event is older than the configured threshold"),
    ("arb_labels_1m", "timestamp", "1-minute UTC timestamp for the arbitrage label"),
    ("arb_labels_1m", "buy_dex", "Venue used to buy WBTC"),
    ("arb_labels_1m", "sell_dex", "Venue used to sell WBTC"),
    ("arb_labels_1m", "gross_edge_bps", "Pre-fee, pre-gas executable arbitrage edge in basis points"),
    ("arb_labels_1m", "fee_cost_bps", "Estimated venue fees in basis points"),
    ("arb_labels_1m", "gas_cost_weth", "Estimated gas cost in WETH"),
    ("arb_labels_1m", "net_edge_bps", "Post-fee, post-gas arbitrage edge in basis points"),
    ("arb_labels_1m", "opportunity_flag", "True when the best direction remains net positive and non-stale"),
]


def _config(config_path: str) -> MarketConfig:
    config = load_market_config(config_path)
    config.ensure_directories()
    return config


def _rpc_client(config: MarketConfig, rpc_url: str | None) -> RpcClient:
    return RpcClient(
        rpc_url=config.rpc_url(rpc_url),
        cache_dir=config.rpc.cache_dir,
        timeout_seconds=config.rpc.timeout_seconds,
        max_retries=config.rpc.max_retries,
        retry_backoff_seconds=config.rpc.retry_backoff_seconds,
    )


def _read_raw_inputs(config: MarketConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(config.raw_data_dir)
    logs = pd.read_parquet(raw_dir / "event_logs.parquet")
    blocks = pd.read_parquet(raw_dir / "block_headers.parquet")
    pairs = pd.read_parquet(raw_dir / "pair_metadata.parquet")
    return logs, blocks, pairs


def _write_dataset_dictionary(config: MarketConfig) -> None:
    rows = [{"dataset": dataset, "column": column, "description": description} for dataset, column, description in DATASET_DICTIONARY]
    Path(config.metadata_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(config.metadata_dir) / "dataset_dictionary.csv", index=False)


def _sanitize_events_for_parquet(events_df: pd.DataFrame) -> pd.DataFrame:
    out = events_df.copy()
    raw_columns = [column for column in out.columns if column.endswith("_raw")]
    for column in raw_columns:
        out[column] = out[column].map(lambda value: None if pd.isna(value) else str(int(value)))
    return out


def _preview_frame(df: pd.DataFrame, preferred_columns: list[str], max_rows: int = PREVIEW_ROW_LIMIT) -> pd.DataFrame:
    columns = [column for column in preferred_columns if column in df.columns]
    preview = df[columns].head(max_rows) if columns else df.head(max_rows)
    return preview


def _write_curated_table(
    curated_dir: Path,
    stem: str,
    parquet_df: pd.DataFrame,
    csv_df: pd.DataFrame | None = None,
) -> None:
    export_df = csv_df if csv_df is not None else parquet_df
    parquet_df.to_parquet(curated_dir / f"{stem}.parquet", index=False)
    export_df.to_csv(curated_dir / f"{stem}.csv", index=False)
    _preview_frame(export_df, CURATED_PREVIEW_COLUMNS.get(stem, [])).to_csv(
        curated_dir / f"{stem}_preview.csv",
        index=False,
    )


def _collect_chain_data(
    config: MarketConfig,
    rpc_url: str | None,
    start_block: int | None,
    end_block: int | None,
    chunk_size: int | None,
) -> None:
    rpc = _rpc_client(config, rpc_url)
    pairs = discover_pairs(config, rpc)
    logs_df, blocks_df, pair_frame, manifest = collect_chain_data(
        config=config,
        rpc=rpc,
        pairs=pairs,
        start_block=start_block,
        end_block=end_block,
        chunk_size=chunk_size,
    )
    write_raw_outputs(config, logs_df, blocks_df, pair_frame, manifest)


def _build_curated_dataset(config: MarketConfig) -> None:
    logs_df, blocks_df, pair_df = _read_raw_inputs(config)
    events_df = normalize_event_logs(logs_df, blocks_df, pair_df, config)
    swaps_df = build_swaps_raw(events_df, config)
    pool_state_df = build_pool_state_1m(events_df, swaps_df, stale_after_minutes=config.stale_after_minutes)
    events_to_write = _sanitize_events_for_parquet(events_df)
    curated_dir = Path(config.curated_data_dir)
    curated_dir.mkdir(parents=True, exist_ok=True)
    _write_curated_table(curated_dir, "events_curated", events_to_write)
    _write_curated_table(curated_dir, "swaps_raw", swaps_df)
    _write_curated_table(curated_dir, "pool_state_1m", pool_state_df)
    _write_dataset_dictionary(config)


def _build_arb_labels(config: MarketConfig) -> None:
    curated_dir = Path(config.curated_data_dir)
    pool_state_df = pd.read_parquet(curated_dir / "pool_state_1m.parquet")
    swaps_df = pd.read_parquet(curated_dir / "swaps_raw.parquet")
    _, blocks_df, pair_df = _read_raw_inputs(config)
    arb_df = build_arb_labels_1m(
        pool_state_df=pool_state_df,
        blocks_df=blocks_df,
        pair_metadata_df=pair_df,
        notional_wbtc=config.arbitrage_notional_wbtc,
        gas=config.gas,
    )
    reference_timestamps = arb_df["timestamp"] if not arb_df.empty else pool_state_df["timestamp"]
    assignments, boundaries = build_split_assignments(reference_timestamps, config.splits)
    pool_state_with_split = pool_state_df.merge(assignments, on="timestamp", how="left")
    arb_with_split = assign_splits(arb_df, boundaries) if not arb_df.empty else arb_df
    _write_curated_table(curated_dir, "pool_state_1m", pool_state_with_split)
    _write_curated_table(curated_dir, "arb_labels_1m", arb_with_split)
    write_split_manifest(curated_dir, assignments, boundaries)
    assignments.to_csv(curated_dir / "split_assignments.csv", index=False)
    _preview_frame(assignments, CURATED_PREVIEW_COLUMNS["split_assignments"]).to_csv(
        curated_dir / "split_assignments_preview.csv",
        index=False,
    )
    report = run_qc(swaps_raw=swaps_df, pool_state_1m=pool_state_with_split, arb_labels_1m=arb_with_split)
    write_qc_report(curated_dir, report)


def _make_background_report(config: MarketConfig) -> None:
    curated_dir = Path(config.curated_data_dir)
    report_dir = Path(config.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    pool_state = pd.read_parquet(curated_dir / "pool_state_1m.parquet")
    arb_labels = pd.read_parquet(curated_dir / "arb_labels_1m.parquet")

    pool_state["timestamp"] = pd.to_datetime(pool_state["timestamp"], utc=True)
    arb_labels["timestamp"] = pd.to_datetime(arb_labels["timestamp"], utc=True)

    price_summary = (
        pool_state.groupby("dex", as_index=False)
        .agg(
            mean_price_eth_per_btc=("mid_price_eth_per_btc", "mean"),
            std_price_eth_per_btc=("mid_price_eth_per_btc", "std"),
            min_price_eth_per_btc=("mid_price_eth_per_btc", "min"),
            max_price_eth_per_btc=("mid_price_eth_per_btc", "max"),
        )
    )
    price_summary.to_csv(report_dir / "price_summary.csv", index=False)

    swap_activity = (
        pool_state.assign(hour_utc=pool_state["timestamp"].dt.hour)
        .groupby(["dex", "hour_utc"], as_index=False)
        .agg(
            avg_swap_count=("swap_count", "mean"),
            avg_volume_wbtc=("volume_wbtc", "mean"),
            avg_volume_weth=("volume_weth", "mean"),
        )
    )
    swap_activity.to_csv(report_dir / "swap_activity_by_hour.csv", index=False)

    liquidity_depth = (
        pool_state.groupby("dex", as_index=False)
        .agg(
            mean_reserve_wbtc=("reserve_wbtc", "mean"),
            median_reserve_wbtc=("reserve_wbtc", "median"),
            mean_reserve_weth=("reserve_weth", "mean"),
            median_reserve_weth=("reserve_weth", "median"),
        )
    )
    liquidity_depth.to_csv(report_dir / "liquidity_depth_summary.csv", index=False)

    opportunity_summary = pd.DataFrame(
        {
            "gross_positive_count": [int((arb_labels["gross_edge_bps"] > 0).sum())],
            "net_positive_count": [int(arb_labels["opportunity_flag"].sum())],
            "mean_net_edge_bps": [float(arb_labels["net_edge_bps"].mean())],
            "median_net_edge_bps": [float(arb_labels["net_edge_bps"].median())],
        }
    )
    opportunity_summary.to_csv(report_dir / "opportunity_summary.csv", index=False)

    if "split" in arb_labels.columns:
        split_descriptives = (
            arb_labels.groupby("split", as_index=False)
            .agg(
                mean_net_edge_bps=("net_edge_bps", "mean"),
                median_net_edge_bps=("net_edge_bps", "median"),
                positive_opportunities=("opportunity_flag", "sum"),
                observations=("timestamp", "count"),
            )
        )
        split_descriptives.to_csv(report_dir / "split_descriptives.csv", index=False)

    price_pivot = pool_state.pivot(index="timestamp", columns="dex", values="mid_price_eth_per_btc")
    if not price_pivot.empty:
        ax = price_pivot.plot(figsize=(12, 5), title="WBTC/WETH Pool Price Series")
        ax.set_ylabel("ETH per BTC")
        ax.figure.tight_layout()
        ax.figure.savefig(report_dir / "pool_price_series.png", dpi=160)
        plt.close(ax.figure)

    if price_pivot.shape[1] >= 2:
        columns = list(price_pivot.columns)
        spread = (price_pivot[columns[0]] / price_pivot[columns[1]] - 1.0) * 10_000.0
        ax = spread.plot(figsize=(12, 5), title="Cross-DEX Spread")
        ax.set_ylabel("Basis points")
        ax.figure.tight_layout()
        ax.figure.savefig(report_dir / "cross_dex_spread.png", dpi=160)
        plt.close(ax.figure)


@app.command("collect-chain-data")
def collect_chain_data_command(
    config_path: str = typer.Option("config/markets.yaml", help="Path to the YAML market config."),
    rpc_url: str | None = typer.Option(None, help="Ethereum JSON-RPC URL. Falls back to ETH_RPC_URL."),
    start_block: int | None = typer.Option(None, help="Optional explicit start block override."),
    end_block: int | None = typer.Option(None, help="Optional explicit end block override."),
    chunk_size: int | None = typer.Option(None, help="Optional block chunk override for eth_getLogs."),
) -> None:
    config = _config(config_path)
    _collect_chain_data(config, rpc_url, start_block, end_block, chunk_size)
    typer.echo(f"Raw chain data written to {config.raw_data_dir}")


@app.command("build-curated-dataset")
def build_curated_dataset_command(
    config_path: str = typer.Option("config/markets.yaml", help="Path to the YAML market config."),
) -> None:
    config = _config(config_path)
    _build_curated_dataset(config)
    typer.echo(f"Curated datasets written to {config.curated_data_dir}")


@app.command("build-arb-labels")
def build_arb_labels_command(
    config_path: str = typer.Option("config/markets.yaml", help="Path to the YAML market config."),
) -> None:
    config = _config(config_path)
    _build_arb_labels(config)
    typer.echo(f"Arbitrage labels written to {config.curated_data_dir}")


@app.command("make-background-report")
def make_background_report_command(
    config_path: str = typer.Option("config/markets.yaml", help="Path to the YAML market config."),
) -> None:
    config = _config(config_path)
    _make_background_report(config)
    typer.echo(f"Background report outputs written to {config.report_dir}")


@app.command("run-all")
def run_all_command(
    config_path: str = typer.Option("config/markets.yaml", help="Path to the YAML market config."),
    rpc_url: str | None = typer.Option(None, help="Ethereum JSON-RPC URL. Falls back to ETH_RPC_URL."),
    start_block: int | None = typer.Option(None, help="Optional explicit start block override."),
    end_block: int | None = typer.Option(None, help="Optional explicit end block override."),
    chunk_size: int | None = typer.Option(None, help="Optional block chunk override for eth_getLogs."),
) -> None:
    config = _config(config_path)
    _collect_chain_data(config, rpc_url, start_block, end_block, chunk_size)
    _build_curated_dataset(config)
    _build_arb_labels(config)
    _make_background_report(config)
    typer.echo("Pipeline complete.")


if __name__ == "__main__":  # pragma: no cover
    app()
