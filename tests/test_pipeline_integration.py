import pandas as pd

from mev_dataset.discovery import discover_pairs
from mev_dataset.extract import collect_chain_data
from mev_dataset.features import build_arb_labels_1m, build_pool_state_1m
from mev_dataset.normalize import build_swaps_raw, normalize_event_logs
from mev_dataset.qc import run_qc
from mev_dataset.split import assign_splits, build_split_assignments


def test_pipeline_builds_curated_datasets(sample_config, sample_rpc):
    pairs = discover_pairs(sample_config, sample_rpc)
    logs_df, blocks_df, pair_df, manifest = collect_chain_data(sample_config, sample_rpc, pairs)

    assert logs_df.duplicated(["transaction_hash", "log_index"]).sum() == 0
    assert manifest["window"]["start_block"] == 100
    assert manifest["window"]["end_block"] == 101

    events_df = normalize_event_logs(logs_df, blocks_df, pair_df, sample_config)
    swaps_df = build_swaps_raw(events_df, sample_config)
    pool_state_df = build_pool_state_1m(events_df, swaps_df, stale_after_minutes=sample_config.stale_after_minutes)
    arb_df = build_arb_labels_1m(pool_state_df, blocks_df, pair_df, sample_config.arbitrage_notional_wbtc, sample_config.gas)

    assert (swaps_df[["reserve_wbtc_post", "reserve_weth_post"]] > 0).all().all()
    assert swaps_df["timestamp"].is_monotonic_increasing
    assert (pool_state_df.groupby("dex")["timestamp"].diff().dropna() == pd.Timedelta(minutes=1)).all()
    assert pool_state_df[(pool_state_df["dex"] == "uniswap_v2") & (pool_state_df["timestamp"].dt.minute == 1)].shape[0] == 1
    assert arb_df["gross_edge_bps"].notna().all()
    assert arb_df["opportunity_flag"].any()

    assignments, boundaries = build_split_assignments(pool_state_df["timestamp"], sample_config.splits)
    pool_state_df = pool_state_df.merge(assignments, on="timestamp", how="left")
    arb_df = assign_splits(arb_df, boundaries)
    report = run_qc(swaps_df, pool_state_df, arb_df)

    assert report["passed"] is True
