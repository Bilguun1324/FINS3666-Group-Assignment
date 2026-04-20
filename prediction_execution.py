from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_HORIZONS_HOURS: tuple[int, ...] = (1, 4, 24)


def _safe_venue_key(*parts: object) -> str:
    raw = "__".join(str(part or "").strip().lower() for part in parts if part)
    clean = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return clean or "venue"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame


def available_market_catalog(project_root: str | Path) -> pd.DataFrame:
    root = Path(project_root)
    rows: list[dict[str, Any]] = []
    for ohlcv_path in sorted((root / "data").glob("*/curated/pool_ohlcv_hour.parquet")):
        market_slug = ohlcv_path.parent.parent.name
        curated_dir = ohlcv_path.parent
        output_manifest = root / "outputs" / market_slug / "metadata" / "dataset_manifest.json"
        ohlcv = pd.read_parquet(ohlcv_path)
        timestamp_series = pd.to_datetime(ohlcv["timestamp"], utc=True, errors="coerce")
        rows.append(
            {
                "market_slug": market_slug,
                "hourly_rows": int(len(ohlcv)),
                "venues": int(ohlcv["pool_address"].nunique()) if "pool_address" in ohlcv.columns else int(ohlcv["dex_id"].nunique()),
                "window_start_utc": timestamp_series.min(),
                "window_end_utc": timestamp_series.max(),
                "has_pairwise_edges": (curated_dir / "pairwise_edges_hour.parquet").exists(),
                "has_swaps_raw": (curated_dir / "swaps_raw.parquet").exists(),
                "has_pool_state_1s": (curated_dir / "pool_state_1s.parquet").exists(),
                "has_manifest": output_manifest.exists(),
            }
        )
    return pd.DataFrame(rows)


def market_paths(project_root: str | Path, market_slug: str) -> dict[str, Path]:
    root = Path(project_root)
    curated_dir = root / "data" / market_slug / "curated"
    output_dir = root / "outputs" / "prediction_ready" / market_slug
    return {
        "curated_dir": curated_dir,
        "pool_ohlcv": curated_dir / "pool_ohlcv_hour.parquet",
        "pairwise_edges": curated_dir / "pairwise_edges_hour.parquet",
        "swaps_raw": curated_dir / "swaps_raw.parquet",
        "pool_state_1s": curated_dir / "pool_state_1s.parquet",
        "manifest": root / "outputs" / market_slug / "metadata" / "dataset_manifest.json",
        "output_dir": output_dir,
        "venue_panel": output_dir / "venue_panel_hour.parquet",
        "execution_summary": output_dir / "execution_summary_hour.parquet",
        "model_frame": output_dir / "model_frame_hour.parquet",
        "prediction_template": output_dir / "predictions_input_template.csv",
        "trade_candidates": output_dir / "trade_candidates_hour.csv",
        "metadata": output_dir / "metadata.json",
    }


def load_market_inputs(project_root: str | Path, market_slug: str) -> dict[str, Any]:
    paths = market_paths(project_root, market_slug)
    pool_ohlcv = _read_parquet_if_exists(paths["pool_ohlcv"])
    if pool_ohlcv.empty:
        raise FileNotFoundError(f"Missing hourly pool data for {market_slug}: {paths['pool_ohlcv']}")

    pairwise_edges = _read_parquet_if_exists(paths["pairwise_edges"])
    swaps_raw = _read_parquet_if_exists(paths["swaps_raw"])
    pool_state_1s = _read_parquet_if_exists(paths["pool_state_1s"])
    manifest = _read_json(paths["manifest"]) if paths["manifest"].exists() else {}
    return {
        "market_slug": market_slug,
        "paths": paths,
        "pool_ohlcv": pool_ohlcv,
        "pairwise_edges": pairwise_edges,
        "swaps_raw": swaps_raw,
        "pool_state_1s": pool_state_1s,
        "manifest": manifest,
    }


def _aggregate_pairwise_features(pairwise_edges: pd.DataFrame) -> pd.DataFrame:
    if pairwise_edges.empty:
        return pd.DataFrame(columns=["timestamp"])

    frame = pairwise_edges.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for column in [
        "observed_spread_bps",
        "optimistic_spread_bps",
        "conservative_spread_bps",
        "fee_adjusted_spread_bps",
        "conservative_fee_adjusted_spread_bps",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    sort_column = "conservative_fee_adjusted_spread_bps" if "conservative_fee_adjusted_spread_bps" in frame.columns else "observed_spread_bps"
    best_rows = (
        frame.sort_values(["timestamp", sort_column], ascending=[True, False])
        .groupby("timestamp", as_index=False)
        .first()
    )
    summary = (
        frame.groupby("timestamp", as_index=False)
        .agg(
            pair_count=("buy_pool_address", "count"),
            pairwise_best_observed_spread_bps=("observed_spread_bps", "max"),
            pairwise_best_optimistic_spread_bps=("optimistic_spread_bps", "max"),
            pairwise_best_conservative_spread_bps=("conservative_spread_bps", "max"),
            pairwise_best_fee_adjusted_spread_bps=("fee_adjusted_spread_bps", "max"),
            pairwise_best_conservative_fee_adjusted_spread_bps=("conservative_fee_adjusted_spread_bps", "max"),
        )
    )
    best_rows = best_rows.rename(
        columns={
            "buy_dex": "pairwise_best_buy_dex",
            "sell_dex": "pairwise_best_sell_dex",
            "buy_pool_address": "pairwise_best_buy_pool_address",
            "sell_pool_address": "pairwise_best_sell_pool_address",
        }
    )
    keep = [
        "timestamp",
        "pairwise_best_buy_dex",
        "pairwise_best_sell_dex",
        "pairwise_best_buy_pool_address",
        "pairwise_best_sell_pool_address",
    ]
    return summary.merge(best_rows.loc[:, [column for column in keep if column in best_rows.columns]], on="timestamp", how="left")


def _aggregate_swap_features(swaps_raw: pd.DataFrame) -> pd.DataFrame:
    if swaps_raw.empty:
        return pd.DataFrame(columns=["timestamp", "pair_address"])

    frame = swaps_raw.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["hour"] = frame["timestamp"].dt.floor("1h")
    frame["buy_base_swaps"] = frame["trade_direction"].eq("buy_base").astype(int)
    frame["sell_base_swaps"] = frame["trade_direction"].eq("sell_base").astype(int)
    for column in ["amount_base", "amount_quote", "volume_base", "volume_quote"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    grouped = (
        frame.groupby(["hour", "pair_address"], as_index=False)
        .agg(
            swap_events=("transaction_hash", "count"),
            swap_transactions=("transaction_hash", "nunique"),
            buy_base_swaps=("buy_base_swaps", "sum"),
            sell_base_swaps=("sell_base_swaps", "sum"),
            net_amount_base=("amount_base", "sum"),
            net_amount_quote=("amount_quote", "sum"),
            gross_volume_base=("volume_base", "sum"),
            gross_volume_quote=("volume_quote", "sum"),
        )
        .rename(columns={"hour": "timestamp"})
    )
    return grouped


def _state_summary(group: pd.DataFrame) -> pd.Series:
    prices = pd.to_numeric(group["mid_price_quote_per_base"], errors="coerce").dropna()
    positive_prices = prices[prices > 0]
    if len(positive_prices) >= 2:
        log_returns = np.diff(np.log(positive_prices.to_numpy()))
        realized_volatility_bps = float(np.nanstd(log_returns) * 10_000.0)
        realized_range_bps = float((positive_prices.max() / positive_prices.min() - 1.0) * 10_000.0)
    else:
        realized_volatility_bps = np.nan
        realized_range_bps = np.nan

    stale = group["stale_state"].astype(bool) if "stale_state" in group.columns else pd.Series(dtype=bool)
    return pd.Series(
        {
            "state_points": int(len(group)),
            "non_stale_points": int((~stale).sum()) if not stale.empty else int(len(group)),
            "stale_share": float(stale.mean()) if not stale.empty else 0.0,
            "realized_volatility_bps": realized_volatility_bps,
            "realized_range_bps": realized_range_bps,
            "last_mid_price_1s": float(prices.iloc[-1]) if not prices.empty else np.nan,
        }
    )


def _aggregate_pool_state_features(pool_state_1s: pd.DataFrame) -> pd.DataFrame:
    if pool_state_1s.empty:
        return pd.DataFrame(columns=["timestamp", "pair_address"])

    frame = pool_state_1s.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["hour"] = frame["timestamp"].dt.floor("1h")
    summary = (
        frame.groupby(["hour", "pair_address"], as_index=False)
        .apply(_state_summary, include_groups=False)
        .reset_index()
    )
    if "level_2" in summary.columns:
        summary = summary.drop(columns=["level_2"])
    summary = summary.rename(columns={"hour": "timestamp"})
    return summary


def build_venue_panel(inputs: dict[str, Any]) -> pd.DataFrame:
    pool_ohlcv = inputs["pool_ohlcv"].copy()
    pool_ohlcv["timestamp"] = pd.to_datetime(pool_ohlcv["timestamp"], utc=True, errors="coerce")
    pool_ohlcv = pool_ohlcv.sort_values(["pool_address", "timestamp"]).reset_index(drop=True)
    pool_ohlcv["market_slug"] = inputs["market_slug"]
    pool_ohlcv["pair_address"] = pool_ohlcv["pool_address"].astype(str).str.lower()
    pool_ohlcv["venue_key"] = pool_ohlcv.apply(
        lambda row: _safe_venue_key(row.get("dex_id"), row.get("pool_name"), row.get("pool_address")),
        axis=1,
    )
    pool_ohlcv["venue_label"] = pool_ohlcv.apply(
        lambda row: f"{row.get('dex_id', 'unknown')} | {row.get('pool_name', row.get('pool_address', 'pool'))}",
        axis=1,
    )

    pool_ohlcv["pool_fee_bps"] = pd.to_numeric(pool_ohlcv.get("explicit_pool_fee_bps"), errors="coerce")
    if "fee_bps_from_name" in pool_ohlcv.columns:
        pool_ohlcv["pool_fee_bps"] = pool_ohlcv["pool_fee_bps"].fillna(pd.to_numeric(pool_ohlcv["fee_bps_from_name"], errors="coerce"))
    pool_ohlcv["fee_known"] = pool_ohlcv["pool_fee_bps"].notna()
    pool_ohlcv["pool_fee_rate"] = pool_ohlcv["pool_fee_bps"].fillna(0.0) / 10_000.0

    for column in ["close_usd", "open_usd", "high_usd", "low_usd", "volume_usd", "reserve_in_usd"]:
        if column in pool_ohlcv.columns:
            pool_ohlcv[column] = pd.to_numeric(pool_ohlcv[column], errors="coerce")

    pool_ohlcv["effective_buy_price_usd"] = pool_ohlcv["close_usd"] * (1.0 + pool_ohlcv["pool_fee_rate"])
    pool_ohlcv["effective_sell_price_usd"] = pool_ohlcv["close_usd"] * (1.0 - pool_ohlcv["pool_fee_rate"])

    grouped = pool_ohlcv.groupby("pool_address", group_keys=False)
    pool_ohlcv["return_1h_bps"] = grouped["close_usd"].pct_change(1) * 10_000.0
    pool_ohlcv["return_4h_bps"] = grouped["close_usd"].pct_change(4) * 10_000.0
    pool_ohlcv["return_24h_bps"] = grouped["close_usd"].pct_change(24) * 10_000.0
    pool_ohlcv["rolling_volatility_24h_bps"] = (
        pool_ohlcv.groupby("pool_address")["return_1h_bps"]
        .rolling(24, min_periods=4)
        .std()
        .reset_index(level=0, drop=True)
    )

    timestamp_summary = (
        pool_ohlcv.groupby("timestamp", as_index=False)
        .agg(
            venue_count=("pool_address", "nunique"),
            total_volume_usd=("volume_usd", "sum"),
            total_reserve_usd=("reserve_in_usd", "sum"),
            reference_price_usd=("close_usd", "median"),
            reference_price_mean_usd=("close_usd", "mean"),
            price_min_usd=("close_usd", "min"),
            price_max_usd=("close_usd", "max"),
        )
    )
    timestamp_summary["cross_exchange_dispersion_bps"] = np.where(
        timestamp_summary["reference_price_usd"] > 0,
        (timestamp_summary["price_max_usd"] / timestamp_summary["price_min_usd"] - 1.0) * 10_000.0,
        np.nan,
    )
    pool_ohlcv = pool_ohlcv.merge(timestamp_summary, on="timestamp", how="left")
    pool_ohlcv["price_vs_reference_bps"] = np.where(
        pool_ohlcv["reference_price_usd"] > 0,
        (pool_ohlcv["close_usd"] / pool_ohlcv["reference_price_usd"] - 1.0) * 10_000.0,
        np.nan,
    )
    pool_ohlcv["volume_share"] = np.where(pool_ohlcv["total_volume_usd"] > 0, pool_ohlcv["volume_usd"] / pool_ohlcv["total_volume_usd"], np.nan)
    pool_ohlcv["reserve_share"] = np.where(pool_ohlcv["total_reserve_usd"] > 0, pool_ohlcv["reserve_in_usd"] / pool_ohlcv["total_reserve_usd"], np.nan)
    pool_ohlcv["buy_rank"] = pool_ohlcv.groupby("timestamp")["effective_buy_price_usd"].rank(method="first", ascending=True)
    pool_ohlcv["sell_rank"] = pool_ohlcv.groupby("timestamp")["effective_sell_price_usd"].rank(method="first", ascending=False)

    swap_features = _aggregate_swap_features(inputs["swaps_raw"])
    state_features = _aggregate_pool_state_features(inputs["pool_state_1s"])
    venue_panel = pool_ohlcv.merge(swap_features, on=["timestamp", "pair_address"], how="left")
    venue_panel = venue_panel.merge(state_features, on=["timestamp", "pair_address"], how="left")
    return venue_panel.sort_values(["timestamp", "buy_rank", "sell_rank", "venue_label"]).reset_index(drop=True)


def build_execution_summary(venue_panel: pd.DataFrame, pairwise_edges: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for timestamp, group in venue_panel.groupby("timestamp", sort=True):
        buy_sorted = group.sort_values(["effective_buy_price_usd", "close_usd"], ascending=[True, True]).reset_index(drop=True)
        sell_sorted = group.sort_values(["effective_sell_price_usd", "close_usd"], ascending=[False, False]).reset_index(drop=True)
        best_buy = buy_sorted.iloc[0]
        second_buy = buy_sorted.iloc[1] if len(buy_sorted) > 1 else best_buy
        best_sell = sell_sorted.iloc[0]
        second_sell = sell_sorted.iloc[1] if len(sell_sorted) > 1 else best_sell

        best_buy_gap_bps = (
            (second_buy["effective_buy_price_usd"] / best_buy["effective_buy_price_usd"] - 1.0) * 10_000.0
            if best_buy["effective_buy_price_usd"] > 0 and len(buy_sorted) > 1
            else np.nan
        )
        best_sell_gap_bps = (
            (best_sell["effective_sell_price_usd"] / second_sell["effective_sell_price_usd"] - 1.0) * 10_000.0
            if second_sell["effective_sell_price_usd"] > 0 and len(sell_sorted) > 1
            else np.nan
        )
        rows.append(
            {
                "timestamp": timestamp,
                "market_slug": best_buy["market_slug"],
                "reference_price_usd": best_buy["reference_price_usd"],
                "reference_price_mean_usd": best_buy["reference_price_mean_usd"],
                "venue_count": int(best_buy["venue_count"]),
                "total_volume_usd": best_buy["total_volume_usd"],
                "total_reserve_usd": best_buy["total_reserve_usd"],
                "cross_exchange_dispersion_bps": best_buy["cross_exchange_dispersion_bps"],
                "best_buy_venue": best_buy["venue_label"],
                "best_buy_venue_key": best_buy["venue_key"],
                "best_buy_pool_address": best_buy["pair_address"],
                "best_buy_price_usd": best_buy["close_usd"],
                "best_buy_effective_price_usd": best_buy["effective_buy_price_usd"],
                "best_buy_fee_bps": best_buy["pool_fee_bps"],
                "best_buy_gap_vs_second_bps": best_buy_gap_bps,
                "best_sell_venue": best_sell["venue_label"],
                "best_sell_venue_key": best_sell["venue_key"],
                "best_sell_pool_address": best_sell["pair_address"],
                "best_sell_price_usd": best_sell["close_usd"],
                "best_sell_effective_price_usd": best_sell["effective_sell_price_usd"],
                "best_sell_fee_bps": best_sell["pool_fee_bps"],
                "best_sell_gap_vs_second_bps": best_sell_gap_bps,
                "best_cross_exchange_edge_bps": (
                    (best_sell["effective_sell_price_usd"] / best_buy["effective_buy_price_usd"] - 1.0) * 10_000.0
                    if best_buy["effective_buy_price_usd"] > 0
                    else np.nan
                ),
                "best_buy_swap_events": best_buy.get("swap_events", np.nan),
                "best_sell_swap_events": best_sell.get("swap_events", np.nan),
                "best_buy_stale_share": best_buy.get("stale_share", np.nan),
                "best_sell_stale_share": best_sell.get("stale_share", np.nan),
            }
        )

    summary = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    if pairwise_edges is not None and not pairwise_edges.empty:
        summary = summary.merge(_aggregate_pairwise_features(pairwise_edges), on="timestamp", how="left")
    return summary


def build_model_frame(
    venue_panel: pd.DataFrame,
    execution_summary: pd.DataFrame,
    *,
    horizons_hours: tuple[int, ...] = DEFAULT_HORIZONS_HOURS,
) -> pd.DataFrame:
    pivot_features = [
        "close_usd",
        "effective_buy_price_usd",
        "effective_sell_price_usd",
        "volume_usd",
        "reserve_in_usd",
        "pool_fee_bps",
        "price_vs_reference_bps",
        "volume_share",
        "reserve_share",
        "return_1h_bps",
        "return_4h_bps",
        "return_24h_bps",
        "rolling_volatility_24h_bps",
        "swap_events",
        "swap_transactions",
        "buy_base_swaps",
        "sell_base_swaps",
        "net_amount_base",
        "net_amount_quote",
        "gross_volume_base",
        "gross_volume_quote",
        "state_points",
        "non_stale_points",
        "stale_share",
        "realized_volatility_bps",
        "realized_range_bps",
        "last_mid_price_1s",
    ]
    available_features = [column for column in pivot_features if column in venue_panel.columns]
    wide = (
        venue_panel.pivot(index="timestamp", columns="venue_key", values=available_features)
        .sort_index(axis=1)
    )
    wide.columns = [f"{feature}__{venue}" for feature, venue in wide.columns]
    wide = wide.reset_index()

    model_frame = execution_summary.merge(wide, on="timestamp", how="left").sort_values("timestamp").reset_index(drop=True)
    model_frame["reference_return_1h_bps"] = model_frame["reference_price_usd"].pct_change(1) * 10_000.0
    model_frame["reference_return_4h_bps"] = model_frame["reference_price_usd"].pct_change(4) * 10_000.0
    model_frame["reference_return_24h_bps"] = model_frame["reference_price_usd"].pct_change(24) * 10_000.0
    model_frame["reference_volatility_24h_bps"] = model_frame["reference_return_1h_bps"].rolling(24, min_periods=4).std()
    model_frame["best_buy_price_change_1h_bps"] = model_frame["best_buy_effective_price_usd"].pct_change(1) * 10_000.0
    model_frame["best_sell_price_change_1h_bps"] = model_frame["best_sell_effective_price_usd"].pct_change(1) * 10_000.0
    model_frame["history_ready_24h"] = model_frame["reference_return_24h_bps"].notna()

    for horizon in horizons_hours:
        future_reference = model_frame["reference_price_usd"].shift(-horizon)
        future_best_buy = model_frame["best_buy_effective_price_usd"].shift(-horizon)
        future_best_sell = model_frame["best_sell_effective_price_usd"].shift(-horizon)
        model_frame[f"target_future_reference_price_{horizon}h"] = future_reference
        model_frame[f"target_future_return_{horizon}h_bps"] = np.where(
            model_frame["reference_price_usd"] > 0,
            (future_reference / model_frame["reference_price_usd"] - 1.0) * 10_000.0,
            np.nan,
        )
        model_frame[f"target_future_best_buy_price_{horizon}h"] = future_best_buy
        model_frame[f"target_future_best_sell_price_{horizon}h"] = future_best_sell

    return model_frame


def build_prediction_template(
    model_frame: pd.DataFrame,
    *,
    horizons_hours: tuple[int, ...] = DEFAULT_HORIZONS_HOURS,
) -> pd.DataFrame:
    template = model_frame.loc[:, ["timestamp", "market_slug", "reference_price_usd"]].copy()
    for horizon in horizons_hours:
        template[f"predicted_return_{horizon}h_bps"] = np.nan
        template[f"predicted_price_{horizon}h"] = np.nan
    template["prediction_source"] = ""
    template["prediction_notes"] = ""
    return template


def prepare_prediction_ready_bundle(
    project_root: str | Path,
    market_slug: str,
    *,
    horizons_hours: tuple[int, ...] = DEFAULT_HORIZONS_HOURS,
) -> dict[str, Any]:
    inputs = load_market_inputs(project_root, market_slug)
    venue_panel = build_venue_panel(inputs)
    execution_summary = build_execution_summary(venue_panel, inputs["pairwise_edges"])
    model_frame = build_model_frame(venue_panel, execution_summary, horizons_hours=horizons_hours)
    prediction_template = build_prediction_template(model_frame, horizons_hours=horizons_hours)
    return {
        **inputs,
        "venue_panel": venue_panel,
        "execution_summary": execution_summary,
        "model_frame": model_frame,
        "prediction_template": prediction_template,
    }


def export_prediction_ready_bundle(bundle: dict[str, Any], *, horizons_hours: tuple[int, ...] = DEFAULT_HORIZONS_HOURS) -> dict[str, Path]:
    paths = bundle["paths"]
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    bundle["venue_panel"].to_parquet(paths["venue_panel"], index=False)
    bundle["execution_summary"].to_parquet(paths["execution_summary"], index=False)
    bundle["model_frame"].to_parquet(paths["model_frame"], index=False)
    bundle["prediction_template"].to_csv(paths["prediction_template"], index=False)

    metadata = {
        "market_slug": bundle["market_slug"],
        "rows": {
            "venue_panel": int(len(bundle["venue_panel"])),
            "execution_summary": int(len(bundle["execution_summary"])),
            "model_frame": int(len(bundle["model_frame"])),
        },
        "columns": {
            "venue_panel": bundle["venue_panel"].columns.tolist(),
            "execution_summary": bundle["execution_summary"].columns.tolist(),
            "model_frame": bundle["model_frame"].columns.tolist(),
        },
        "horizons_hours": list(horizons_hours),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return {
        "venue_panel": paths["venue_panel"],
        "execution_summary": paths["execution_summary"],
        "model_frame": paths["model_frame"],
        "prediction_template": paths["prediction_template"],
        "metadata": paths["metadata"],
    }


def _prediction_fair_value(frame: pd.DataFrame) -> pd.Series:
    if "predicted_fair_value" in frame.columns and frame["predicted_fair_value"].notna().any():
        return pd.to_numeric(frame["predicted_fair_value"], errors="coerce")
    if "predicted_price_1h" in frame.columns and frame["predicted_price_1h"].notna().any():
        return pd.to_numeric(frame["predicted_price_1h"], errors="coerce")
    if "predicted_return_1h_bps" in frame.columns and frame["predicted_return_1h_bps"].notna().any():
        predicted_return = pd.to_numeric(frame["predicted_return_1h_bps"], errors="coerce") / 10_000.0
        return frame["reference_price_usd"] * (1.0 + predicted_return)
    if "predicted_return_1h" in frame.columns and frame["predicted_return_1h"].notna().any():
        predicted_return = pd.to_numeric(frame["predicted_return_1h"], errors="coerce")
        return frame["reference_price_usd"] * (1.0 + predicted_return)
    return pd.Series(np.nan, index=frame.index, dtype=float)


def merge_external_predictions(model_frame: pd.DataFrame, prediction_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(prediction_path)
    if not path.exists():
        merged = model_frame.copy()
        merged["predicted_fair_value"] = np.nan
        return merged, {"status": "missing", "path": str(path)}

    predictions = pd.read_csv(path)
    if "timestamp" not in predictions.columns:
        raise ValueError(f"Prediction file must include a timestamp column: {path}")
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True, errors="coerce")

    merge_columns = [column for column in predictions.columns if column != "market_slug"]
    merged = model_frame.merge(predictions.loc[:, merge_columns], on="timestamp", how="left")
    merged["predicted_fair_value"] = _prediction_fair_value(merged)
    coverage = float(merged["predicted_fair_value"].notna().mean()) if len(merged) else 0.0
    return merged, {"status": "loaded", "path": str(path), "coverage": coverage, "prediction_columns": predictions.columns.tolist()}


def build_trade_candidates(
    prediction_frame: pd.DataFrame,
    *,
    min_expected_edge_bps: float = 5.0,
) -> pd.DataFrame:
    frame = prediction_frame.copy()
    frame["predicted_fair_value"] = _prediction_fair_value(frame)
    frame["expected_buy_edge_bps"] = np.where(
        frame["best_buy_effective_price_usd"] > 0,
        (frame["predicted_fair_value"] / frame["best_buy_effective_price_usd"] - 1.0) * 10_000.0,
        np.nan,
    )
    frame["expected_sell_edge_bps"] = np.where(
        frame["predicted_fair_value"] > 0,
        (frame["best_sell_effective_price_usd"] / frame["predicted_fair_value"] - 1.0) * 10_000.0,
        np.nan,
    )
    frame["expected_edge_bps"] = frame[["expected_buy_edge_bps", "expected_sell_edge_bps"]].max(axis=1)
    frame["trade_side"] = np.select(
        [
            frame["expected_buy_edge_bps"] >= frame["expected_sell_edge_bps"],
            frame["expected_sell_edge_bps"] > frame["expected_buy_edge_bps"],
        ],
        [
            "buy",
            "sell",
        ],
        default="hold",
    )
    frame.loc[frame["expected_edge_bps"].lt(min_expected_edge_bps) | frame["predicted_fair_value"].isna(), "trade_side"] = "hold"
    frame["execution_venue"] = np.where(
        frame["trade_side"].eq("buy"),
        frame["best_buy_venue"],
        np.where(frame["trade_side"].eq("sell"), frame["best_sell_venue"], None),
    )
    frame["execution_price_usd"] = np.where(
        frame["trade_side"].eq("buy"),
        frame["best_buy_effective_price_usd"],
        np.where(frame["trade_side"].eq("sell"), frame["best_sell_effective_price_usd"], np.nan),
    )
    frame["confidence_proxy"] = np.where(
        frame["cross_exchange_dispersion_bps"].notna(),
        frame["expected_edge_bps"] / frame["cross_exchange_dispersion_bps"].replace(0, np.nan),
        np.nan,
    )
    keep_columns = [
        "timestamp",
        "market_slug",
        "reference_price_usd",
        "predicted_fair_value",
        "trade_side",
        "execution_venue",
        "execution_price_usd",
        "expected_buy_edge_bps",
        "expected_sell_edge_bps",
        "expected_edge_bps",
        "best_buy_venue",
        "best_buy_effective_price_usd",
        "best_sell_venue",
        "best_sell_effective_price_usd",
        "cross_exchange_dispersion_bps",
        "best_buy_gap_vs_second_bps",
        "best_sell_gap_vs_second_bps",
        "confidence_proxy",
    ]
    existing = [column for column in keep_columns if column in frame.columns]
    candidates = frame.loc[:, existing].copy()
    return candidates.sort_values(["expected_edge_bps", "timestamp"], ascending=[False, True]).reset_index(drop=True)
