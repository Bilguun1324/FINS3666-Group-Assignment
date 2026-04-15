from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests


GECKOTERMINAL_API_BASE = "https://api.geckoterminal.com/api/v2"


MARKET_DATA_SPECS: dict[str, dict[str, Any]] = {
    "arbitrum_wbtc_weth": {
        "network": "arbitrum",
        "search_query": "WBTC WETH",
        "target_asset": "WBTC",
        "pool_count": 3,
    },
    "optimism_wbtc_weth": {
        "network": "optimism",
        "search_query": "WBTC WETH",
        "target_asset": "WBTC",
        "pool_count": 3,
    },
    "base_wbtc_weth": {
        "network": "base",
        "search_query": "WBTC WETH",
        "target_asset": "WBTC",
        "pool_count": 3,
    },
    "mainnet_usdc_usdt": {
        "network": "eth",
        "search_query": "USDC USDT",
        "target_asset": "USDC",
        "pool_count": 3,
    },
    "mainnet_dai_usdc": {
        "network": "eth",
        "search_query": "DAI USDC",
        "target_asset": "DAI",
        "pool_count": 3,
    },
    "mainnet_weth_usdc": {
        "network": "eth",
        "search_query": "WETH USDC",
        "target_asset": "WETH",
        "pool_count": 3,
    },
    "mainnet_wbtc_usdc": {
        "network": "eth",
        "search_query": "WBTC USDC",
        "target_asset": "WBTC",
        "pool_count": 3,
    },
    "solana_research": {
        "network": "solana",
        "search_query": "SOL USDC",
        "target_asset": "SOL",
        "pool_count": 3,
    },
    "cosmos_research": {
        # GeckoTerminal does not expose Osmosis as a network. Kava is used here
        # as a Cosmos-ecosystem proxy with listed ATOM/stablecoin pools.
        "network": "kava",
        "search_query": "ATOM USD",
        "target_asset": "ATOM",
        "pool_count": 3,
    },
}


@dataclass(frozen=True)
class FetchSettings:
    timeframe: str = "hour"
    aggregate: int = 1
    limit: int = 720
    min_pool_ohlcv_rows: int = 24
    request_sleep_seconds: float = float(os.getenv("GECKOTERMINAL_REQUEST_SLEEP_SECONDS", "2.6"))
    max_retries: int = 4


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _api_get(
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    session: requests.Session | None = None,
    settings: FetchSettings = FetchSettings(),
) -> dict[str, Any]:
    client = session or requests.Session()
    url = f"{GECKOTERMINAL_API_BASE}{endpoint}"

    for attempt in range(settings.max_retries):
        response = client.get(url, params=params, timeout=30, headers={"accept": "application/json"})
        if response.status_code == 200:
            time.sleep(settings.request_sleep_seconds)
            return response.json()

        if response.status_code == 429 and attempt < settings.max_retries - 1:
            retry_after = response.headers.get("retry-after")
            wait_seconds = float(retry_after) if retry_after else settings.request_sleep_seconds * (attempt + 3)
            time.sleep(wait_seconds)
            continue

        message = response.text[:500].replace("\n", " ")
        raise RuntimeError(f"GeckoTerminal request failed: {response.status_code} {url} {message}")

    raise RuntimeError(f"GeckoTerminal request failed after retries: {url}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _token_address_from_relationship(token_id: str | None) -> str | None:
    if not token_id:
        return None
    return token_id.split("_", 1)[-1]


def _pool_fee_bps(pool_name: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)%", pool_name)
    if not match:
        return None
    return float(match.group(1)) * 100.0


def _pool_asset_order(pool_name: str) -> list[str]:
    name_without_fee = re.sub(r"\s+\d+(?:\.\d+)?%.*$", "", pool_name)
    return [part.strip().upper() for part in name_without_fee.split("/")[:2]]


def _pool_token_for_asset(pool_record: dict[str, Any], target_asset: str) -> str | None:
    assets = _pool_asset_order(pool_record["pool_name"])
    target = target_asset.upper()
    if len(assets) >= 1 and assets[0] == target:
        return pool_record["base_token_address"]
    if len(assets) >= 2 and assets[1] == target:
        return pool_record["quote_token_address"]
    return pool_record["base_token_address"]


def _normalise_search_results(market_slug: str, payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in payload.get("data", []):
        attrs = item.get("attributes", {})
        rel = item.get("relationships", {})
        pool_assets = _pool_asset_order(attrs.get("name") or "")
        base_token_id = (rel.get("base_token", {}).get("data") or {}).get("id")
        quote_token_id = (rel.get("quote_token", {}).get("data") or {}).get("id")
        dex_id = (rel.get("dex", {}).get("data") or {}).get("id")
        rows.append(
            {
                "market_slug": market_slug,
                "pool_id": item.get("id"),
                "pool_address": attrs.get("address"),
                "pool_name": attrs.get("name"),
                "pool_asset_0": pool_assets[0] if len(pool_assets) > 0 else None,
                "pool_asset_1": pool_assets[1] if len(pool_assets) > 1 else None,
                "dex_id": dex_id,
                "base_token_id": base_token_id,
                "quote_token_id": quote_token_id,
                "base_token_address": _token_address_from_relationship(base_token_id),
                "quote_token_address": _token_address_from_relationship(quote_token_id),
                "pool_created_at": attrs.get("pool_created_at"),
                "reserve_in_usd": _safe_float(attrs.get("reserve_in_usd")),
                "volume_h24_usd": _safe_float((attrs.get("volume_usd") or {}).get("h24")),
                "transactions_h24": sum((attrs.get("transactions") or {}).get("h24", {}).get(k, 0) for k in ["buys", "sells"]),
                "fee_bps_from_name": _pool_fee_bps(attrs.get("name") or ""),
            }
        )
    return pd.DataFrame(rows)


def _select_pools(pools: pd.DataFrame, pool_count: int) -> pd.DataFrame:
    if pools.empty:
        return pools

    ranked = pools.copy()
    ranked["reserve_in_usd"] = ranked["reserve_in_usd"].fillna(0.0)
    ranked["volume_h24_usd"] = ranked["volume_h24_usd"].fillna(0.0)
    ranked["selection_score"] = ranked["reserve_in_usd"] + ranked["volume_h24_usd"]
    ranked = ranked.sort_values(["selection_score", "reserve_in_usd"], ascending=False)

    selected_rows = []
    seen_dexes: set[str] = set()
    for row in ranked.to_dict("records"):
        dex_id = row.get("dex_id") or row.get("pool_address")
        if dex_id in seen_dexes:
            continue
        selected_rows.append(row)
        seen_dexes.add(dex_id)
        if len(selected_rows) >= pool_count:
            break

    if len(selected_rows) < min(pool_count, len(ranked)):
        selected_addresses = {row["pool_address"] for row in selected_rows}
        for row in ranked.to_dict("records"):
            if row["pool_address"] in selected_addresses:
                continue
            selected_rows.append(row)
            if len(selected_rows) >= pool_count:
                break

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _fetch_or_load_pool_search(
    market_slug: str,
    spec: dict[str, Any],
    raw_dir: Path,
    *,
    refresh: bool,
    session: requests.Session,
    settings: FetchSettings,
) -> pd.DataFrame:
    search_path = raw_dir / "pool_search.json"
    if search_path.exists() and not refresh:
        payload = _read_json(search_path)
    else:
        payload = _api_get(
            "/search/pools",
            params={"network": spec["network"], "query": spec["search_query"]},
            session=session,
            settings=settings,
        )
        _write_json(search_path, payload)

    pools = _normalise_search_results(market_slug, payload)
    if pools.empty:
        raise RuntimeError(f"No GeckoTerminal pools found for {market_slug}: {spec['search_query']} on {spec['network']}")
    return pools


def _fetch_or_load_ohlcv(
    pool_record: dict[str, Any],
    target_asset: str,
    raw_dir: Path,
    *,
    refresh: bool,
    session: requests.Session,
    settings: FetchSettings,
) -> pd.DataFrame:
    network = pool_record["network"]
    pool_address = pool_record["pool_address"]
    safe_pool_address = re.sub(r"[^a-zA-Z0-9]+", "_", str(pool_address)).strip("_")
    ohlcv_path = raw_dir / f"ohlcv_{settings.timeframe}_{safe_pool_address}.json"

    if ohlcv_path.exists() and not refresh:
        payload = _read_json(ohlcv_path)
    else:
        token_address = _pool_token_for_asset(pool_record, target_asset)
        params = {
            "aggregate": settings.aggregate,
            "limit": settings.limit,
            "currency": "usd",
            "include_empty_intervals": "true",
        }
        if token_address:
            params["token"] = token_address

        payload = _api_get(
            f"/networks/{network}/pools/{pool_address}/ohlcv/{settings.timeframe}",
            params=params,
            session=session,
            settings=settings,
        )
        _write_json(ohlcv_path, payload)

    ohlcv_list = payload.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
    rows = []
    for timestamp, open_, high, low, close, volume in ohlcv_list:
        rows.append(
            {
                "timestamp": pd.to_datetime(timestamp, unit="s", utc=True),
                "open_usd": float(open_),
                "high_usd": float(high),
                "low_usd": float(low),
                "close_usd": float(close),
                "volume_usd": float(volume),
                "market_slug": pool_record["market_slug"],
                "network": network,
                "pool_address": pool_address,
                "pool_name": pool_record["pool_name"],
                "dex_id": pool_record["dex_id"],
                "reserve_in_usd": pool_record.get("reserve_in_usd"),
                "volume_h24_usd": pool_record.get("volume_h24_usd"),
                "fee_bps_from_name": pool_record.get("fee_bps_from_name"),
            }
        )

    return pd.DataFrame(rows).sort_values(["pool_address", "timestamp"]).reset_index(drop=True)


def build_market_edges(pool_ohlcv: pd.DataFrame) -> pd.DataFrame:
    edge_columns = [
        "timestamp",
        "pool_count",
        "buy_pool_address",
        "sell_pool_address",
        "buy_pool_name",
        "sell_pool_name",
        "buy_dex",
        "sell_dex",
        "buy_price_usd",
        "sell_price_usd",
        "observed_spread_bps",
        "total_volume_usd",
    ]
    if pool_ohlcv.empty:
        return pd.DataFrame(columns=edge_columns)

    prices = pool_ohlcv.pivot_table(index="timestamp", columns="pool_address", values="close_usd", aggfunc="last")
    volumes = pool_ohlcv.pivot_table(index="timestamp", columns="pool_address", values="volume_usd", aggfunc="sum")
    pool_names = pool_ohlcv.drop_duplicates("pool_address").set_index("pool_address")["pool_name"].to_dict()
    dex_ids = pool_ohlcv.drop_duplicates("pool_address").set_index("pool_address")["dex_id"].to_dict()

    rows = []
    for timestamp, price_row in prices.iterrows():
        active = price_row.dropna()
        if len(active) < 2:
            continue
        buy_pool = active.idxmin()
        sell_pool = active.idxmax()
        buy_price = float(active.loc[buy_pool])
        sell_price = float(active.loc[sell_pool])
        spread_bps = ((sell_price / buy_price) - 1.0) * 10_000.0 if buy_price > 0 else None
        rows.append(
            {
                "timestamp": timestamp,
                "pool_count": int(len(active)),
                "buy_pool_address": buy_pool,
                "sell_pool_address": sell_pool,
                "buy_pool_name": pool_names.get(buy_pool),
                "sell_pool_name": pool_names.get(sell_pool),
                "buy_dex": dex_ids.get(buy_pool),
                "sell_dex": dex_ids.get(sell_pool),
                "buy_price_usd": buy_price,
                "sell_price_usd": sell_price,
                "observed_spread_bps": spread_bps,
                "total_volume_usd": float(volumes.loc[timestamp, active.index].sum()),
            }
        )

    if not rows:
        return pd.DataFrame(columns=edge_columns)

    return pd.DataFrame(rows, columns=edge_columns).sort_values("timestamp").reset_index(drop=True)


def build_real_size_sensitivity(
    market_edges: pd.DataFrame,
    assumptions: dict[str, Any],
    *,
    size_multipliers: list[float] | None = None,
) -> pd.DataFrame:
    if market_edges.empty:
        return pd.DataFrame()

    multipliers = size_multipliers or [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    base_notional_usd = float(assumptions["trade_notional_usd"])
    slippage_buffer_bps = float(assumptions["slippage_buffer_bps"])
    infra_buffer_bps = float(assumptions.get("infra_buffer_bps", 0.0))
    fee_tier_bps = float(min(assumptions.get("fee_tiers_bps", [0.0])))
    base_gas_cost_usd = float(assumptions.get("gas_cost_usd_scenarios", {}).get("base", 0.0))

    rows = []
    for size_multiple in multipliers:
        asset_notional_usd = base_notional_usd * size_multiple
        size_adjusted_slippage_bps = slippage_buffer_bps * (size_multiple**0.5)
        gas_bps = (base_gas_cost_usd / asset_notional_usd) * 10_000.0 if asset_notional_usd else 0.0

        gross_edge = market_edges["observed_spread_bps"] - size_adjusted_slippage_bps
        net_edge = gross_edge - (fee_tier_bps * 2.0) - gas_bps - infra_buffer_bps

        rows.append(
            {
                "asset_notional_usd": asset_notional_usd,
                "asset_size_multiple": size_multiple,
                "observed_intervals": int(market_edges["observed_spread_bps"].notna().sum()),
                "size_adjusted_slippage_bps": size_adjusted_slippage_bps,
                "modeled_fee_tier_bps_per_side": fee_tier_bps,
                "modeled_base_gas_cost_usd": base_gas_cost_usd,
                "modeled_gas_bps": gas_bps,
                "gross_positive_intervals": int((gross_edge > 0).sum()),
                "gross_positive_share_pct": float((gross_edge > 0).mean() * 100.0),
                "net_positive_intervals": int((net_edge > 0).sum()),
                "net_positive_share_pct": float((net_edge > 0).mean() * 100.0),
                "mean_gross_edge_bps": float(gross_edge.mean()),
                "median_gross_edge_bps": float(gross_edge.median()),
                "max_gross_edge_bps": float(gross_edge.max()),
                "mean_net_edge_bps": float(net_edge.mean()),
                "median_net_edge_bps": float(net_edge.median()),
                "max_net_edge_bps": float(net_edge.max()),
            }
        )

    return pd.DataFrame(rows)


def write_dataset_outputs(
    *,
    selected_pools: pd.DataFrame,
    pool_ohlcv: pd.DataFrame,
    market_edges: pd.DataFrame,
    size_sensitivity: pd.DataFrame,
    raw_dir: Path,
    curated_dir: Path,
    report_dir: Path,
    metadata_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    curated_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "selected_pools_csv": raw_dir / "selected_pools.csv",
        "pool_ohlcv_parquet": curated_dir / "pool_ohlcv_hour.parquet",
        "pool_ohlcv_preview_csv": curated_dir / "pool_ohlcv_hour_preview.csv",
        "market_edges_parquet": curated_dir / "market_edges_hour.parquet",
        "market_edges_preview_csv": curated_dir / "market_edges_hour_preview.csv",
        "size_sensitivity_csv": report_dir / "real_size_sensitivity_hour.csv",
        "manifest_json": metadata_dir / "dataset_manifest.json",
    }

    selected_pools.to_csv(paths["selected_pools_csv"], index=False)
    pool_ohlcv.to_parquet(paths["pool_ohlcv_parquet"], index=False)
    pool_ohlcv.head(200).to_csv(paths["pool_ohlcv_preview_csv"], index=False)
    market_edges.to_parquet(paths["market_edges_parquet"], index=False)
    market_edges.head(200).to_csv(paths["market_edges_preview_csv"], index=False)
    size_sensitivity.to_csv(paths["size_sensitivity_csv"], index=False)
    _write_json(paths["manifest_json"], manifest)
    return paths


def load_or_fetch_market_dataset(
    market_slug: str,
    assumptions: dict[str, Any],
    *,
    project_root: str | Path = ".",
    refresh: bool = False,
    settings: FetchSettings | None = None,
) -> dict[str, Any]:
    if market_slug not in MARKET_DATA_SPECS:
        raise KeyError(f"No market data spec exists for {market_slug!r}")

    fetch_settings = settings or FetchSettings()
    spec = MARKET_DATA_SPECS[market_slug]
    project_root = Path(project_root)
    raw_dir = project_root / "data" / market_slug / "raw"
    curated_dir = project_root / "data" / market_slug / "curated"
    report_dir = project_root / "outputs" / market_slug / "report"
    metadata_dir = project_root / "outputs" / market_slug / "metadata"

    cached_paths = {
        "selected_pools_csv": raw_dir / "selected_pools.csv",
        "pool_ohlcv_parquet": curated_dir / "pool_ohlcv_hour.parquet",
        "market_edges_parquet": curated_dir / "market_edges_hour.parquet",
        "size_sensitivity_csv": report_dir / "real_size_sensitivity_hour.csv",
        "manifest_json": metadata_dir / "dataset_manifest.json",
    }
    if not refresh and all(path.exists() for path in cached_paths.values()):
        selected_pools = pd.read_csv(cached_paths["selected_pools_csv"])
        pool_ohlcv = pd.read_parquet(cached_paths["pool_ohlcv_parquet"])
        market_edges = pd.read_parquet(cached_paths["market_edges_parquet"])
        size_sensitivity = pd.read_csv(cached_paths["size_sensitivity_csv"])
        manifest = _read_json(cached_paths["manifest_json"])
        return {
            "spec": spec,
            "selected_pools": selected_pools,
            "pool_ohlcv": pool_ohlcv,
            "market_edges": market_edges,
            "size_sensitivity": size_sensitivity,
            "paths": cached_paths,
            "manifest": manifest,
            "loaded_from_cache": True,
        }

    with requests.Session() as session:
        pool_search = _fetch_or_load_pool_search(
            market_slug,
            spec,
            raw_dir,
            refresh=refresh,
            session=session,
            settings=fetch_settings,
        )
        pool_search["network"] = spec["network"]
        target_asset = spec["target_asset"].upper()
        target_pool_search = pool_search[
            (pool_search["pool_asset_0"] == target_asset) | (pool_search["pool_asset_1"] == target_asset)
        ].copy()
        if len(target_pool_search) >= 2:
            pool_search = target_pool_search
        selected_pools = _select_pools(pool_search, int(spec.get("pool_count", 3)))
        selected_pools["selected_for_dataset"] = True

        ohlcv_frames = []
        for pool_record in selected_pools.to_dict("records"):
            ohlcv_frames.append(
                _fetch_or_load_ohlcv(
                    pool_record,
                    spec["target_asset"],
                    raw_dir,
                    refresh=refresh,
                    session=session,
                    settings=fetch_settings,
                )
            )

    pool_ohlcv = pd.concat(ohlcv_frames, ignore_index=True) if ohlcv_frames else pd.DataFrame()
    if not pool_ohlcv.empty:
        row_counts = pool_ohlcv.groupby("pool_address").size()
        valid_pool_addresses = row_counts[row_counts >= fetch_settings.min_pool_ohlcv_rows].index
        pool_ohlcv = pool_ohlcv[pool_ohlcv["pool_address"].isin(valid_pool_addresses)].reset_index(drop=True)
        selected_pools = selected_pools[selected_pools["pool_address"].isin(valid_pool_addresses)].reset_index(drop=True)

    market_edges = build_market_edges(pool_ohlcv)
    size_sensitivity = build_real_size_sensitivity(market_edges, assumptions)

    manifest = {
        "market_slug": market_slug,
        "source": "GeckoTerminal Public API",
        "api_base": GECKOTERMINAL_API_BASE,
        "network": spec["network"],
        "search_query": spec["search_query"],
        "target_asset": spec["target_asset"],
        "timeframe": fetch_settings.timeframe,
        "aggregate": fetch_settings.aggregate,
        "limit": fetch_settings.limit,
        "selected_pool_count": int(len(selected_pools)),
        "pool_ohlcv_rows": int(len(pool_ohlcv)),
        "market_edge_rows": int(len(market_edges)),
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    paths = write_dataset_outputs(
        selected_pools=selected_pools,
        pool_ohlcv=pool_ohlcv,
        market_edges=market_edges,
        size_sensitivity=size_sensitivity,
        raw_dir=raw_dir,
        curated_dir=curated_dir,
        report_dir=report_dir,
        metadata_dir=metadata_dir,
        manifest=manifest,
    )

    return {
        "spec": spec,
        "selected_pools": selected_pools,
        "pool_ohlcv": pool_ohlcv,
        "market_edges": market_edges,
        "size_sensitivity": size_sensitivity,
        "paths": paths,
        "manifest": manifest,
        "loaded_from_cache": False,
    }


def dataset_summary_frame(dataset: dict[str, Any]) -> pd.DataFrame:
    manifest = dataset["manifest"]
    paths = dataset["paths"]
    return pd.DataFrame(
        [
            {"field": "source", "value": manifest["source"]},
            {"field": "network", "value": manifest["network"]},
            {"field": "search_query", "value": manifest["search_query"]},
            {"field": "target_asset", "value": manifest["target_asset"]},
            {"field": "selected_pool_count", "value": manifest["selected_pool_count"]},
            {"field": "pool_ohlcv_rows", "value": manifest["pool_ohlcv_rows"]},
            {"field": "market_edge_rows", "value": manifest["market_edge_rows"]},
            {"field": "loaded_from_cache", "value": dataset["loaded_from_cache"]},
            {"field": "pool_ohlcv_path", "value": str(paths["pool_ohlcv_parquet"])},
            {"field": "market_edges_path", "value": str(paths["market_edges_parquet"])},
            {"field": "size_sensitivity_path", "value": str(paths["size_sensitivity_csv"])},
        ]
    )
