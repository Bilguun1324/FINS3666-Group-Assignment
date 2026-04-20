from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from web3 import Web3


def selector(signature: str) -> str:
    return Web3.keccak(text=signature)[:4].hex()


def topic(signature: str) -> str:
    return Web3.to_hex(Web3.keccak(text=signature))


TOKEN0_SELECTOR = f"0x{selector('token0()')}"
TOKEN1_SELECTOR = f"0x{selector('token1()')}"

UNISWAP_V2_SWAP_TOPIC = topic("Swap(address,uint256,uint256,uint256,uint256,address)")
SOLIDLY_V2_SWAP_TOPIC = topic("Swap(address,address,uint256,uint256,uint256,uint256)")
SYNC_112_TOPIC = topic("Sync(uint112,uint112)")
SYNC_256_TOPIC = topic("Sync(uint256,uint256)")
FEES_256_TOPIC = topic("Fees(address,uint256,uint256)")


EVENT_STYLES: dict[str, dict[str, str]] = {
    "uniswap_v2": {
        "swap_topic": UNISWAP_V2_SWAP_TOPIC,
        "sync_topic": SYNC_112_TOPIC,
    },
    "solidly_v2": {
        "swap_topic": SOLIDLY_V2_SWAP_TOPIC,
        "sync_topic": SYNC_256_TOPIC,
    },
    "equilibre_v2": {
        "swap_topic": UNISWAP_V2_SWAP_TOPIC,
        "sync_topic": SYNC_256_TOPIC,
        "fees_topic": FEES_256_TOPIC,
    },
}


TOPIC_TO_EVENT_NAME = {
    UNISWAP_V2_SWAP_TOPIC: "Swap",
    SOLIDLY_V2_SWAP_TOPIC: "Swap",
    SYNC_112_TOPIC: "Sync",
    SYNC_256_TOPIC: "Sync",
    FEES_256_TOPIC: "Fees",
}


CHAIN_CONNECTIONS: dict[str, dict[str, Any]] = {
    "ethereum": {
        "chain_id": 1,
        "env_var": "ETH_RPC_URL",
        "public_urls": [
            "https://rpc.flashbots.net",
            "https://ethereum.publicnode.com",
            "https://1rpc.io/eth",
            "https://eth.llamarpc.com",
        ],
    },
    "arbitrum": {
        "chain_id": 42161,
        "env_var": "ARBITRUM_RPC_URL",
        "public_urls": [
            "https://arbitrum-one-rpc.publicnode.com",
            "https://1rpc.io/arb",
            "https://arbitrum.meowrpc.com",
        ],
    },
    "optimism": {
        "chain_id": 10,
        "env_var": "OPTIMISM_RPC_URL",
        "public_urls": [
            "https://optimism-rpc.publicnode.com",
            "https://1rpc.io/op",
            "https://optimism.llamarpc.com",
        ],
    },
    "base": {
        "chain_id": 8453,
        "env_var": "BASE_RPC_URL",
        "public_urls": [
            "https://base-rpc.publicnode.com",
            "https://mainnet.base.org",
            "https://1rpc.io/base",
        ],
    },
    "kava": {
        "chain_id": 2222,
        "env_var": "KAVA_RPC_URL",
        "public_urls": [
            "https://evm.kava.io",
        ],
    },
}


@dataclass(frozen=True)
class TokenSpec:
    symbol: str
    address: str
    decimals: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "address", self.address.lower())


@dataclass(frozen=True)
class PairSpec:
    dex: str
    pair_address: str
    fee_bps: float
    event_style: str
    base_token: TokenSpec
    quote_token: TokenSpec
    role: str = "arb_pair"
    factory_address: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_address", self.pair_address.lower())
        if self.factory_address:
            object.__setattr__(self, "factory_address", self.factory_address.lower())
        if self.event_style not in EVENT_STYLES:
            raise ValueError(f"Unsupported event style: {self.event_style}")

    @property
    def fee_rate(self) -> float:
        return self.fee_bps / 10_000.0


@dataclass(frozen=True)
class MarketSpec:
    slug: str
    chain_key: str
    chain_name: str
    pair_label: str
    base_token: TokenSpec
    quote_token: TokenSpec
    gas_token: TokenSpec
    dexes: tuple[PairSpec, ...] = ()
    gas_reference_pair: PairSpec | None = None
    gas_reference_dex: str | None = None
    primary_trade_size_base: float | None = None
    trade_sizes_base: tuple[float, ...] = ()
    sample_window_days: int = 7
    stale_after_minutes: int = 15
    block_chunk_size: int = 4_000
    anchor_lookback_blocks: int = 500_000
    gas_units: int = 220_000
    priority_fee_gwei: float = 0.0
    blockers: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.blockers


@dataclass(frozen=True)
class BlockHeader:
    block_number: int
    block_hash: str
    parent_hash: str
    block_timestamp: datetime
    base_fee_per_gas_wei: int
    gas_used: int
    gas_limit: int


class JsonRpcError(RuntimeError):
    pass


class RpcClient:
    def __init__(
        self,
        rpc_urls: str | list[str],
        cache_dir: str | Path,
        *,
        timeout_seconds: int = 30,
        max_retries: int = 4,
        retry_backoff_seconds: float = 1.5,
        session: requests.Session | None = None,
    ) -> None:
        self.rpc_urls = [rpc_urls] if isinstance(rpc_urls, str) else list(rpc_urls)
        if not self.rpc_urls:
            raise ValueError("RpcClient requires at least one RPC URL")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.session = session or requests.Session()

    @staticmethod
    def to_hex_quantity(value: int) -> str:
        return hex(int(value))

    def _cache_path(self, method: str, params: list[Any]) -> Path:
        payload = json.dumps([method, params], sort_keys=True, default=str).encode("utf-8")
        return self.cache_dir / f"{hashlib.sha256(payload).hexdigest()}.json"

    def _post_json(self, payload: Any, rpc_url: str) -> Any:
        response = self.session.post(rpc_url, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def _rpc_url_for_attempt(self, attempt_index: int) -> str:
        return self.rpc_urls[attempt_index % len(self.rpc_urls)]

    def call(self, method: str, params: list[Any], *, use_cache: bool = False) -> Any:
        cache_path = self._cache_path(method, params)
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text())["result"]

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            rpc_url = self._rpc_url_for_attempt(attempt - 1)
            try:
                payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
                response = self._post_json(payload, rpc_url)
                if "error" in response:
                    raise JsonRpcError(f"{method} failed: {response['error']}")
                result = response["result"]
                if use_cache:
                    cache_path.write_text(json.dumps({"result": result}))
                return result
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_backoff_seconds * attempt)
        raise JsonRpcError(f"{method} failed after {self.max_retries} attempts: {last_error}")

    def batch_call(self, calls: list[tuple[str, list[Any]]], *, use_cache: bool = False) -> list[Any]:
        results: list[Any] = [None] * len(calls)
        pending: list[tuple[int, str, list[Any], Path]] = []
        for index, (method, params) in enumerate(calls):
            cache_path = self._cache_path(method, params)
            if use_cache and cache_path.exists():
                results[index] = json.loads(cache_path.read_text())["result"]
            else:
                pending.append((index, method, params, cache_path))

        if not pending:
            return results

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            rpc_url = self._rpc_url_for_attempt(attempt - 1)
            try:
                payload = [
                    {"jsonrpc": "2.0", "id": index, "method": method, "params": params}
                    for index, method, params, _ in pending
                ]
                response_items = self._post_json(payload, rpc_url)
                if not isinstance(response_items, list):
                    raise JsonRpcError(f"batch call returned non-list response: {response_items}")
                response_by_id = {item["id"]: item for item in response_items}
                for index, method, params, cache_path in pending:
                    item = response_by_id.get(index)
                    if item is None:
                        raise JsonRpcError(f"batch call missing response for id={index}")
                    if "error" in item:
                        raise JsonRpcError(f"{method} failed: {item['error']}")
                    results[index] = item["result"]
                    if use_cache:
                        cache_path.write_text(json.dumps({"result": item['result']}))
                return results
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_backoff_seconds * attempt)
        raise JsonRpcError(f"batch call failed after {self.max_retries} attempts: {last_error}")

    def eth_call(self, to: str, data: str, *, block: str = "latest", use_cache: bool = True) -> str:
        return self.call("eth_call", [{"to": to, "data": data}, block], use_cache=use_cache)

    def eth_get_logs(
        self,
        *,
        address: str,
        from_block: int,
        to_block: int,
        topics: list[Any] | None = None,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        params = [{
            "address": address,
            "fromBlock": self.to_hex_quantity(from_block),
            "toBlock": self.to_hex_quantity(to_block),
        }]
        if topics is not None:
            params[0]["topics"] = topics
        return self.call("eth_getLogs", params, use_cache=use_cache)

    def get_latest_block_number(self) -> int:
        return int(self.call("eth_blockNumber", [], use_cache=False), 16)

    @staticmethod
    def _parse_block_header(raw: dict[str, Any]) -> BlockHeader:
        return BlockHeader(
            block_number=int(raw["number"], 16),
            block_hash=raw["hash"],
            parent_hash=raw["parentHash"],
            block_timestamp=datetime.fromtimestamp(int(raw["timestamp"], 16), tz=UTC),
            base_fee_per_gas_wei=int(raw.get("baseFeePerGas", "0x0"), 16),
            gas_used=int(raw["gasUsed"], 16),
            gas_limit=int(raw["gasLimit"], 16),
        )

    def get_block_by_number(self, block_number: int, *, use_cache: bool = True) -> BlockHeader:
        raw = self.call(
            "eth_getBlockByNumber",
            [self.to_hex_quantity(block_number), False],
            use_cache=use_cache,
        )
        return self._parse_block_header(raw)

    def get_blocks_by_number(
        self,
        block_numbers: list[int],
        *,
        use_cache: bool = True,
        batch_size: int = 50,
    ) -> list[BlockHeader]:
        unique_blocks = sorted(set(block_numbers))
        headers: list[BlockHeader] = []
        for start in range(0, len(unique_blocks), batch_size):
            chunk = unique_blocks[start : start + batch_size]
            calls = [("eth_getBlockByNumber", [self.to_hex_quantity(block_number), False]) for block_number in chunk]
            raw_headers = self.batch_call(calls, use_cache=use_cache)
            headers.extend(self._parse_block_header(raw_header) for raw_header in raw_headers)
        return headers

    def block_timestamp(self, block_number: int) -> datetime:
        return self.get_block_by_number(block_number, use_cache=True).block_timestamp

    def find_block_by_timestamp(self, target: datetime, *, direction: str = "after") -> int:
        if target.tzinfo is None:
            raise ValueError("target must be timezone-aware")
        latest = self.get_latest_block_number()
        earliest = 1
        earliest_ts = self.block_timestamp(earliest)
        latest_ts = self.block_timestamp(latest)
        if target <= earliest_ts:
            return earliest
        if target >= latest_ts:
            return latest

        lo = earliest
        hi = latest
        while lo <= hi:
            mid = (lo + hi) // 2
            mid_ts = self.block_timestamp(mid)
            if mid_ts < target:
                lo = mid + 1
            elif mid_ts > target:
                hi = mid - 1
            else:
                return mid

        if direction == "after":
            return lo
        if direction == "before":
            return hi
        raise ValueError("direction must be 'after' or 'before'")


MAINNET_WETH = TokenSpec("WETH", "0xc02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", 18)
MAINNET_WBTC = TokenSpec("WBTC", "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", 8)
MAINNET_USDC = TokenSpec("USDC", "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", 6)
MAINNET_USDT = TokenSpec("USDT", "0xdac17f958d2ee523a2206206994597c13d831ec7", 6)
MAINNET_DAI = TokenSpec("DAI", "0x6b175474e89094c44da98b954eedeac495271d0f", 18)

ARBITRUM_WETH = TokenSpec("WETH", "0x82af49447d8a07e3bd95bd0d56f35241523fbab1", 18)
ARBITRUM_WBTC = TokenSpec("WBTC", "0x2f2a2543b76a4166549f7aab2e75bef0aefc5b0f", 8)

OPTIMISM_WETH = TokenSpec("WETH", "0x4200000000000000000000000000000000000006", 18)
OPTIMISM_WBTC = TokenSpec("WBTC", "0x68f180fcce6836688e9084f035309e29bf0a2095", 8)

BASE_WETH = TokenSpec("WETH", "0x4200000000000000000000000000000000000006", 18)
BASE_WBTC = TokenSpec("WBTC", "0x0555e30da8f98308edb960aa94c0db47230d2b9c", 8)

KAVA_WKAVA = TokenSpec("WKAVA", "0xc86c7c0efbd6a49b35e8714c5f59d99de09a225b", 18)
KAVA_ATOM = TokenSpec("ATOM", "0x15932e26f5bd4923d46a2b205191c4b5d5f43fe3", 6)
KAVA_USDT = TokenSpec("USDt", "0x919c1c267bc06a7039e03fcc2ef738525769109c", 6)


SECOND_LEVEL_MARKETS: dict[str, MarketSpec] = {
    "03_arbitrum_wbtc_weth": MarketSpec(
        slug="arbitrum_wbtc_weth",
        chain_key="arbitrum",
        chain_name="Arbitrum",
        pair_label="WBTC/WETH",
        base_token=ARBITRUM_WBTC,
        quote_token=ARBITRUM_WETH,
        gas_token=ARBITRUM_WETH,
        dexes=(
            PairSpec(
                dex="sushiswap_arbitrum",
                pair_address="0x515e252b2b5c22b4b2b6df66c2ebeea871aa4d69",
                factory_address="0xc35dadb65012ec5796536bd9864ed8773abc74c4",
                fee_bps=30.0,
                event_style="uniswap_v2",
                base_token=ARBITRUM_WBTC,
                quote_token=ARBITRUM_WETH,
            ),
            PairSpec(
                dex="uniswap_v2_arbitrum",
                pair_address="0x8c1d83a25ee2da1643a5d937562682b1ac6c856b",
                factory_address="0xf1d7cc64fb4452f05c498126312ebe29f30fbcf9",
                fee_bps=30.0,
                event_style="uniswap_v2",
                base_token=ARBITRUM_WBTC,
                quote_token=ARBITRUM_WETH,
            ),
        ),
        primary_trade_size_base=0.01,
        trade_sizes_base=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05),
        block_chunk_size=20_000,
        notes=(
            "This is a real second-level reserve reconstruction on two Arbitrum constant-product pools.",
            "Gas uses on-chain block base fee only with a fixed gas-units proxy and no synthetic price interpolation.",
        ),
    ),
    "04_optimism_wbtc_weth": MarketSpec(
        slug="optimism_wbtc_weth",
        chain_key="optimism",
        chain_name="Optimism",
        pair_label="WBTC/WETH",
        base_token=OPTIMISM_WBTC,
        quote_token=OPTIMISM_WETH,
        gas_token=OPTIMISM_WETH,
        dexes=(
            PairSpec(
                dex="uniswap_v2_optimism",
                pair_address="0x8782edc55c8514bf30e36a3585afcadbff525c77",
                factory_address="0x0c3c1c532f1e39edf36be9fe0be1410313e074bf",
                fee_bps=30.0,
                event_style="uniswap_v2",
                base_token=OPTIMISM_WBTC,
                quote_token=OPTIMISM_WETH,
            ),
            PairSpec(
                dex="velodrome_v2",
                pair_address="0x17547215d96696de1e1c8a9791d6da18f92045af",
                factory_address="0xf1046053aa5682b4f9a81b5481394da16be5ff5a",
                fee_bps=30.0,
                event_style="solidly_v2",
                base_token=OPTIMISM_WBTC,
                quote_token=OPTIMISM_WETH,
            ),
        ),
        primary_trade_size_base=0.01,
        trade_sizes_base=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05),
        block_chunk_size=20_000,
        notes=(
            "This notebook switches from the earlier v3/hourly screen to two reserve-based pools that emit reconstructable Sync/Swap events.",
            "Velodrome V2 uses a Solidly-style event layout; the decoder handles that explicitly.",
        ),
    ),
    "05_base_wbtc_weth": MarketSpec(
        slug="base_wbtc_weth",
        chain_key="base",
        chain_name="Base",
        pair_label="WBTC/WETH",
        base_token=BASE_WBTC,
        quote_token=BASE_WETH,
        gas_token=BASE_WETH,
        blockers=(
            "The canonical Base WBTC/WETH market does not currently have two verified reserve-based venues in this repo's search universe.",
            "Aerodrome is reconstructable, but the other liquid canonical venue is Uniswap v3, which is not a reserve-based second-level pipeline like notebook 02.",
            "Using a different BTC wrapper to force a second venue would not be an honest same-market backtest.",
        ),
        notes=(
            "This notebook stays intact by writing an explicit blocker manifest instead of pretending the hourly screen is second-level data.",
        ),
    ),
    "06_mainnet_usdc_usdt": MarketSpec(
        slug="mainnet_usdc_usdt",
        chain_key="ethereum",
        chain_name="Ethereum Mainnet",
        pair_label="USDC/USDT",
        base_token=MAINNET_USDC,
        quote_token=MAINNET_USDT,
        gas_token=MAINNET_WETH,
        dexes=(
            PairSpec(
                dex="uniswap_v2",
                pair_address="0x3041cbd36888becc7bbcbc0045e3b1f144466f5f",
                factory_address="0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f",
                fee_bps=30.0,
                event_style="uniswap_v2",
                base_token=MAINNET_USDC,
                quote_token=MAINNET_USDT,
            ),
            PairSpec(
                dex="sushiswap_v2",
                pair_address="0xd86a120a06255df8d4e2248ab04d4267e23adfaa",
                factory_address="0xc0aee478e3658e2610c5f7a4a2e1777ce9e4f2ac",
                fee_bps=30.0,
                event_style="uniswap_v2",
                base_token=MAINNET_USDC,
                quote_token=MAINNET_USDT,
            ),
        ),
        gas_reference_pair=PairSpec(
            dex="weth_usdt_uniswap_v2_gas_reference",
            pair_address="0x0d4a11d5eeaac28ec3f61d100daf4d40471f1852",
            factory_address="0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f",
            fee_bps=30.0,
            event_style="uniswap_v2",
            base_token=MAINNET_WETH,
            quote_token=MAINNET_USDT,
            role="gas_reference",
        ),
        primary_trade_size_base=5_000.0,
        trade_sizes_base=(100.0, 250.0, 500.0, 1_000.0, 2_500.0, 5_000.0, 10_000.0),
        notes=(
            "Gas is converted into USDT using a real WETH/USDT pool, not a USDC equals USDT shortcut.",
        ),
    ),
    "07_mainnet_dai_usdc": MarketSpec(
        slug="mainnet_dai_usdc",
        chain_key="ethereum",
        chain_name="Ethereum Mainnet",
        pair_label="DAI/USDC",
        base_token=MAINNET_DAI,
        quote_token=MAINNET_USDC,
        gas_token=MAINNET_WETH,
        dexes=(
            PairSpec(
                dex="uniswap_v2",
                pair_address="0xae461ca67b15dc8dc81ce7615e0320da1a9ab8d5",
                factory_address="0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f",
                fee_bps=30.0,
                event_style="uniswap_v2",
                base_token=MAINNET_DAI,
                quote_token=MAINNET_USDC,
            ),
            PairSpec(
                dex="sushiswap_v2",
                pair_address="0xaaf5110db6e744ff70fb339de037b990a20bdace",
                factory_address="0xc0aee478e3658e2610c5f7a4a2e1777ce9e4f2ac",
                fee_bps=30.0,
                event_style="uniswap_v2",
                base_token=MAINNET_DAI,
                quote_token=MAINNET_USDC,
            ),
        ),
        gas_reference_pair=PairSpec(
            dex="weth_usdc_uniswap_v2_gas_reference",
            pair_address="0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc",
            factory_address="0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f",
            fee_bps=30.0,
            event_style="uniswap_v2",
            base_token=MAINNET_WETH,
            quote_token=MAINNET_USDC,
            role="gas_reference",
        ),
        primary_trade_size_base=5_000.0,
        trade_sizes_base=(100.0, 250.0, 500.0, 1_000.0, 2_500.0, 5_000.0, 10_000.0),
    ),
    "08_mainnet_weth_usdc": MarketSpec(
        slug="mainnet_weth_usdc",
        chain_key="ethereum",
        chain_name="Ethereum Mainnet",
        pair_label="WETH/USDC",
        base_token=MAINNET_WETH,
        quote_token=MAINNET_USDC,
        gas_token=MAINNET_WETH,
        dexes=(
            PairSpec(
                dex="uniswap_v2",
                pair_address="0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc",
                factory_address="0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f",
                fee_bps=30.0,
                event_style="uniswap_v2",
                base_token=MAINNET_WETH,
                quote_token=MAINNET_USDC,
            ),
            PairSpec(
                dex="sushiswap_v2",
                pair_address="0x397ff1542f962076d0bfe58ea045ffa2d347aca0",
                factory_address="0xc0aee478e3658e2610c5f7a4a2e1777ce9e4f2ac",
                fee_bps=30.0,
                event_style="uniswap_v2",
                base_token=MAINNET_WETH,
                quote_token=MAINNET_USDC,
            ),
        ),
        gas_reference_dex="uniswap_v2",
        primary_trade_size_base=0.25,
        trade_sizes_base=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        notes=(
            "Gas is converted into USDC using the observed WETH/USDC market itself rather than a synthetic FX assumption.",
        ),
    ),
    "09_mainnet_wbtc_usdc": MarketSpec(
        slug="mainnet_wbtc_usdc",
        chain_key="ethereum",
        chain_name="Ethereum Mainnet",
        pair_label="WBTC/USDC",
        base_token=MAINNET_WBTC,
        quote_token=MAINNET_USDC,
        gas_token=MAINNET_WETH,
        dexes=(
            PairSpec(
                dex="uniswap_v2",
                pair_address="0x004375dff511095cc5a197a54140a24efef3a416",
                factory_address="0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f",
                fee_bps=30.0,
                event_style="uniswap_v2",
                base_token=MAINNET_WBTC,
                quote_token=MAINNET_USDC,
            ),
            PairSpec(
                dex="pancakeswap_v2",
                pair_address="0xbc03ce3f4236c82a3a3270af02c15a6a42857e90",
                factory_address="0x1097053fd2ea711dad45caccc45eff7548fcb362",
                fee_bps=25.0,
                event_style="uniswap_v2",
                base_token=MAINNET_WBTC,
                quote_token=MAINNET_USDC,
            ),
        ),
        gas_reference_pair=PairSpec(
            dex="weth_usdc_uniswap_v2_gas_reference",
            pair_address="0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc",
            factory_address="0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f",
            fee_bps=30.0,
            event_style="uniswap_v2",
            base_token=MAINNET_WETH,
            quote_token=MAINNET_USDC,
            role="gas_reference",
        ),
        primary_trade_size_base=0.01,
        trade_sizes_base=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05),
    ),
    "10_solana_research": MarketSpec(
        slug="solana_research",
        chain_key="solana",
        chain_name="Solana",
        pair_label="SOL/USDC",
        base_token=TokenSpec("SOL", "sol", 9),
        quote_token=TokenSpec("USDC", "usdc", 6),
        gas_token=TokenSpec("SOL", "sol", 9),
        blockers=(
            "The repo does not contain a Solana archive/event parser that can reconstruct one-second pool state from on-chain swap and reserve events.",
            "The available saved Solana market data is hourly GeckoTerminal OHLCV, and upsampling it to seconds would not be real data.",
            "Completing this honestly would require a separate Solana historical ingestion pipeline beyond the existing notebook 02 architecture.",
        ),
    ),
    "11_cosmos_research": MarketSpec(
        slug="cosmos_research",
        chain_key="kava",
        chain_name="Cosmos Proxy (Kava EVM)",
        pair_label="ATOM/USDt",
        base_token=KAVA_ATOM,
        quote_token=KAVA_USDT,
        gas_token=KAVA_WKAVA,
        blockers=(
            "The tested Kava proxy pools in this repo are not exposing a notebook-02-style reserve event interface that supports honest one-second reconstruction.",
            "The direct Wagmi candidates expose token addresses but do not expose the V2 or Solidly reserve/swap surface needed here.",
            "This notebook therefore writes a real blocker manifest instead of fabricating second-level data for a Cosmos proxy market.",
        ),
        notes=(
            "This remains a Kava-based Cosmos proxy and not a direct Osmosis backtest.",
        ),
    ),
}


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "notebooks").exists() and (candidate / "data").exists():
            return candidate
    return current


def frequency_tag(freq: str) -> str:
    return (
        freq.lower()
        .replace(" ", "")
        .replace("minute", "m")
        .replace("min", "m")
        .replace("second", "s")
        .replace("sec", "s")
    )


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(parsed) else parsed


def _decode_address_output(output: str) -> str:
    return f"0x{output[-40:]}".lower()


def _probe_rpc_candidate(url: str) -> bool:
    try:
        response = requests.post(
            url,
            json={"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []},
            timeout=10,
        )
        response.raise_for_status()
        body = response.json()
        latest_hex = body.get("result") if isinstance(body, dict) else None
        if not latest_hex:
            return False
        latest_block = max(int(latest_hex, 16) - 1, 1)
        test = requests.post(
            url,
            json={"jsonrpc": "2.0", "id": 1, "method": "eth_getBlockByNumber", "params": [hex(latest_block), False]},
            timeout=10,
        )
        test.raise_for_status()
        test_body = test.json()
        return bool(isinstance(test_body, dict) and test_body.get("result"))
    except Exception:
        return False


def select_rpc_urls(chain_key: str) -> tuple[list[str], str, str]:
    config = CHAIN_CONNECTIONS[chain_key]
    env_var = config["env_var"]
    env_value = os.getenv(env_var)
    if env_value:
        return [env_value], env_var, "environment"

    working = [url for url in config["public_urls"] if _probe_rpc_candidate(url)]
    if working:
        return working, env_var, "public_rpc_fallback"
    raise RuntimeError(f"No working RPC endpoint found for {chain_key} ({env_var})")


def market_paths(project_root: Path, slug: str) -> dict[str, Path]:
    data_root = project_root / "data" / slug
    output_root = project_root / "outputs" / slug
    return {
        "data_root": data_root,
        "raw_dir": data_root / "raw",
        "curated_dir": data_root / "curated",
        "report_dir": output_root / "report",
        "metadata_dir": output_root / "metadata",
        "rpc_cache_dir": data_root / "raw" / "rpc_cache",
        "raw_logs": data_root / "raw" / "event_logs.parquet",
        "raw_logs_preview": data_root / "raw" / "event_logs_preview.csv",
        "raw_blocks": data_root / "raw" / "block_headers.parquet",
        "raw_blocks_preview": data_root / "raw" / "block_headers_preview.csv",
        "raw_pair_metadata": data_root / "raw" / "pair_metadata.parquet",
        "raw_pair_metadata_preview": data_root / "raw" / "pair_metadata_preview.csv",
        "raw_manifest": data_root / "raw" / "source_manifest.json",
        "events_curated": data_root / "curated" / "events_curated.parquet",
        "events_curated_preview": data_root / "curated" / "events_curated_preview.csv",
        "swaps_raw": data_root / "curated" / "swaps_raw.parquet",
        "swaps_raw_preview": data_root / "curated" / "swaps_raw_preview.csv",
        "pool_state": data_root / "curated" / "pool_state_1s.parquet",
        "pool_state_preview": data_root / "curated" / "pool_state_1s_preview.csv",
        "gas_reference_state": data_root / "curated" / "gas_reference_state_1s.parquet",
        "gas_reference_state_preview": data_root / "curated" / "gas_reference_state_1s_preview.csv",
        "arb_labels": data_root / "curated" / "arb_labels_1s.parquet",
        "arb_labels_preview": data_root / "curated" / "arb_labels_1s_preview.csv",
        "qc_report": data_root / "curated" / "qc_report_1s.json",
        "size_sensitivity": output_root / "report" / "opportunity_size_sensitivity_1s.csv",
        "opportunity_windows": output_root / "report" / "opportunity_windows_1s.csv",
        "dataset_manifest": output_root / "metadata" / "dataset_manifest.json",
    }


RAW_PREVIEW_COLUMNS = {
    "event_logs": ["role", "dex", "pair_address", "block_number", "transaction_hash", "log_index", "event_name", "topic0"],
    "block_headers": ["block_number", "block_timestamp", "base_fee_per_gas_wei", "gas_used", "gas_limit"],
    "pair_metadata": ["role", "dex", "pair_address", "event_style", "base_symbol", "quote_symbol", "fee_bps"],
}

CURATED_PREVIEW_COLUMNS = {
    "events_curated": [
        "timestamp",
        "role",
        "dex",
        "pair_address",
        "block_number",
        "transaction_hash",
        "log_index",
        "event_name",
        "reserve_base_post",
        "reserve_quote_post",
        "mid_price_quote_per_base",
    ],
    "swaps_raw": [
        "timestamp",
        "role",
        "dex",
        "pair_address",
        "block_number",
        "transaction_hash",
        "log_index",
        "amount_base",
        "amount_quote",
        "volume_base",
        "volume_quote",
        "trade_direction",
    ],
    "pool_state_1s": [
        "timestamp",
        "role",
        "dex",
        "pair_address",
        "mid_price_quote_per_base",
        "reserve_base",
        "reserve_quote",
        "swap_count",
        "volume_base",
        "volume_quote",
        "stale_state",
    ],
    "gas_reference_state_1s": [
        "timestamp",
        "role",
        "dex",
        "pair_address",
        "mid_price_quote_per_base",
        "reserve_base",
        "reserve_quote",
        "stale_state",
    ],
    "arb_labels_1s": [
        "timestamp",
        "buy_dex",
        "sell_dex",
        "trade_size_base",
        "gross_edge_bps",
        "fee_cost_bps",
        "gas_cost_quote",
        "net_edge_quote",
        "net_edge_bps",
        "opportunity_flag",
        "stale_state",
    ],
}


def _preview_frame(df: pd.DataFrame, preferred_columns: list[str], *, max_rows: int = 200) -> pd.DataFrame:
    columns = [column for column in preferred_columns if column in df.columns]
    return df[columns].head(max_rows) if columns else df.head(max_rows)


def _write_preview_csv(path: Path, df: pd.DataFrame, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _preview_frame(df, columns).to_csv(path, index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build_pair_metadata(rpc: RpcClient, pair_specs: list[PairSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in pair_specs:
        token0 = _decode_address_output(rpc.eth_call(pair.pair_address, TOKEN0_SELECTOR))
        token1 = _decode_address_output(rpc.eth_call(pair.pair_address, TOKEN1_SELECTOR))
        expected = {pair.base_token.address, pair.quote_token.address}
        observed = {token0, token1}
        if observed != expected:
            raise ValueError(
                f"{pair.dex} token mismatch: observed {sorted(observed)} expected {sorted(expected)}"
            )
        rows.append(
            {
                "role": pair.role,
                "dex": pair.dex,
                "pair_address": pair.pair_address,
                "factory_address": pair.factory_address,
                "event_style": pair.event_style,
                "token0": token0,
                "token1": token1,
                "base_symbol": pair.base_token.symbol,
                "base_token_address": pair.base_token.address,
                "base_token_decimals": pair.base_token.decimals,
                "quote_symbol": pair.quote_token.symbol,
                "quote_token_address": pair.quote_token.address,
                "quote_token_decimals": pair.quote_token.decimals,
                "base_is_token0": token0 == pair.base_token.address,
                "fee_bps": pair.fee_bps,
                "fee_rate": pair.fee_rate,
            }
        )
    return pd.DataFrame(rows)


def resolve_block_window(spec: MarketSpec, rpc: RpcClient) -> tuple[int, int, datetime, datetime]:
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=spec.sample_window_days)
    start_block = rpc.find_block_by_timestamp(start_time, direction="after")
    end_block = rpc.find_block_by_timestamp(end_time, direction="before")
    return start_block, end_block, start_time, end_time


def _event_topics_for_style(event_style: str) -> list[str]:
    style = EVENT_STYLES[event_style]
    return [
        topic_value
        for key, topic_value in style.items()
        if key in {"sync_topic", "swap_topic", "fees_topic"}
    ]


def _sync_topic_for_style(event_style: str) -> str:
    return EVENT_STYLES[event_style]["sync_topic"]


def _collect_logs_adaptive(
    rpc: RpcClient,
    *,
    address: str,
    from_block: int,
    to_block: int,
    topics: list[Any],
    min_chunk_size: int = 50,
) -> list[dict[str, Any]]:
    try:
        return rpc.eth_get_logs(
            address=address,
            from_block=from_block,
            to_block=to_block,
            topics=topics,
            use_cache=True,
        )
    except Exception:
        if from_block >= to_block or (to_block - from_block + 1) <= min_chunk_size:
            raise
        mid = (from_block + to_block) // 2
        left = _collect_logs_adaptive(
            rpc,
            address=address,
            from_block=from_block,
            to_block=mid,
            topics=topics,
            min_chunk_size=min_chunk_size,
        )
        right = _collect_logs_adaptive(
            rpc,
            address=address,
            from_block=mid + 1,
            to_block=to_block,
            topics=topics,
            min_chunk_size=min_chunk_size,
        )
        return left + right


def _find_anchor_sync_log(
    rpc: RpcClient,
    pair: PairSpec,
    *,
    start_block: int,
    lookback_blocks: int,
    chunk_size: int,
) -> dict[str, Any] | None:
    if start_block <= 1:
        return None
    search_end = start_block - 1
    search_start_limit = max(1, start_block - lookback_blocks)
    sync_topic = _sync_topic_for_style(pair.event_style)
    while search_end >= search_start_limit:
        chunk_start = max(search_start_limit, search_end - chunk_size + 1)
        logs = _collect_logs_adaptive(
            rpc,
            address=pair.pair_address,
            from_block=chunk_start,
            to_block=search_end,
            topics=[[sync_topic]],
        )
        if logs:
            return sorted(logs, key=lambda item: (int(item["blockNumber"], 16), int(item["logIndex"], 16)))[-1]
        search_end = chunk_start - 1
    return None


def collect_pair_logs(
    rpc: RpcClient,
    pair: PairSpec,
    *,
    start_block: int,
    end_block: int,
    chunk_size: int,
    lookback_blocks: int,
) -> list[dict[str, Any]]:
    raw_logs: list[dict[str, Any]] = []
    anchor = _find_anchor_sync_log(
        rpc,
        pair,
        start_block=start_block,
        lookback_blocks=lookback_blocks,
        chunk_size=chunk_size,
    )
    if anchor is not None:
        raw_logs.append(anchor)

    event_topics = _event_topics_for_style(pair.event_style)
    for chunk_start in range(start_block, end_block + 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size - 1, end_block)
        raw_logs.extend(
            _collect_logs_adaptive(
                rpc,
                address=pair.pair_address,
                from_block=chunk_start,
                to_block=chunk_end,
                topics=[event_topics],
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


def fetch_block_headers(rpc: RpcClient, block_numbers: list[int]) -> pd.DataFrame:
    headers = rpc.get_blocks_by_number(block_numbers, use_cache=True)
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


def _decode_words(data: str) -> list[int]:
    payload = data[2:] if data.startswith("0x") else data
    if not payload:
        return []
    return [int(payload[index : index + 64], 16) for index in range(0, len(payload), 64)]


def _decode_event_fields(topic0: str, data: str) -> dict[str, int]:
    words = _decode_words(data)
    if topic0 in {SYNC_112_TOPIC, SYNC_256_TOPIC} and len(words) >= 2:
        return {"reserve0_raw": words[0], "reserve1_raw": words[1]}
    if topic0 == FEES_256_TOPIC and len(words) >= 2:
        return {"fee0_raw": words[0], "fee1_raw": words[1]}
    if topic0 in {UNISWAP_V2_SWAP_TOPIC, SOLIDLY_V2_SWAP_TOPIC} and len(words) >= 4:
        return {
            "amount0_in_raw": words[0],
            "amount1_in_raw": words[1],
            "amount0_out_raw": words[2],
            "amount1_out_raw": words[3],
        }
    return {}


def normalize_event_logs(logs_df: pd.DataFrame, blocks_df: pd.DataFrame, pair_metadata_df: pd.DataFrame) -> pd.DataFrame:
    if logs_df.empty:
        return pd.DataFrame()

    events = (
        logs_df.merge(blocks_df, on="block_number", how="left")
        .merge(pair_metadata_df, on=["role", "dex", "pair_address"], how="left", suffixes=("", "_meta"))
        .sort_values(["pair_address", "block_number", "log_index"])
        .reset_index(drop=True)
    )
    decoded = pd.DataFrame([_decode_event_fields(row.topic0, row.data) for row in events.itertuples()])
    events = pd.concat([events, decoded], axis=1)

    events["reserve_base_post"] = np.where(events["base_is_token0"], events.get("reserve0_raw"), events.get("reserve1_raw"))
    events["reserve_quote_post"] = np.where(events["base_is_token0"], events.get("reserve1_raw"), events.get("reserve0_raw"))
    events["reserve_base_post"] = events.groupby("pair_address")["reserve_base_post"].ffill()
    events["reserve_quote_post"] = events.groupby("pair_address")["reserve_quote_post"].ffill()
    events["reserve_base_post"] = events["reserve_base_post"] / np.power(10.0, events["base_token_decimals"].astype(float))
    events["reserve_quote_post"] = events["reserve_quote_post"] / np.power(10.0, events["quote_token_decimals"].astype(float))
    events["mid_price_quote_per_base"] = events["reserve_quote_post"] / events["reserve_base_post"]
    events = events.rename(columns={"block_timestamp": "timestamp"})
    return events


def build_swaps_raw(events_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "timestamp",
        "role",
        "dex",
        "pair_address",
        "block_number",
        "transaction_hash",
        "log_index",
        "amount_base",
        "amount_quote",
        "volume_base",
        "volume_quote",
        "reserve_base_post",
        "reserve_quote_post",
        "mid_price_quote_per_base",
        "trade_direction",
    ]
    if events_df.empty:
        return pd.DataFrame(columns=columns)
    swaps = events_df[events_df["event_name"] == "Swap"].copy()
    if swaps.empty:
        return pd.DataFrame(columns=columns)

    swaps["amount0_net_raw"] = swaps["amount0_in_raw"] - swaps["amount0_out_raw"]
    swaps["amount1_net_raw"] = swaps["amount1_in_raw"] - swaps["amount1_out_raw"]
    swaps["amount_base_raw"] = np.where(swaps["base_is_token0"], swaps["amount0_net_raw"], swaps["amount1_net_raw"])
    swaps["amount_quote_raw"] = np.where(swaps["base_is_token0"], swaps["amount1_net_raw"], swaps["amount0_net_raw"])
    swaps["amount_base"] = swaps["amount_base_raw"] / np.power(10.0, swaps["base_token_decimals"].astype(float))
    swaps["amount_quote"] = swaps["amount_quote_raw"] / np.power(10.0, swaps["quote_token_decimals"].astype(float))
    swaps["volume_base"] = swaps["amount_base"].abs()
    swaps["volume_quote"] = swaps["amount_quote"].abs()
    swaps["trade_direction"] = np.where(
        swaps["amount_base"] < 0,
        "buy_base",
        "sell_base",
    )

    return swaps[columns].sort_values(["block_number", "log_index"]).reset_index(drop=True)


def build_pool_state(
    events_df: pd.DataFrame,
    swaps_df: pd.DataFrame,
    *,
    role: str,
    stale_after_minutes: int,
    frequency: str,
    window_start: datetime,
    window_end: datetime,
) -> pd.DataFrame:
    window_start_ts = pd.Timestamp(window_start).floor("s")
    window_end_ts = pd.Timestamp(window_end).floor("s")
    syncs = events_df[(events_df["event_name"] == "Sync") & (events_df["role"] == role)].copy()
    if syncs.empty:
        return pd.DataFrame()

    swap_stats = (
        swaps_df[swaps_df["role"] == role]
        .assign(timestamp=swaps_df[swaps_df["role"] == role]["timestamp"].dt.floor(frequency))
        .groupby(["role", "dex", "pair_address", "timestamp"], as_index=False)
        .agg(
            swap_count=("transaction_hash", "count"),
            volume_base=("volume_base", "sum"),
            volume_quote=("volume_quote", "sum"),
        )
    )

    frames: list[pd.DataFrame] = []
    for (dex, pair_address), group in syncs.groupby(["dex", "pair_address"]):
        state = group[["timestamp", "role", "dex", "pair_address", "reserve_base_post", "reserve_quote_post"]].copy()
        state["timestamp"] = pd.to_datetime(state["timestamp"], utc=True)
        state["last_sync_timestamp"] = state["timestamp"]
        state = (
            state.set_index("timestamp")
            .resample(frequency)
            .last()
            .ffill()
            .rename_axis("timestamp")
            .reset_index()
        )
        state = state[(state["timestamp"] >= window_start_ts) & (state["timestamp"] <= window_end_ts)].copy()
        state["role"] = role
        state["dex"] = dex
        state["pair_address"] = pair_address
        stats = swap_stats[(swap_stats["dex"] == dex) & (swap_stats["pair_address"] == pair_address)]
        state = state.merge(stats, on=["role", "dex", "pair_address", "timestamp"], how="left")
        state[["swap_count", "volume_base", "volume_quote"]] = state[["swap_count", "volume_base", "volume_quote"]].fillna(0.0)
        state["swap_count"] = state["swap_count"].astype(int)
        state["reserve_base"] = state["reserve_base_post"]
        state["reserve_quote"] = state["reserve_quote_post"]
        state["mid_price_quote_per_base"] = state["reserve_quote"] / state["reserve_base"]
        state["stale_state"] = (state["timestamp"] - state["last_sync_timestamp"]) > pd.Timedelta(minutes=stale_after_minutes)
        frames.append(
            state[
                [
                    "timestamp",
                    "role",
                    "dex",
                    "pair_address",
                    "reserve_base",
                    "reserve_quote",
                    "mid_price_quote_per_base",
                    "last_sync_timestamp",
                    "swap_count",
                    "volume_base",
                    "volume_quote",
                    "stale_state",
                ]
            ]
        )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "dex"]).reset_index(drop=True)


def build_base_fee(blocks_df: pd.DataFrame, *, frequency: str) -> pd.DataFrame:
    if blocks_df.empty:
        return pd.DataFrame(columns=["timestamp", "base_fee_per_gas_wei"])
    blocks = blocks_df.copy()
    blocks["block_timestamp"] = pd.to_datetime(blocks["block_timestamp"], utc=True)
    return (
        blocks.set_index("block_timestamp")[["base_fee_per_gas_wei"]]
        .sort_index()
        .resample(frequency)
        .last()
        .ffill()
        .rename_axis("timestamp")
        .reset_index()
    )


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


def _evaluate_direction(
    *,
    buy_reserve_base: np.ndarray,
    buy_reserve_quote: np.ndarray,
    sell_reserve_base: np.ndarray,
    sell_reserve_quote: np.ndarray,
    buy_fee_rate: float,
    sell_fee_rate: float,
    trade_size_base: float,
    gas_cost_quote: np.ndarray,
) -> dict[str, np.ndarray]:
    buy_cost_no_fee = amount_in_for_exact_out(
        trade_size_base,
        reserve_in=buy_reserve_quote,
        reserve_out=buy_reserve_base,
        fee_rate=0.0,
    )
    buy_cost_with_fee = amount_in_for_exact_out(
        trade_size_base,
        reserve_in=buy_reserve_quote,
        reserve_out=buy_reserve_base,
        fee_rate=buy_fee_rate,
    )
    sell_return_no_fee = amount_out_for_exact_in(
        trade_size_base,
        reserve_in=sell_reserve_base,
        reserve_out=sell_reserve_quote,
        fee_rate=0.0,
    )
    sell_return_with_fee = amount_out_for_exact_in(
        trade_size_base,
        reserve_in=sell_reserve_base,
        reserve_out=sell_reserve_quote,
        fee_rate=sell_fee_rate,
    )

    gross_edge_quote = sell_return_no_fee - buy_cost_no_fee
    fee_cost_quote = (buy_cost_with_fee - buy_cost_no_fee) + (sell_return_no_fee - sell_return_with_fee)
    net_edge_quote = sell_return_with_fee - buy_cost_with_fee - gas_cost_quote

    denominator = buy_cost_with_fee
    valid = (
        np.isfinite(buy_cost_no_fee)
        & np.isfinite(buy_cost_with_fee)
        & np.isfinite(sell_return_no_fee)
        & np.isfinite(sell_return_with_fee)
        & np.isfinite(gas_cost_quote)
        & (denominator > 0)
    )
    gross_edge_bps = np.full(denominator.shape, np.nan, dtype=float)
    fee_cost_bps = np.full(denominator.shape, np.nan, dtype=float)
    gas_cost_bps = np.full(denominator.shape, np.nan, dtype=float)
    net_edge_bps = np.full(denominator.shape, np.nan, dtype=float)
    gross_edge_bps[valid] = gross_edge_quote[valid] / denominator[valid] * 10_000.0
    fee_cost_bps[valid] = fee_cost_quote[valid] / denominator[valid] * 10_000.0
    gas_cost_bps[valid] = gas_cost_quote[valid] / denominator[valid] * 10_000.0
    net_edge_bps[valid] = net_edge_quote[valid] / denominator[valid] * 10_000.0
    return {
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


def build_gas_quote_series(
    spec: MarketSpec,
    arb_pool_state: pd.DataFrame,
    gas_reference_state: pd.DataFrame,
    *,
    frequency: str,
    window_start: datetime,
    window_end: datetime,
) -> pd.DataFrame:
    window_start_ts = pd.Timestamp(window_start).floor("s")
    window_end_ts = pd.Timestamp(window_end).floor("s")
    if spec.quote_token.address == spec.gas_token.address:
        return pd.DataFrame(
            {
                "timestamp": pd.date_range(window_start_ts, window_end_ts, freq=frequency, tz="UTC"),
                "gas_quote_price": 1.0,
                "gas_reference_stale_state": False,
            }
        )

    if spec.gas_reference_dex:
        reference = arb_pool_state[arb_pool_state["dex"] == spec.gas_reference_dex].copy()
    else:
        reference = gas_reference_state.copy()

    if reference.empty:
        raise ValueError(f"No gas reference state available for {spec.slug}")

    out = reference[["timestamp", "mid_price_quote_per_base", "stale_state"]].copy()
    out = out.rename(
        columns={
            "mid_price_quote_per_base": "gas_quote_price",
            "stale_state": "gas_reference_stale_state",
        }
    )
    out = out[(out["timestamp"] >= window_start_ts) & (out["timestamp"] <= window_end_ts)].copy()
    return out


def build_arb_labels(
    pool_state_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    pair_metadata_df: pd.DataFrame,
    *,
    trade_size_base: float,
    gas_units: int,
    priority_fee_gwei: float,
    gas_quote_series: pd.DataFrame,
    frequency: str,
) -> pd.DataFrame:
    if pool_state_df.empty:
        return pd.DataFrame()

    arb_metadata = pair_metadata_df[pair_metadata_df["role"] == "arb_pair"].copy()
    fee_lookup = arb_metadata.set_index("dex")["fee_rate"].to_dict()
    dexes = sorted(pool_state_df["dex"].dropna().unique().tolist())
    if len(dexes) != 2:
        raise ValueError("Arbitrage labeling expects exactly two arbitrage venues.")

    wide = (
        pool_state_df.pivot(index="timestamp", columns="dex", values=["reserve_base", "reserve_quote", "stale_state"])
        .sort_index()
    )
    gas_frame = build_base_fee(blocks_df, frequency=frequency).set_index("timestamp")
    gas_frame = gas_frame.join(gas_quote_series.set_index("timestamp"), how="outer")
    gas_frame["base_fee_per_gas_wei"] = gas_frame["base_fee_per_gas_wei"].ffill().fillna(0.0)
    gas_frame["gas_quote_price"] = gas_frame["gas_quote_price"].ffill()
    gas_frame["gas_reference_stale_state"] = (
        gas_frame["gas_reference_stale_state"].astype("boolean").ffill().fillna(False)
    )
    gas_frame.columns = pd.MultiIndex.from_tuples([("__meta__", column) for column in gas_frame.columns])
    wide = wide.join(gas_frame, how="left")

    base_fee = wide[("__meta__", "base_fee_per_gas_wei")].ffill().fillna(0.0).to_numpy(dtype=float)
    gas_quote_price = wide[("__meta__", "gas_quote_price")].ffill().to_numpy(dtype=float)
    gas_ref_stale = wide[("__meta__", "gas_reference_stale_state")].astype("boolean").fillna(True).to_numpy(dtype=bool)
    gas_cost_native = gas_units * (base_fee + priority_fee_gwei * 1e9) / 1e18
    gas_cost_quote = gas_cost_native * gas_quote_price

    dex_a, dex_b = dexes
    stale_a = wide[("stale_state", dex_a)].astype("boolean").fillna(True).to_numpy(dtype=bool)
    stale_b = wide[("stale_state", dex_b)].astype("boolean").fillna(True).to_numpy(dtype=bool)
    stale_any = stale_a | stale_b | gas_ref_stale

    direction_ab = _evaluate_direction(
        buy_reserve_base=wide[("reserve_base", dex_a)].to_numpy(dtype=float),
        buy_reserve_quote=wide[("reserve_quote", dex_a)].to_numpy(dtype=float),
        sell_reserve_base=wide[("reserve_base", dex_b)].to_numpy(dtype=float),
        sell_reserve_quote=wide[("reserve_quote", dex_b)].to_numpy(dtype=float),
        buy_fee_rate=float(fee_lookup[dex_a]),
        sell_fee_rate=float(fee_lookup[dex_b]),
        trade_size_base=trade_size_base,
        gas_cost_quote=gas_cost_quote,
    )
    direction_ba = _evaluate_direction(
        buy_reserve_base=wide[("reserve_base", dex_b)].to_numpy(dtype=float),
        buy_reserve_quote=wide[("reserve_quote", dex_b)].to_numpy(dtype=float),
        sell_reserve_base=wide[("reserve_base", dex_a)].to_numpy(dtype=float),
        sell_reserve_quote=wide[("reserve_quote", dex_a)].to_numpy(dtype=float),
        buy_fee_rate=float(fee_lookup[dex_b]),
        sell_fee_rate=float(fee_lookup[dex_a]),
        trade_size_base=trade_size_base,
        gas_cost_quote=gas_cost_quote,
    )

    score_ab = np.where(direction_ab["valid"], direction_ab["net_edge_quote"], -np.inf)
    score_ba = np.where(direction_ba["valid"], direction_ba["net_edge_quote"], -np.inf)
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
            "gross_edge_quote": choose("gross_edge_quote"),
            "gross_edge_bps": choose("gross_edge_bps"),
            "fee_cost_quote": choose("fee_cost_quote"),
            "fee_cost_bps": choose("fee_cost_bps"),
            "gas_cost_quote": choose("gas_cost_quote"),
            "gas_cost_bps": choose("gas_cost_bps"),
            "net_edge_quote": choose("net_edge_quote"),
            "net_edge_bps": choose("net_edge_bps"),
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
                "buy_dex",
                "sell_dex",
                "max_net_edge_bps",
                "mean_net_edge_bps",
                "max_net_profit_quote",
                "mean_net_profit_quote",
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
            max_net_edge_bps=("net_edge_bps", "max"),
            mean_net_edge_bps=("net_edge_bps", "mean"),
            max_net_profit_quote=("net_edge_quote", "max"),
            mean_net_profit_quote=("net_edge_quote", "mean"),
        )
        .sort_values(["max_net_edge_bps", "seconds"], ascending=[False, False])
        .reset_index(drop=True)
    )


def build_size_sensitivity(
    spec: MarketSpec,
    pool_state_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    pair_metadata_df: pd.DataFrame,
    gas_quote_series: pd.DataFrame,
    *,
    frequency: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    primary_labels = pd.DataFrame()
    for trade_size in spec.trade_sizes_base:
        labels = build_arb_labels(
            pool_state_df,
            blocks_df,
            pair_metadata_df,
            trade_size_base=trade_size,
            gas_units=spec.gas_units,
            priority_fee_gwei=spec.priority_fee_gwei,
            gas_quote_series=gas_quote_series,
            frequency=frequency,
        )
        if spec.primary_trade_size_base == trade_size:
            primary_labels = labels.copy()
        if labels.empty:
            summary_rows.append(
                {
                    "trade_size_base": trade_size,
                    "observed_seconds": 0,
                    "gross_positive_seconds": 0,
                    "net_positive_seconds": 0,
                    "max_net_profit_quote": np.nan,
                    "mean_net_profit_quote": np.nan,
                    "max_net_edge_bps": np.nan,
                    "mean_net_edge_bps": np.nan,
                }
            )
            continue
        outputs.append(labels.assign(trade_size_base=trade_size))
        summary_rows.append(
            {
                "trade_size_base": trade_size,
                "observed_seconds": int(len(labels)),
                "gross_positive_seconds": int((labels["gross_edge_quote"] > 0).sum()),
                "net_positive_seconds": int(labels["opportunity_flag"].sum()),
                "max_net_profit_quote": float(labels["net_edge_quote"].max()),
                "mean_net_profit_quote": float(labels["net_edge_quote"].mean()),
                "max_net_edge_bps": float(labels["net_edge_bps"].max()),
                "mean_net_edge_bps": float(labels["net_edge_bps"].mean()),
            }
        )
    if primary_labels.empty and outputs:
        primary_labels = outputs[0].copy()
    return pd.DataFrame(summary_rows), primary_labels


def run_qc(swaps_raw: pd.DataFrame, pool_state_df: pd.DataFrame, arb_labels_df: pd.DataFrame, *, frequency: str) -> dict[str, Any]:
    schema = {
        "swaps_raw": ["timestamp", "dex", "pair_address", "amount_base", "amount_quote", "volume_base", "volume_quote"],
        "pool_state_1s": ["timestamp", "dex", "pair_address", "reserve_base", "reserve_quote", "stale_state"],
        "arb_labels_1s": ["timestamp", "buy_dex", "sell_dex", "net_edge_quote", "net_edge_bps", "opportunity_flag"],
    }

    def missing(df: pd.DataFrame, required: list[str]) -> list[str]:
        return [column for column in required if column not in df.columns]

    gaps: dict[str, int] = {}
    if not pool_state_df.empty:
        for dex, group in pool_state_df.groupby("dex"):
            expected = pd.date_range(group["timestamp"].min(), group["timestamp"].max(), freq=frequency, tz="UTC")
            gaps[dex] = int(len(expected.difference(pd.DatetimeIndex(group["timestamp"]))))

    monotonic_violations = 0
    if not swaps_raw.empty:
        ordered = swaps_raw.sort_values(["block_number", "log_index"]).copy()
        monotonic_violations = int((ordered["timestamp"].diff().dropna() < pd.Timedelta(0)).sum())

    report = {
        "schema_errors": {
            "swaps_raw": missing(swaps_raw, schema["swaps_raw"]),
            "pool_state_1s": missing(pool_state_df, schema["pool_state_1s"]),
            "arb_labels_1s": missing(arb_labels_df, schema["arb_labels_1s"]),
        },
        "duplicate_swap_events": int(swaps_raw.duplicated(["transaction_hash", "log_index"]).sum()) if not swaps_raw.empty else 0,
        "monotonic_timestamp_violations": monotonic_violations,
        "non_positive_reserves": int(((pool_state_df["reserve_base"] <= 0) | (pool_state_df["reserve_quote"] <= 0)).sum()) if not pool_state_df.empty else 0,
        "pool_state_gaps_1s": gaps,
        "stale_state_nulls": int(pool_state_df["stale_state"].isna().sum()) if not pool_state_df.empty else 0,
    }
    report["passed"] = (
        all(len(errors) == 0 for errors in report["schema_errors"].values())
        and report["duplicate_swap_events"] == 0
        and report["monotonic_timestamp_violations"] == 0
        and report["non_positive_reserves"] == 0
        and all(value == 0 for value in report["pool_state_gaps_1s"].values())
        and report["stale_state_nulls"] == 0
    )
    return report


def _market_manifest(spec: MarketSpec, *, status: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "status": status,
        "market_slug": spec.slug,
        "chain_key": spec.chain_key,
        "chain_name": spec.chain_name,
        "pair_label": spec.pair_label,
        "base_token": asdict(spec.base_token),
        "quote_token": asdict(spec.quote_token),
        "gas_token": asdict(spec.gas_token),
        "sample_window_days": spec.sample_window_days,
        "state_frequency": "1s",
        "gas_units": spec.gas_units,
        "priority_fee_gwei": spec.priority_fee_gwei,
        "notes": list(spec.notes),
        "blockers": list(spec.blockers),
    }
    if extra:
        payload.update(extra)
    return payload


def _write_supported_outputs(
    paths: dict[str, Path],
    *,
    logs_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    pair_metadata_df: pd.DataFrame,
    raw_manifest: dict[str, Any],
    events_df: pd.DataFrame,
    swaps_df: pd.DataFrame,
    pool_state_df: pd.DataFrame,
    gas_reference_state_df: pd.DataFrame,
    arb_labels_df: pd.DataFrame,
    qc_report: dict[str, Any],
    size_sensitivity_df: pd.DataFrame,
    opportunity_windows_df: pd.DataFrame,
    dataset_manifest: dict[str, Any],
) -> None:
    for directory_key in ["raw_dir", "curated_dir", "report_dir", "metadata_dir"]:
        paths[directory_key].mkdir(parents=True, exist_ok=True)

    logs_df.to_parquet(paths["raw_logs"], index=False)
    _write_preview_csv(paths["raw_logs_preview"], logs_df, RAW_PREVIEW_COLUMNS["event_logs"])
    blocks_df.to_parquet(paths["raw_blocks"], index=False)
    _write_preview_csv(paths["raw_blocks_preview"], blocks_df, RAW_PREVIEW_COLUMNS["block_headers"])
    pair_metadata_df.to_parquet(paths["raw_pair_metadata"], index=False)
    _write_preview_csv(paths["raw_pair_metadata_preview"], pair_metadata_df, RAW_PREVIEW_COLUMNS["pair_metadata"])
    _write_json(paths["raw_manifest"], raw_manifest)

    events_df.to_parquet(paths["events_curated"], index=False)
    _write_preview_csv(paths["events_curated_preview"], events_df, CURATED_PREVIEW_COLUMNS["events_curated"])
    swaps_df.to_parquet(paths["swaps_raw"], index=False)
    _write_preview_csv(paths["swaps_raw_preview"], swaps_df, CURATED_PREVIEW_COLUMNS["swaps_raw"])
    pool_state_df.to_parquet(paths["pool_state"], index=False)
    _write_preview_csv(paths["pool_state_preview"], pool_state_df, CURATED_PREVIEW_COLUMNS["pool_state_1s"])
    if not gas_reference_state_df.empty:
        gas_reference_state_df.to_parquet(paths["gas_reference_state"], index=False)
        _write_preview_csv(
            paths["gas_reference_state_preview"],
            gas_reference_state_df,
            CURATED_PREVIEW_COLUMNS["gas_reference_state_1s"],
        )
    arb_labels_df.to_parquet(paths["arb_labels"], index=False)
    _write_preview_csv(paths["arb_labels_preview"], arb_labels_df, CURATED_PREVIEW_COLUMNS["arb_labels_1s"])
    _write_json(paths["qc_report"], qc_report)

    size_sensitivity_df.to_csv(paths["size_sensitivity"], index=False)
    opportunity_windows_df.to_csv(paths["opportunity_windows"], index=False)
    _write_json(paths["dataset_manifest"], dataset_manifest)


def _load_supported_outputs(paths: dict[str, Path], spec: MarketSpec) -> dict[str, Any]:
    gas_reference_state = pd.read_parquet(paths["gas_reference_state"]) if paths["gas_reference_state"].exists() else pd.DataFrame()
    return {
        "status": "supported",
        "market_spec": spec,
        "blockers": [],
        "raw_manifest": _read_json(paths["raw_manifest"]),
        "manifest": _read_json(paths["dataset_manifest"]),
        "pair_metadata": pd.read_parquet(paths["raw_pair_metadata"]),
        "events_curated": pd.read_parquet(paths["events_curated"]),
        "swaps_raw": pd.read_parquet(paths["swaps_raw"]),
        "pool_state": pd.read_parquet(paths["pool_state"]),
        "gas_reference_state": gas_reference_state,
        "arb_labels": pd.read_parquet(paths["arb_labels"]),
        "size_sensitivity": pd.read_csv(paths["size_sensitivity"]),
        "opportunity_windows": pd.read_csv(paths["opportunity_windows"], parse_dates=["start_timestamp", "end_timestamp"]) if paths["opportunity_windows"].exists() else pd.DataFrame(),
        "qc_report": _read_json(paths["qc_report"]),
    }


def _write_blocked_manifest(paths: dict[str, Path], spec: MarketSpec) -> dict[str, Any]:
    for directory_key in ["report_dir", "metadata_dir"]:
        paths[directory_key].mkdir(parents=True, exist_ok=True)
    manifest = _market_manifest(
        spec,
        status="blocked",
        extra={
            "generated_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    _write_json(paths["dataset_manifest"], manifest)
    return manifest


def _supported_files_exist(paths: dict[str, Path]) -> bool:
    required = [
        paths["raw_logs"],
        paths["raw_blocks"],
        paths["raw_pair_metadata"],
        paths["raw_manifest"],
        paths["events_curated"],
        paths["swaps_raw"],
        paths["pool_state"],
        paths["arb_labels"],
        paths["qc_report"],
        paths["size_sensitivity"],
        paths["opportunity_windows"],
        paths["dataset_manifest"],
    ]
    return all(path.exists() for path in required)


def _dataset_key(slug: str) -> str:
    for key, spec in SECOND_LEVEL_MARKETS.items():
        if spec.slug == slug:
            return key
    raise KeyError(slug)


def load_or_build_second_level_dataset(
    slug: str,
    *,
    project_root: str | Path | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    project = find_project_root(Path(project_root) if project_root is not None else Path.cwd())
    spec = SECOND_LEVEL_MARKETS[_dataset_key(slug)] if slug not in SECOND_LEVEL_MARKETS else SECOND_LEVEL_MARKETS[slug]
    paths = market_paths(project, spec.slug)

    if not spec.supported:
        manifest = _write_blocked_manifest(paths, spec)
        return {
            "status": "blocked",
            "market_spec": spec,
            "blockers": list(spec.blockers),
            "manifest": manifest,
            "pair_metadata": pd.DataFrame(),
            "events_curated": pd.DataFrame(),
            "swaps_raw": pd.DataFrame(),
            "pool_state": pd.DataFrame(),
            "gas_reference_state": pd.DataFrame(),
            "arb_labels": pd.DataFrame(),
            "size_sensitivity": pd.DataFrame(),
            "opportunity_windows": pd.DataFrame(),
            "qc_report": {},
        }

    if not refresh and _supported_files_exist(paths):
        return _load_supported_outputs(paths, spec)

    rpc_urls, rpc_env_var, rpc_source = select_rpc_urls(spec.chain_key)
    rpc = RpcClient(rpc_urls, paths["rpc_cache_dir"])
    start_block, end_block, window_start, window_end = resolve_block_window(spec, rpc)

    pair_specs = list(spec.dexes)
    if spec.gas_reference_pair is not None:
        pair_specs.append(spec.gas_reference_pair)
    pair_metadata_df = build_pair_metadata(rpc, pair_specs)

    log_rows: list[dict[str, Any]] = []
    for pair in pair_specs:
        log_rows.extend(
            collect_pair_logs(
                rpc,
                pair,
                start_block=start_block,
                end_block=end_block,
                chunk_size=spec.block_chunk_size,
                lookback_blocks=spec.anchor_lookback_blocks,
            )
        )
    logs_df = pd.DataFrame(log_rows)
    if logs_df.empty:
        raise ValueError(f"No on-chain logs collected for {spec.slug}")
    logs_df = logs_df.sort_values(["block_number", "log_index", "dex"]).reset_index(drop=True)

    block_numbers = sorted(set(logs_df["block_number"].tolist() + [start_block, end_block]))
    blocks_df = fetch_block_headers(rpc, block_numbers)
    raw_manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "market_slug": spec.slug,
        "chain_key": spec.chain_key,
        "chain_name": spec.chain_name,
        "chain_id": CHAIN_CONNECTIONS[spec.chain_key]["chain_id"],
        "window_start_utc": window_start.isoformat(),
        "window_end_utc": window_end.isoformat(),
        "start_block": start_block,
        "end_block": end_block,
        "rpc_env_var": rpc_env_var,
        "rpc_source": rpc_source,
        "rpc_urls_used": rpc_urls,
    }

    events_df = normalize_event_logs(logs_df, blocks_df, pair_metadata_df)
    swaps_df = build_swaps_raw(events_df)
    arb_pool_state = build_pool_state(
        events_df,
        swaps_df,
        role="arb_pair",
        stale_after_minutes=spec.stale_after_minutes,
        frequency="1s",
        window_start=window_start,
        window_end=window_end,
    )
    gas_reference_state = build_pool_state(
        events_df,
        swaps_df,
        role="gas_reference",
        stale_after_minutes=spec.stale_after_minutes,
        frequency="1s",
        window_start=window_start,
        window_end=window_end,
    )
    gas_quote_series = build_gas_quote_series(
        spec,
        arb_pool_state,
        gas_reference_state,
        frequency="1s",
        window_start=window_start,
        window_end=window_end,
    )

    size_sensitivity_df, arb_labels_df = build_size_sensitivity(
        spec,
        arb_pool_state,
        blocks_df,
        pair_metadata_df,
        gas_quote_series,
        frequency="1s",
    )
    opportunity_windows_df = build_opportunity_windows(arb_labels_df)
    qc_report = run_qc(swaps_df[swaps_df["role"] == "arb_pair"], arb_pool_state, arb_labels_df, frequency="1s")

    dataset_manifest = _market_manifest(
        spec,
        status="supported",
        extra={
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "window_start_utc": window_start.isoformat(),
            "window_end_utc": window_end.isoformat(),
            "start_block": start_block,
            "end_block": end_block,
            "rpc_env_var": rpc_env_var,
            "rpc_source": rpc_source,
            "pair_metadata": pair_metadata_df.to_dict(orient="records"),
            "rows": {
                "event_logs": int(len(logs_df)),
                "events_curated": int(len(events_df)),
                "swaps_raw": int(len(swaps_df)),
                "pool_state_1s": int(len(arb_pool_state)),
                "gas_reference_state_1s": int(len(gas_reference_state)),
                "arb_labels_1s": int(len(arb_labels_df)),
            },
            "primary_trade_size_base": spec.primary_trade_size_base,
            "size_sensitivity_rows": int(len(size_sensitivity_df)),
            "positive_seconds": int(arb_labels_df["opportunity_flag"].sum()) if not arb_labels_df.empty else 0,
            "max_net_edge_bps": float(arb_labels_df["net_edge_bps"].max()) if not arb_labels_df.empty else None,
            "qc_passed": bool(qc_report.get("passed", False)),
        },
    )

    _write_supported_outputs(
        paths,
        logs_df=logs_df,
        blocks_df=blocks_df,
        pair_metadata_df=pair_metadata_df,
        raw_manifest=raw_manifest,
        events_df=events_df,
        swaps_df=swaps_df,
        pool_state_df=arb_pool_state,
        gas_reference_state_df=gas_reference_state,
        arb_labels_df=arb_labels_df,
        qc_report=qc_report,
        size_sensitivity_df=size_sensitivity_df,
        opportunity_windows_df=opportunity_windows_df,
        dataset_manifest=dataset_manifest,
    )
    return _load_supported_outputs(paths, spec)


def dataset_summary_frame(dataset: dict[str, Any]) -> pd.DataFrame:
    spec: MarketSpec = dataset["market_spec"]
    if dataset["status"] == "blocked":
        return pd.DataFrame(
            [
                {"field": "status", "value": "blocked"},
                {"field": "chain_name", "value": spec.chain_name},
                {"field": "pair_label", "value": spec.pair_label},
                {"field": "sample_window_days", "value": spec.sample_window_days},
                {"field": "state_frequency", "value": "1s"},
                {"field": "blocker_count", "value": len(spec.blockers)},
                {"field": "notes", "value": " | ".join(spec.notes)},
            ]
        )

    manifest = dataset["manifest"]
    arb_labels = dataset["arb_labels"]
    summary_rows = [
        {"field": "status", "value": dataset["status"]},
        {"field": "chain_name", "value": spec.chain_name},
        {"field": "pair_label", "value": spec.pair_label},
        {"field": "sample_window_days", "value": spec.sample_window_days},
        {"field": "state_frequency", "value": "1s"},
        {"field": "window_start_utc", "value": manifest.get("window_start_utc")},
        {"field": "window_end_utc", "value": manifest.get("window_end_utc")},
        {"field": "primary_trade_size_base", "value": spec.primary_trade_size_base},
        {"field": "event_log_rows", "value": manifest.get("rows", {}).get("event_logs", 0)},
        {"field": "pool_state_rows", "value": manifest.get("rows", {}).get("pool_state_1s", 0)},
        {"field": "arb_label_rows", "value": manifest.get("rows", {}).get("arb_labels_1s", 0)},
        {"field": "positive_seconds", "value": int(arb_labels["opportunity_flag"].sum()) if not arb_labels.empty else 0},
        {"field": "max_net_edge_bps", "value": float(arb_labels["net_edge_bps"].max()) if not arb_labels.empty else np.nan},
        {"field": "qc_passed", "value": manifest.get("qc_passed", False)},
        {"field": "notes", "value": " | ".join(spec.notes)},
    ]
    return pd.DataFrame(summary_rows)
