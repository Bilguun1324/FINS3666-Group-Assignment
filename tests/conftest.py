from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from mev_dataset.config import MarketConfig
from mev_dataset.constants import BURN_TOPIC, GET_PAIR_SELECTOR, MINT_TOPIC, SWAP_TOPIC, SYNC_TOPIC, TOKEN0_SELECTOR, TOKEN1_SELECTOR
from mev_dataset.rpc import BlockHeader


WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
UNISWAP_FACTORY = "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f"
SUSHI_FACTORY = "0xc0aee478e3658e2610c5f7a4a2e1777ce9e4f2ac"
UNISWAP_PAIR = "0x1111111111111111111111111111111111111111"
SUSHI_PAIR = "0x2222222222222222222222222222222222222222"


def encode_address_result(address: str) -> str:
    return "0x" + ("0" * 24) + address.lower().replace("0x", "")


def encode_words(*values: int) -> str:
    return "0x" + "".join(f"{value:064x}" for value in values)


def make_log(block_number: int, tx_hash: str, log_index: int, address: str, topic0: str, data: str) -> dict[str, str]:
    return {
        "address": address,
        "blockNumber": hex(block_number),
        "transactionHash": tx_hash,
        "logIndex": hex(log_index),
        "topics": [topic0],
        "data": data,
        "removed": False,
    }


class FakeRpc:
    def __init__(self, logs_by_address: dict[str, list[dict]], block_headers: dict[int, BlockHeader]) -> None:
        self.logs_by_address = logs_by_address
        self.block_headers = block_headers

    def eth_call(self, to: str, data: str, block: str = "latest", use_cache: bool = True) -> str:
        selector = data[:10]
        lookup = {
            (UNISWAP_FACTORY, f"0x{GET_PAIR_SELECTOR}"): encode_address_result(UNISWAP_PAIR),
            (SUSHI_FACTORY, f"0x{GET_PAIR_SELECTOR}"): encode_address_result(SUSHI_PAIR),
            (UNISWAP_PAIR, f"0x{TOKEN0_SELECTOR}"): encode_address_result(WBTC),
            (UNISWAP_PAIR, f"0x{TOKEN1_SELECTOR}"): encode_address_result(WETH),
            (SUSHI_PAIR, f"0x{TOKEN0_SELECTOR}"): encode_address_result(WETH),
            (SUSHI_PAIR, f"0x{TOKEN1_SELECTOR}"): encode_address_result(WBTC),
        }
        return lookup[(to.lower(), selector)]

    def eth_get_logs(self, address: str, from_block: int, to_block: int, topics=None, use_cache: bool = True):
        rows = []
        for log in self.logs_by_address[address.lower()]:
            block_number = int(log["blockNumber"], 16)
            if from_block <= block_number <= to_block:
                rows.append(log)
        return rows

    def get_block_by_number(self, block_number: int, use_cache: bool = True) -> BlockHeader:
        return self.block_headers[block_number]

    def get_latest_block_number(self) -> int:
        return max(self.block_headers)

    def find_block_by_timestamp(self, target: datetime, direction: str = "after") -> int:
        ordered = sorted(self.block_headers.values(), key=lambda item: item.block_number)
        if direction == "after":
            for header in ordered:
                if header.block_timestamp >= target:
                    return header.block_number
            return ordered[-1].block_number
        for header in reversed(ordered):
            if header.block_timestamp <= target:
                return header.block_number
        return ordered[0].block_number


@pytest.fixture()
def sample_config(tmp_path: Path) -> MarketConfig:
    return MarketConfig.model_validate(
        {
            "market": "wbtc_weth_mainnet_test",
            "chain_id": 1,
            "sample_window_days": 1,
            "start_date": datetime(2026, 1, 1, tzinfo=UTC),
            "end_date": datetime(2026, 1, 2, tzinfo=UTC),
            "block_chunk_size": 10,
            "stale_after_minutes": 15,
            "arbitrage_notional_wbtc": 0.10,
            "raw_data_dir": str(tmp_path / "raw"),
            "curated_data_dir": str(tmp_path / "curated"),
            "report_dir": str(tmp_path / "report"),
            "metadata_dir": str(tmp_path / "metadata"),
            "tokens": {
                "wbtc": {"symbol": "WBTC", "address": WBTC, "decimals": 8},
                "weth": {"symbol": "WETH", "address": WETH, "decimals": 18},
            },
            "dexes": [
                {"name": "uniswap_v2", "factory_address": UNISWAP_FACTORY, "fee_bps": 30},
                {"name": "sushiswap_v2", "factory_address": SUSHI_FACTORY, "fee_bps": 30},
            ],
        }
    )


@pytest.fixture()
def sample_rpc() -> FakeRpc:
    block_headers = {
        100: BlockHeader(
            block_number=100,
            block_hash="0xabc100",
            parent_hash="0xabc099",
            block_timestamp=datetime(2026, 1, 1, 0, 0, 10, tzinfo=UTC),
            base_fee_per_gas_wei=int(50e9),
            gas_used=15_000_000,
            gas_limit=30_000_000,
        ),
        101: BlockHeader(
            block_number=101,
            block_hash="0xabc101",
            parent_hash="0xabc100",
            block_timestamp=datetime(2026, 1, 1, 0, 2, 5, tzinfo=UTC),
            base_fee_per_gas_wei=int(55e9),
            gas_used=15_200_000,
            gas_limit=30_000_000,
        ),
    }

    logs_by_address = {
        UNISWAP_PAIR: [
            make_log(100, "0xtxuni100", 0, UNISWAP_PAIR, SYNC_TOPIC, encode_words(int(99.5 * 1e8), int(3015.1 * 1e18))),
            make_log(100, "0xtxuni100", 1, UNISWAP_PAIR, SWAP_TOPIC, encode_words(0, int(15.1 * 1e18), int(0.5 * 1e8), 0)),
            make_log(101, "0xtxuni101", 0, UNISWAP_PAIR, SYNC_TOPIC, encode_words(int(100.0 * 1e8), int(3030.0 * 1e18))),
            make_log(101, "0xtxuni101", 1, UNISWAP_PAIR, SWAP_TOPIC, encode_words(int(0.5 * 1e8), 0, 0, int(15.0 * 1e18))),
        ],
        SUSHI_PAIR: [
            make_log(100, "0xtxsushi100", 0, SUSHI_PAIR, SYNC_TOPIC, encode_words(int(3090.0 * 1e18), int(99.6 * 1e8))),
            make_log(100, "0xtxsushi100", 1, SUSHI_PAIR, SWAP_TOPIC, encode_words(0, int(0.4 * 1e8), int(12.2 * 1e18), 0)),
            make_log(101, "0xtxsushi101", 0, SUSHI_PAIR, SYNC_TOPIC, encode_words(int(3075.0 * 1e18), int(99.8 * 1e8))),
            make_log(101, "0xtxsushi101", 1, SUSHI_PAIR, SWAP_TOPIC, encode_words(int(18.0 * 1e18), 0, 0, int(0.58 * 1e8))),
        ],
    }
    return FakeRpc(logs_by_address=logs_by_address, block_headers=block_headers)
