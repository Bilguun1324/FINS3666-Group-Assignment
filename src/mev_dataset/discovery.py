"""Pair discovery and token ordering for Uniswap V2-style pools."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from pydantic import BaseModel, ConfigDict

from mev_dataset.config import DexConfig, MarketConfig
from mev_dataset.constants import GET_PAIR_SELECTOR, TOKEN0_SELECTOR, TOKEN1_SELECTOR
from mev_dataset.rpc import RpcClient


class PairMetadata(BaseModel):
    dex: str
    factory_address: str
    pair_address: str
    token0: str
    token1: str
    wbtc_is_token0: bool
    fee_rate: float

    model_config = ConfigDict(frozen=True)


def _encode_address_arg(address: str) -> str:
    return address.lower().replace("0x", "").rjust(64, "0")


def _decode_address_output(output: str) -> str:
    return f"0x{output[-40:]}".lower()


def get_pair_address(rpc: RpcClient, factory_address: str, token_a: str, token_b: str) -> str:
    calldata = f"0x{GET_PAIR_SELECTOR}{_encode_address_arg(token_a)}{_encode_address_arg(token_b)}"
    response = rpc.eth_call(factory_address, calldata)
    pair_address = _decode_address_output(response)
    if pair_address == "0x0000000000000000000000000000000000000000":
        raise ValueError(f"Factory {factory_address} returned the zero address for the requested pair")
    return pair_address


def get_token_order(rpc: RpcClient, pair_address: str) -> tuple[str, str]:
    token0 = _decode_address_output(rpc.eth_call(pair_address, f"0x{TOKEN0_SELECTOR}"))
    token1 = _decode_address_output(rpc.eth_call(pair_address, f"0x{TOKEN1_SELECTOR}"))
    return token0, token1


def discover_pairs(config: MarketConfig, rpc: RpcClient) -> list[PairMetadata]:
    wbtc = config.tokens["wbtc"].address
    weth = config.tokens["weth"].address
    pairs: list[PairMetadata] = []
    for dex in config.dexes:
        pair_address = dex.pair_address or get_pair_address(rpc, dex.factory_address, wbtc, weth)
        token0, token1 = get_token_order(rpc, pair_address)
        pairs.append(
            PairMetadata(
                dex=dex.name,
                factory_address=dex.factory_address,
                pair_address=pair_address,
                token0=token0,
                token1=token1,
                wbtc_is_token0=(token0 == wbtc),
                fee_rate=dex.fee_rate,
            )
        )
    return pairs


def pair_metadata_frame(pairs: Iterable[PairMetadata]) -> pd.DataFrame:
    return pd.DataFrame([pair.model_dump() for pair in pairs])
