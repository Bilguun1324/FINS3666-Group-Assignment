"""Ethereum ABI constants used by the dataset pipeline."""

from __future__ import annotations

from web3 import Web3

GET_PAIR_SIGNATURE = "getPair(address,address)"
TOKEN0_SIGNATURE = "token0()"
TOKEN1_SIGNATURE = "token1()"

SWAP_EVENT_SIGNATURE = "Swap(address,uint256,uint256,uint256,uint256,address)"
SYNC_EVENT_SIGNATURE = "Sync(uint112,uint112)"
MINT_EVENT_SIGNATURE = "Mint(address,uint256,uint256)"
BURN_EVENT_SIGNATURE = "Burn(address,uint256,uint256,address)"


def selector(signature: str) -> str:
    return Web3.keccak(text=signature)[:4].hex()


def topic(signature: str) -> str:
    return Web3.keccak(text=signature).hex()


GET_PAIR_SELECTOR = selector(GET_PAIR_SIGNATURE)
TOKEN0_SELECTOR = selector(TOKEN0_SIGNATURE)
TOKEN1_SELECTOR = selector(TOKEN1_SIGNATURE)

SWAP_TOPIC = topic(SWAP_EVENT_SIGNATURE)
SYNC_TOPIC = topic(SYNC_EVENT_SIGNATURE)
MINT_TOPIC = topic(MINT_EVENT_SIGNATURE)
BURN_TOPIC = topic(BURN_EVENT_SIGNATURE)

TOPIC_TO_EVENT = {
    SWAP_TOPIC: "Swap",
    SYNC_TOPIC: "Sync",
    MINT_TOPIC: "Mint",
    BURN_TOPIC: "Burn",
}

PAIR_EVENT_TOPICS = [SWAP_TOPIC, SYNC_TOPIC, MINT_TOPIC, BURN_TOPIC]
