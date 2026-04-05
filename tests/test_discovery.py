from conftest import SUSHI_FACTORY, SUSHI_PAIR, UNISWAP_FACTORY, UNISWAP_PAIR, WBTC, WETH
from mev_dataset.discovery import discover_pairs, get_pair_address, get_token_order


def test_factory_pair_discovery(sample_config, sample_rpc):
    assert get_pair_address(sample_rpc, UNISWAP_FACTORY, WBTC, WETH) == UNISWAP_PAIR
    assert get_pair_address(sample_rpc, SUSHI_FACTORY, WBTC, WETH) == SUSHI_PAIR
    assert get_token_order(sample_rpc, SUSHI_PAIR) == (WETH, WBTC)

    pairs = discover_pairs(sample_config, sample_rpc)
    assert [pair.dex for pair in pairs] == ["uniswap_v2", "sushiswap_v2"]
    assert pairs[0].wbtc_is_token0 is True
    assert pairs[1].wbtc_is_token0 is False
