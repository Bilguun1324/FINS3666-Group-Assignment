import math

from mev_dataset.config import GasConfig
from mev_dataset.features import amount_in_for_exact_out, amount_out_for_exact_in, gas_cost_bps, gas_cost_weth


def test_constant_product_math_and_gas_costs():
    reserve_btc = 100.0
    reserve_eth = 3_000.0
    out_eth = amount_out_for_exact_in(0.10, reserve_btc, reserve_eth, fee_rate=0.003)
    in_eth = amount_in_for_exact_out(0.10, reserve_eth, reserve_btc, fee_rate=0.003)

    assert out_eth > 0
    assert in_eth > 0
    assert out_eth < reserve_eth
    assert in_eth > reserve_eth * 0.10 / reserve_btc

    gas = GasConfig(gas_units=220000, priority_fee_gwei=2.0)
    cost = gas_cost_weth(50e9, gas)
    assert math.isclose(cost, 0.01144, rel_tol=1e-6)
    assert gas_cost_bps(cost, 10.0) > 0
