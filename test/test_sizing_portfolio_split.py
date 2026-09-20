from math import isclose

from core.modules.models.pipeline_types import (
    EstimatorOutput,
    PortfolioState,
    RawPairTarget,
    SignalOutput,
    SizingState,
)
from core.modules.sizing.pair_sizing import BetaNeutralSizing


def test_beta_neutral_sizing_outputs_weights_not_amounts():
    sizing = BetaNeutralSizing(hedge_beta_lookback_bars=10, hedge_beta_min_samples=30)
    signal = SignalOutput(
        pair_id="btc_coin",
        ready=True,
        action="open",
        side="long_x",
        signal_strength=2.0,
    )
    estimator = EstimatorOutput(pair_id="btc_coin", ready=True, hedge_beta=2.0)

    target = sizing.compute(SizingState(), signal, estimator, x_price=100.0, y_price=50.0)

    assert target.ready
    assert isclose(target.x_weight, 2.0 / 3.0)
    assert isclose(target.y_weight, 1.0 / 3.0)
    assert target.x_notional == 0.0
    assert target.y_notional == 0.0
    assert target.x_quantity == 0.0
    assert target.y_quantity == 0.0
    assert isclose(target.hedge_ratio, 2.0)




def test_pair_target_capital_uses_available_margin_and_preserves_leg_weights():
    from core.modules.portfolio.allocator import PairTargetCapitalAllocator
    allocator = PairTargetCapitalAllocator(equity=3_000_000, minimum_entry_capital_ratio=0.5)
    raw = RawPairTarget(
        pair_id="btc_coin", ready=True, side="long_x", x_weight=0.6, y_weight=0.4,
        x_price=100.0, y_price=50.0, gross_notional=600_000.0, target_capital=600_000.0,
    )
    state = PortfolioState(ready=True, equity=3_000_000, available_balance=350_000)
    target = allocator.allocate(state, {"btc_coin": raw}, 1).pair_targets["btc_coin"]
    assert target.final_x_notional == 210_000.0
    assert target.final_y_notional == 140_000.0


def test_pair_target_capital_rejects_below_minimum_ratio_without_equity_fallback():
    from core.modules.portfolio.allocator import PairTargetCapitalAllocator
    allocator = PairTargetCapitalAllocator(equity=3_000_000, minimum_entry_capital_ratio=0.5)
    raw = RawPairTarget(
        pair_id="btc_coin", ready=True, side="long_x", x_weight=0.5, y_weight=0.5,
        x_price=100.0, y_price=50.0, gross_notional=600_000.0, target_capital=600_000.0,
    )
    state = PortfolioState(ready=True, equity=3_000_000, available_balance=250_000)
    allocation = allocator.allocate(state, {"btc_coin": raw}, 1)
    assert allocation.pair_targets == {}
