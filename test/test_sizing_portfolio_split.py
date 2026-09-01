from math import isclose

from core.modules.models.pipeline_types import (
    EstimatorOutput,
    PortfolioState,
    RawPairTarget,
    SignalOutput,
    SizingState,
)
from core.modules.portfolio.allocator import EqualWeightAllocator
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


def test_equal_weight_allocator_turns_weights_into_amounts():
    allocator = EqualWeightAllocator(equity=100000.0, max_open_pairs=4, total_pair_count=2)
    raw = RawPairTarget(
        pair_id="btc_coin",
        ready=True,
        side="long_x",
        x_weight=0.75,
        y_weight=0.25,
        x_price=100.0,
        y_price=50.0,
        hedge_ratio=3.0,
        para={
            "portfolio": {
                "entry_gross_cap": 25000.0,
                "remaining_pair_gross": 50000.0,
            },
            "pair": {
                "x_symbol": "BTCUSDT",
                "y_symbol": "COINUSDT",
            },
        },
    )

    allocation = allocator.allocate(PortfolioState(), {"btc_coin": raw}, bar_index=1)
    target = allocation.pair_targets["btc_coin"]

    assert isclose(target.final_x_notional, 18750.0)
    assert isclose(target.final_y_notional, 6250.0)
    assert isclose(target.final_x_quantity, 187.5)
    assert isclose(target.final_y_quantity, 125.0)
    assert isclose(allocation.portfolio_gross_exposure, 25000.0)


def test_equal_weight_allocator_caps_symbol_by_max_open_pairs():
    allocator = EqualWeightAllocator(equity=100000.0, max_open_pairs=20, total_pair_count=100)
    state = PortfolioState(symbol_exposure_map={"BTCUSDT": 3000.0})
    targets = {
        "btc_coin": RawPairTarget(
            pair_id="btc_coin",
            ready=True,
            side="long_x",
            x_weight=0.5,
            y_weight=0.5,
            x_price=100.0,
            y_price=50.0,
            para={
                "portfolio": {
                    "entry_gross_cap": 5000.0,
                    "remaining_pair_gross": 5000.0,
                },
                "pair": {
                    "x_symbol": "BTCUSDT",
                    "y_symbol": "COINUSDT",
                },
            },
        ),
        "btc_mstr": RawPairTarget(
            pair_id="btc_mstr",
            ready=True,
            side="long_x",
            x_weight=0.5,
            y_weight=0.5,
            x_price=100.0,
            y_price=200.0,
            para={
                "portfolio": {
                    "entry_gross_cap": 5000.0,
                    "remaining_pair_gross": 5000.0,
                },
                "pair": {
                    "x_symbol": "BTCUSDT",
                    "y_symbol": "MSTRUSDT",
                },
            },
        ),
    }

    allocation = allocator.allocate(state, targets, bar_index=1)

    btc_total = (
        3000.0
        + sum(target.final_x_notional for target in allocation.pair_targets.values())
    )
    assert btc_total <= 5000.0 + 1e-9
    assert isclose(allocation.symbol_exposure_map["BTCUSDT"], 5000.0)


def test_score_weighted_allocator_prioritizes_high_score_candidates():
    allocator = EqualWeightAllocator(
        equity=100000.0,
        max_open_pairs=2,
        allocation_mode="score_weighted",
    )
    targets = {
        "low": RawPairTarget(
            pair_id="low",
            ready=True,
            side="long_x",
            x_weight=0.5,
            y_weight=0.5,
            x_price=100.0,
            y_price=100.0,
            signal_strength=1.0,
            para={"portfolio": {"entry_gross_cap": 50000.0, "remaining_pair_gross": 50000.0}},
        ),
        "high": RawPairTarget(
            pair_id="high",
            ready=True,
            side="long_x",
            x_weight=0.5,
            y_weight=0.5,
            x_price=100.0,
            y_price=100.0,
            signal_strength=10.0,
            para={"portfolio": {"entry_gross_cap": 50000.0, "remaining_pair_gross": 50000.0}},
        ),
        "mid": RawPairTarget(
            pair_id="mid",
            ready=True,
            side="long_x",
            x_weight=0.5,
            y_weight=0.5,
            x_price=100.0,
            y_price=100.0,
            signal_strength=5.0,
            para={"portfolio": {"entry_gross_cap": 50000.0, "remaining_pair_gross": 50000.0}},
        ),
    }

    allocation = allocator.allocate(PortfolioState(), targets, bar_index=1)

    assert allocation.selected_pair_ids == ["high", "mid"]
    assert "low" not in allocation.pair_targets


def test_score_weighted_allocator_scales_by_score_without_exceeding_pair_cap():
    allocator = EqualWeightAllocator(
        equity=100000.0,
        max_open_pairs=2,
        allocation_mode="score_weighted",
    )
    targets = {
        "strong": RawPairTarget(
            pair_id="strong",
            ready=True,
            side="long_x",
            x_weight=0.5,
            y_weight=0.5,
            x_price=100.0,
            y_price=100.0,
            signal_strength=3.0,
            para={"portfolio": {"entry_gross_cap": 50000.0, "remaining_pair_gross": 50000.0}},
        ),
        "weak": RawPairTarget(
            pair_id="weak",
            ready=True,
            side="long_x",
            x_weight=0.5,
            y_weight=0.5,
            x_price=100.0,
            y_price=100.0,
            signal_strength=1.0,
            para={"portfolio": {"entry_gross_cap": 50000.0, "remaining_pair_gross": 50000.0}},
        ),
    }

    allocation = allocator.allocate(PortfolioState(), targets, bar_index=1)
    strong = allocation.pair_targets["strong"]
    weak = allocation.pair_targets["weak"]

    strong_gross = strong.final_x_notional + strong.final_y_notional
    weak_gross = weak.final_x_notional + weak.final_y_notional
    assert isclose(strong_gross, 50000.0)
    assert isclose(weak_gross, 25000.0)
