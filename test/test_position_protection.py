from types import SimpleNamespace
from dataclasses import replace

import pytest

from core.modules.models.pipeline_types import BarSnapshot, PairBarBundle
from core.modules.strategy.config import (
    EstimatorConfig,
    PairDefinition,
    ProtectionConfig,
)
from core.modules.strategy.multi_pair_strategy import MultiPairStrategy
from core.modules.strategy.factory import build_strategy
from core.modules.strategy.position_protection import (
    mark_net_pnl,
    solve_zero_net_x_price,
    theoretical_y_price,
)
from core.modules.estimator.tls import TLSEstimator


def test_price_zero_net_boundary_includes_costs_and_long_x_direction():
    no_cost = solve_zero_net_x_price(
        "price", 0.0, 0.5, 0.0, 1.0, 0.0, "long_x",
        10.0, 10.0, 1.0, 0.5, 0.0, 0.0, 0.0,
    )
    with_cost = solve_zero_net_x_price(
        "price", 0.0, 0.5, 0.0, 1.0, 0.0, "long_x",
        10.0, 10.0, 1.0, 0.5, 0.2, 0.01, 0.01,
    )
    assert no_cost.x_price == pytest.approx(6.6666667, abs=1e-5)
    assert with_cost.x_price is not None
    assert with_cost.x_price > no_cost.x_price
    assert with_cost.x_price < 10.0


def test_price_zero_net_boundary_uses_opposite_direction_for_short_x():
    boundary = solve_zero_net_x_price(
        "price", 0.0, 0.5, 0.0, 1.0, 0.0, "short_x",
        10.0, 2.0, 1.0, 1.0, 0.1, 0.0, 0.0,
    )
    assert boundary.x_price is not None
    assert boundary.x_price > 10.0


def test_log_price_theoretical_y_price_is_positive():
    assert theoretical_y_price("log_price", 0.0, 1.0, 10.0, 0.0) == pytest.approx(10.0)


def test_mark_net_pnl_deducts_entry_funding_and_estimated_close_cost():
    gross = mark_net_pnl("long_x", 11.0, 9.0, 10.0, 10.0, 1.0, 1.0, 0, 0, 0, 0)
    net = mark_net_pnl("long_x", 11.0, 9.0, 10.0, 10.0, 1.0, 1.0, 0.1, 0.2, 0.01, 0.01)
    assert gross == pytest.approx(2.0)
    assert net < gross


def _strategy_with_protection():
    strategy = MultiPairStrategy(
        pairs=[PairDefinition(pair_id="p", x_symbol="X", y_symbol="Y")],
        estimator_ctor=TLSEstimator,
        estimator_cfg=EstimatorConfig(
            regression_method="price", position_update_policy="freeze"
        ),
        protection_cfg=ProtectionConfig(
            enabled=True,
            take_profit_return=0.03,
            pair_loss_stop_freeze_model_lookback_multiplier=1.0,
        ),
    )
    pipeline = strategy.pipelines["p"]
    pipeline.state.sizing_state.position_side = "long_x"
    pipeline.state.sizing_state.x_quantity = 1.0
    pipeline.state.sizing_state.y_quantity = 1.0
    ps = pipeline.state.protection_state
    ps.active = True
    ps.side = "long_x"
    ps.entry_x_price = 10.0
    ps.entry_y_price = 10.0
    ps.entry_gross_notional = 20.0
    ps.take_profit_return = 0.03
    ps.stop_loss_x_price = 9.5
    return strategy, pipeline


def test_stop_loss_has_priority_over_take_profit():
    strategy, pipeline = _strategy_with_protection()
    bundle = PairBarBundle(
        "p", 1, 1,
        BarSnapshot(1, 9.4, 9.4, 9.4, 9.4, 1, "binance", "X"),
        BarSnapshot(1, 10.0, 10.0, 10.0, 10.0, 1, "binance", "Y"),
    )
    decision = strategy._protection_exit_decision(pipeline, bundle)
    assert decision == {"reason": "protective_stop_loss", "trigger": "stop_loss"}
    assert pipeline.state.protection_state.pending_exit is True
    assert strategy._protection_exit_decision(pipeline, bundle) is None


def test_take_profit_does_not_double_count_entry_slippage_in_fill_basis():
    strategy, pipeline = _strategy_with_protection()
    protection = pipeline.state.protection_state
    protection.entry_fee = 0.0
    # The entry prices in protection state are fill prices already containing
    # this execution slippage. It must not be subtracted again.
    protection.entry_slippage = 1.0
    bundle = PairBarBundle(
        "p", 1, 1,
        BarSnapshot(1, 11.0, 11.0, 11.0, 11.0, 1, "binance", "X"),
        BarSnapshot(1, 9.0, 9.0, 9.0, 9.0, 1, "binance", "Y"),
    )
    decision = strategy._protection_exit_decision(pipeline, bundle)
    assert decision == {"reason": "protective_take_profit", "trigger": "take_profit"}


def test_protective_close_reopen_lock_requires_a_later_model_update():
    strategy, pipeline = _strategy_with_protection()
    protection = pipeline.state.protection_state
    protection.reopen_lock_pending = True
    protection.close_bar_index = 10
    pipeline.state.estimator_state.last_model_update_index = 10
    strategy._maybe_release_protection_lock(pipeline)
    assert protection.reopen_lock_pending is True
    pipeline.state.estimator_state.last_model_update_index = 11
    strategy._maybe_release_protection_lock(pipeline)
    assert protection.reopen_lock_pending is False


def test_funding_payment_is_allocated_once_by_shared_symbol_quantity():
    strategy, pipeline = _strategy_with_protection()
    pipeline.state.protection_state.entry_x_quantity = 1.0

    second = MultiPairStrategy(
        pairs=[PairDefinition("p2", x_symbol="X", y_symbol="Z")],
        estimator_ctor=TLSEstimator,
        estimator_cfg=EstimatorConfig(regression_method="price", position_update_policy="freeze"),
        protection_cfg=ProtectionConfig(enabled=True),
    )
    second_state = second.pipelines["p2"].state.protection_state
    second_state.active = True
    second_state.entry_x_quantity = 1.0

    class Payment:
        symbol = "X"
        payment = 2.0

    # The two strategies are independent in production; verify the shared
    # symbol allocation rule directly on one strategy with two active pairs.
    strategy.pipelines["p2"] = second.pipelines["p2"]
    strategy.on_funding_payments([Payment()])
    assert pipeline.state.protection_state.funding_cost == pytest.approx(1.0)
    assert second_state.funding_cost == pytest.approx(1.0)


def test_position_attributed_funding_is_not_redistributed_to_shared_pairs():
    strategy, pipeline = _strategy_with_protection()
    pipeline.state.protection_state.position_id = "position-p"
    second = MultiPairStrategy(
        pairs=[PairDefinition("p2", x_symbol="X", y_symbol="Z")],
        estimator_ctor=TLSEstimator,
        estimator_cfg=EstimatorConfig(regression_method="price", position_update_policy="freeze"),
        protection_cfg=ProtectionConfig(enabled=True),
    )
    second_state = second.pipelines["p2"].state.protection_state
    second_state.active = True
    second_state.position_id = "position-p2"
    strategy.pipelines["p2"] = second.pipelines["p2"]

    payment = SimpleNamespace(
        symbol="X", payment=2.0, pair_id="p", position_id="position-p"
    )
    strategy.on_funding_payments([payment])

    assert pipeline.state.protection_state.funding_cost == pytest.approx(2.0)
    assert second_state.funding_cost == pytest.approx(0.0)


def test_factory_rejects_protection_for_update_or_return_models():
    setup = SimpleNamespace(
        pairs=[{"pair_id": "p", "x_symbol": "X", "y_symbol": "Y"}],
        pipeline={
            "estimator": {
                "method": "tls", "regression_method": "log_price",
                "position_update_policy": "update",
            },
            "signal": {"method": "zscore"},
            "protection": {"enabled": True},
        },
    )
    with pytest.raises(ValueError, match="freeze Price/Log-Price"):
        build_strategy(setup, {"X": {}, "Y": {}})


def test_protection_rejects_negative_take_profit_threshold():
    with pytest.raises(ValueError, match="finite non-negative"):
        ProtectionConfig(take_profit_return=-0.01)


def test_protection_rejects_non_positive_max_holding_multiplier():
    with pytest.raises(ValueError, match="finite positive"):
        ProtectionConfig(max_holding_time_model_lookback_multiplier=0.0)


def test_max_holding_multiplier_is_ceil_of_effective_model_lookback():
    strategy = MultiPairStrategy(
        pairs=[PairDefinition(pair_id="p", x_symbol="X", y_symbol="Y")],
        estimator_ctor=TLSEstimator,
        estimator_cfg=EstimatorConfig(
            regression_method="log_price",
            position_update_policy="freeze",
            model_lookback_bars=2880,
        ),
        protection_cfg=ProtectionConfig(
            enabled=True,
            max_holding_time_enabled=True,
            max_holding_time_model_lookback_multiplier=2.0,
            wait_for_model_update_after_non_z_exit=False,
        ),
    )
    pipeline = strategy.pipelines["p"]
    assert strategy._max_holding_bars_for_pipeline(pipeline) == 5760


def test_max_holding_switch_can_run_without_price_stop_take_profit_master_switch():
    strategy = MultiPairStrategy(
        pairs=[PairDefinition(pair_id="p", x_symbol="X", y_symbol="Y")],
        estimator_ctor=TLSEstimator,
        estimator_cfg=EstimatorConfig(regression_method="price", position_update_policy="freeze"),
        protection_cfg=ProtectionConfig(
            enabled=False,
            max_holding_time_enabled=True,
            wait_for_model_update_after_non_z_exit=False,
        ),
    )
    assert strategy.protection_supported is True


def test_max_holding_deadline_uses_first_fill_and_add_does_not_reset_it():
    strategy = MultiPairStrategy(
        pairs=[PairDefinition(pair_id="p", x_symbol="X", y_symbol="Y")],
        estimator_ctor=TLSEstimator,
        estimator_cfg=EstimatorConfig(
            regression_method="price",
            position_update_policy="freeze",
            model_lookback_bars=10,
        ),
        protection_cfg=ProtectionConfig(
            enabled=True,
            max_holding_time_enabled=True,
            max_holding_time_model_lookback_multiplier=2.0,
            wait_for_model_update_after_non_z_exit=False,
        ),
    )
    pipeline = strategy.pipelines["p"]
    pipeline.state.last_bar_index = 100
    first = [
        SimpleNamespace(symbol="X", quantity=1.0, price=10.0, fee=0.0, slippage=0.0, ts=1, para={}),
        SimpleNamespace(symbol="Y", quantity=1.0, price=10.0, fee=0.0, slippage=0.0, ts=1, para={}),
    ]
    strategy._initialize_or_update_protection(pipeline, first, "long_x")
    assert pipeline.state.protection_state.entry_bar_index == 101
    assert pipeline.state.protection_state.max_holding_bars == 20
    assert pipeline.state.protection_state.max_holding_deadline_bar == 121

    pipeline.state.last_bar_index = 130
    add = [
        SimpleNamespace(symbol="X", quantity=1.0, price=10.0, fee=0.0, slippage=0.0, ts=2, para={}),
        SimpleNamespace(symbol="Y", quantity=1.0, price=10.0, fee=0.0, slippage=0.0, ts=2, para={}),
    ]
    strategy._initialize_or_update_protection(pipeline, add, "long_x")
    assert pipeline.state.protection_state.entry_bar_index == 101
    assert pipeline.state.protection_state.max_holding_deadline_bar == 121


def test_max_holding_exits_without_a_zero_net_stop_root():
    strategy = MultiPairStrategy(
        pairs=[PairDefinition(pair_id="p", x_symbol="X", y_symbol="Y")],
        estimator_ctor=TLSEstimator,
        estimator_cfg=EstimatorConfig(regression_method="price", position_update_policy="freeze"),
        protection_cfg=ProtectionConfig(
            enabled=True,
            max_holding_time_enabled=True,
            max_holding_time_model_lookback_multiplier=2.0,
            wait_for_model_update_after_non_z_exit=True,
        ),
    )
    pipeline = strategy.pipelines["p"]
    pipeline.state.sizing_state.position_side = "long_x"
    pipeline.state.sizing_state.x_quantity = 1.0
    pipeline.state.sizing_state.y_quantity = 1.0
    state = pipeline.state.protection_state
    state.active = True
    state.side = "long_x"
    state.max_holding_deadline_bar = 5
    state.stop_loss_x_price = None
    pipeline.state.last_bar_index = 5
    bundle = PairBarBundle(
        "p", 1, 5,
        BarSnapshot(1, 10.0, 10.0, 10.0, 10.0, 1, "binance", "X"),
        BarSnapshot(1, 10.0, 10.0, 10.0, 10.0, 1, "binance", "Y"),
    )
    assert strategy._protection_exit_decision(pipeline, bundle) == {
        "reason": "protective_max_holding_time",
        "trigger": "max_holding_time",
    }


def test_take_profit_close_does_not_lock_reentry_but_stop_loss_does():
    strategy, pipeline = _strategy_with_protection()
    sizing = pipeline.state.sizing_state
    sizing.position_side = "long_x"
    sizing.x_quantity = 1.0
    sizing.y_quantity = 1.0

    take_profit = strategy._close_orders(
        "p", pipeline, pipeline.pair_def, 1.0,
        exit_reason="protective_take_profit", protection_trigger="take_profit",
    )
    assert take_profit
    assert all(order.exit_class == "take_profit" for order in take_profit)
    assert all(order.reopen_lock_pending is False for order in take_profit)

    stop_loss = strategy._close_orders(
        "p", pipeline, pipeline.pair_def, 1.0,
        exit_reason="protective_stop_loss", protection_trigger="stop_loss",
    )
    assert stop_loss
    assert all(order.exit_class == "stop_loss" for order in stop_loss)
    assert all(order.reopen_lock_pending is True for order in stop_loss)


def test_take_profit_fill_keeps_the_pair_open_for_a_new_entry():
    strategy, pipeline = _strategy_with_protection()
    sizing = pipeline.state.sizing_state
    sizing.position_side = "long_x"
    sizing.x_quantity = 1.0
    sizing.y_quantity = 1.0
    pipeline.state.last_bar_index = 10
    group = [
        SimpleNamespace(
            pair_id="p", group_id="g", action="close", symbol="X", quantity=1.0,
            exit_class="take_profit", reopen_lock_pending=False,
            exit_reason="protective_take_profit",
        ),
        SimpleNamespace(
            pair_id="p", group_id="g", action="close", symbol="Y", quantity=1.0,
            exit_class="take_profit", reopen_lock_pending=False,
            exit_reason="protective_take_profit",
        ),
    ]

    strategy.on_trades_filled(group)

    protection = pipeline.state.protection_state
    assert protection.last_exit_class == "take_profit"
    assert protection.reopen_lock_pending is False
    assert sizing.position_side is None
    # Capacity is refreshed by the strategy on each bar; give the pair a free
    # slot so only the (absent) reopen lock could block the next entry.
    sizing.remaining_pair_gross = 10000.0
    assert strategy._entry_schedule_decision(
        pipeline, SimpleNamespace(side="long_x"), 11
    )["allowed"] is True


def test_exit_class_and_wait_switch_are_propagated_to_close_orders():
    strategy, pipeline = _strategy_with_protection()
    strategy.protection_cfg.wait_for_model_update_after_non_z_exit = False
    pipeline.state.sizing_state.position_side = "long_x"
    pipeline.state.sizing_state.x_quantity = 1.0
    pipeline.state.sizing_state.y_quantity = 1.0
    orders = strategy._close_orders(
        "p", pipeline, pipeline.pair_def, 1.0,
        exit_reason="protective_max_holding_time",
        protection_trigger="max_holding_time",
    )
    assert orders
    assert all(order.exit_class == "stop_loss" for order in orders)
    assert all(order.reopen_lock_pending is False for order in orders)

    ordinary = strategy._close_orders("p", pipeline, pipeline.pair_def, 1.0)
    assert ordinary
    assert all(order.exit_class == "zscore_reversion" for order in ordinary)
    assert all(order.reopen_lock_pending is False for order in ordinary)

    legacy = _strategy_with_protection()[0]
    legacy_pipeline = legacy.pipelines["p"]
    legacy_pipeline.state.sizing_state.position_side = "long_x"
    legacy_pipeline.state.sizing_state.x_quantity = 1.0
    legacy_pipeline.state.sizing_state.y_quantity = 1.0
    legacy_orders = legacy._close_orders(
        "p", legacy_pipeline, legacy_pipeline.pair_def, 1.0,
        exit_reason="protective_stop_loss", protection_trigger="stop_loss",
    )
    assert all(order.reopen_lock_pending is True for order in legacy_orders)


def test_close_fill_uses_exit_class_for_reopen_lock_and_actual_fill_bar():
    strategy, pipeline = _strategy_with_protection()
    strategy.protection_cfg.wait_for_model_update_after_non_z_exit = True
    sizing = pipeline.state.sizing_state
    sizing.position_side = "long_x"
    sizing.x_quantity = 1.0
    sizing.y_quantity = 1.0
    pipeline.state.last_bar_index = 10
    close_group = [
        SimpleNamespace(
            pair_id="p", group_id="g", action="close", symbol="X", quantity=1.0,
            exit_class="stop_loss", reopen_lock_pending=True, exit_reason="protective_stop_loss",
        ),
        SimpleNamespace(
            pair_id="p", group_id="g", action="close", symbol="Y", quantity=1.0,
            exit_class="stop_loss", reopen_lock_pending=True, exit_reason="protective_stop_loss",
        ),
    ]
    strategy.on_trades_filled(close_group)
    protection = pipeline.state.protection_state
    assert protection.last_exit_class == "stop_loss"
    assert protection.reopen_lock_pending is True
    assert protection.close_bar_index == 11


def test_pair_equity_zero_forced_close_freezes_one_model_lookback_and_waits_for_update():
    strategy, pipeline = _strategy_with_protection()
    sizing = pipeline.state.sizing_state
    sizing.position_side = "long_x"
    sizing.x_quantity = 1.0
    sizing.y_quantity = 1.0
    pipeline.state.last_bar_index = 10
    close_group = [
        SimpleNamespace(
            pair_id="p", group_id="forced", action="close", symbol="X", quantity=1.0,
            exit_class="stop_loss", reopen_lock_pending=True,
            exit_reason="protective_pair_equity_zero",
            protection_trigger="pair_equity_zero", protection_rule="pair_equity_zero",
            protection_freeze_bars=0,
        ),
        SimpleNamespace(
            pair_id="p", group_id="forced", action="close", symbol="Y", quantity=1.0,
            exit_class="stop_loss", reopen_lock_pending=True,
            exit_reason="protective_pair_equity_zero",
            protection_trigger="pair_equity_zero", protection_rule="pair_equity_zero",
            protection_freeze_bars=0,
        ),
    ]

    strategy.on_trades_filled(close_group)

    protection = pipeline.state.protection_state
    expected_freeze = pipeline.estimator.model_lookback_bars
    assert protection.freeze_rule == "pair_equity_zero"
    assert protection.freeze_bars == expected_freeze
    assert protection.freeze_until_bar == 11 + expected_freeze
    assert protection.reopen_lock_pending is True
    assert all(trade.protection_freeze_bars == expected_freeze for trade in close_group)
    assert all(
        trade.protection_freeze_until_bar == 11 + expected_freeze
        for trade in close_group
    )


def test_forced_margin_partial_close_keeps_pair_state_and_updates_notional():
    strategy, pipeline = _strategy_with_protection()
    sizing = pipeline.state.sizing_state
    sizing.position_side = "long_x"
    sizing.position_id = "position-p"
    sizing.x_quantity = 5.0
    sizing.y_quantity = 5.0
    sizing.pair_cap_gross = 100.0
    sizing.current_gross_notional = 100.0
    sizing.remaining_pair_gross = 0.0
    protection = pipeline.state.protection_state
    protection.entry_x_quantity = 5.0
    protection.entry_y_quantity = 5.0
    protection.entry_gross_notional = 100.0
    protection.entry_fee = 1.0
    protection.entry_slippage = 1.0
    protection.funding_cost = 2.0
    group = [
        SimpleNamespace(
            pair_id="p", group_id="margin", action="close", symbol="X",
            quantity=1.0, price=10.0, exit_class="stop_loss",
            protection_trigger="margin_deleveraging",
            exit_reason="forced_margin_deleveraging", reopen_lock_pending=True,
        ),
        SimpleNamespace(
            pair_id="p", group_id="margin", action="close", symbol="Y",
            quantity=1.0, price=12.0, exit_class="stop_loss",
            protection_trigger="margin_deleveraging",
            exit_reason="forced_margin_deleveraging", reopen_lock_pending=True,
        ),
    ]

    strategy.on_trades_filled(group)

    assert sizing.position_side == "long_x"
    assert sizing.position_id == "position-p"
    assert sizing.x_quantity == 4.0
    assert sizing.y_quantity == 4.0
    assert sizing.current_gross_notional == 88.0
    assert sizing.remaining_pair_gross == 12.0
    assert protection.entry_x_quantity == 4.0
    assert protection.entry_y_quantity == 4.0
    assert protection.entry_gross_notional == 80.0
    assert protection.entry_fee == pytest.approx(0.8)
    assert protection.entry_slippage == pytest.approx(0.8)
    assert protection.funding_cost == pytest.approx(1.6)
    assert protection.reopen_lock_pending is False


def test_explicit_false_wait_switch_allows_reopen_after_protective_close():
    strategy, pipeline = _strategy_with_protection()
    strategy.protection_cfg.wait_for_model_update_after_non_z_exit = False
    sizing = pipeline.state.sizing_state
    sizing.position_side = "long_x"
    sizing.x_quantity = 1.0
    sizing.y_quantity = 1.0
    close_group = [
        SimpleNamespace(
            pair_id="p", group_id="g", action="close", symbol="X", quantity=1.0,
            exit_class="stop_loss", reopen_lock_pending=False, exit_reason="protective_take_profit",
        ),
        SimpleNamespace(
            pair_id="p", group_id="g", action="close", symbol="Y", quantity=1.0,
            exit_class="stop_loss", reopen_lock_pending=False, exit_reason="protective_take_profit",
        ),
    ]
    strategy.on_trades_filled(close_group)
    assert pipeline.state.protection_state.reopen_lock_pending is False


def test_pair_loss_stop_triggers_on_net_return_and_exposes_freeze_config():
    strategy = MultiPairStrategy(
        pairs=[PairDefinition(pair_id="p", x_symbol="X", y_symbol="Y")],
        estimator_ctor=TLSEstimator,
        estimator_cfg=EstimatorConfig(regression_method="price", position_update_policy="freeze"),
        protection_cfg=ProtectionConfig(
            enabled=True, pair_loss_stop_enabled=True, pair_loss_stop_return=0.05,
            pair_loss_stop_freeze_bars=120,
        ),
    )
    pipeline = strategy.pipelines["p"]
    pipeline.state.sizing_state.position_side = "long_x"
    pipeline.state.sizing_state.x_quantity = 1.0
    pipeline.state.sizing_state.y_quantity = 1.0
    state = pipeline.state.protection_state
    state.active = True
    state.side = "long_x"
    state.entry_x_price = 10.0
    state.entry_y_price = 10.0
    state.entry_gross_notional = 20.0
    state.pair_loss_stop_return = 0.05
    bundle = PairBarBundle(
        "p", 1, 1,
        BarSnapshot(1, 9.0, 9.0, 9.0, 9.0, 1, "binance", "X"),
        BarSnapshot(1, 10.0, 10.0, 10.0, 10.0, 1, "binance", "Y"),
    )
    assert strategy._protection_exit_decision(pipeline, bundle) == {
        "reason": "protective_pair_loss_stop", "trigger": "pair_loss_stop"
    }


def test_pair_loss_stop_freeze_can_follow_model_lookback():
    strategy = MultiPairStrategy(
        pairs=[PairDefinition(pair_id="p", x_symbol="X", y_symbol="Y")],
        estimator_ctor=TLSEstimator,
        estimator_cfg=EstimatorConfig(regression_method="price", position_update_policy="freeze", model_lookback_bars=10),
        protection_cfg=ProtectionConfig(enabled=True, pair_loss_stop_enabled=True, pair_loss_stop_freeze_model_lookback_multiplier=1.0),
    )
    pipeline = strategy.pipelines["p"]
    pipeline.state.last_bar_index = 1
    group = [SimpleNamespace(symbol="X", quantity=1.0, price=10.0, fee=0.0, slippage=0.0, ts=1, para={}), SimpleNamespace(symbol="Y", quantity=1.0, price=10.0, fee=0.0, slippage=0.0, ts=1, para={})]
    strategy._initialize_or_update_protection(pipeline, group, "long_x")
    assert pipeline.state.protection_state.pair_loss_stop_freeze_bars == 10


def test_pair_loss_stop_freeze_blocks_entry_until_deadline():
    strategy, pipeline = _strategy_with_protection()
    pipeline.state.protection_state.pair_loss_stop_freeze_until_bar = 15
    signal = SimpleNamespace(side="long_x")
    pipeline.state.sizing_state.position_side = None
    pipeline.state.sizing_state.remaining_pair_gross = 100.0
    pipeline.state.last_bar_index = 10
    blocked = strategy._entry_schedule_decision(pipeline, signal, 10)
    assert blocked["allowed"] is False
    assert "pair loss stop freeze" in blocked["reason"]
    allowed = strategy._entry_schedule_decision(pipeline, signal, 15)
    assert allowed["allowed"] is True


def test_rebalance_close_is_frozen_without_model_update_lock():
    strategy, pipeline = _strategy_with_protection()
    pipeline.state.sizing_state.position_side = "long_x"
    pipeline.state.sizing_state.x_quantity = 1.0
    pipeline.state.sizing_state.y_quantity = 1.0
    orders = strategy._close_orders(
        "p", pipeline, pipeline.pair_def, 1.0,
        exit_reason="rebalance_replacement",
        protection_trigger="rebalance_replacement",
        exit_class="rebalance_replacement",
        protection_freeze_bars=12,
    )
    assert orders
    assert all(order.protection_rule == "rebalance_replacement" for order in orders)
    assert all(order.protection_freeze_bars == 12 for order in orders)
    assert all(order.reopen_lock_pending is False for order in orders)


def test_rebalance_release_uses_pair_net_liquidation_value():
    from core.modules.models.pipeline_types import RawPairTarget
    from core.modules.strategy.config import RebalanceConfig

    strategy = MultiPairStrategy(
        pairs=[
            PairDefinition(pair_id="old_a", x_symbol="A", y_symbol="B"),
            PairDefinition(pair_id="old_b", x_symbol="C", y_symbol="D"),
            PairDefinition(pair_id="new", x_symbol="E", y_symbol="F"),
        ],
        estimator_ctor=TLSEstimator,
        estimator_cfg=EstimatorConfig(regression_method="price", position_update_policy="freeze"),
        rebalance_cfg=RebalanceConfig(enabled=True, minimum_entry_capital_ratio=0.5),
    )
    for pair_id in ("old_a", "old_b"):
        pp = strategy.pipelines[pair_id]
        pp.state.sizing_state.position_side = "long_x"
        pp.state.sizing_state.x_quantity = 5.0
        pp.state.sizing_state.y_quantity = 5.0
        pp.state.sizing_state.current_gross_notional = 100.0
        pp.state.sizing_state.target_hedge_ratio = 1.0
        pp.state.protection_state.active = True
        pp.state.protection_state.side = "long_x"
        pp.state.protection_state.entry_x_price = 10.0
        pp.state.protection_state.entry_y_price = 10.0
        pp.state.protection_state.entry_x_quantity = 5.0
        pp.state.protection_state.entry_y_quantity = 5.0
        pp.state.protection_state.entry_gross_notional = 100.0
    bundles = {
        pair_id: PairBarBundle(
            pair_id, 1, 1,
            BarSnapshot(1, 8.0, 8.0, 8.0, 8.0, 1, "binance", strategy.pipelines[pair_id].pair_def.x_symbol),
            BarSnapshot(1, 10.0, 10.0, 10.0, 10.0, 1, "binance", strategy.pipelines[pair_id].pair_def.y_symbol),
        ) for pair_id in ("old_a", "old_b")
    }
    raw = RawPairTarget(
        pair_id="new", ready=True, side="long_x", x_weight=0.5, y_weight=0.5,
        x_price=10.0, y_price=10.0, gross_notional=600.0, target_capital=600.0,
    )
    candidate = {"pair_id": "new", "pair_def": strategy.pipelines["new"].pair_def, "raw_target": raw, "result": SimpleNamespace(estimator=SimpleNamespace(cointegration_pvalue=0.1))}
    strategy.portfolio_state.available_balance = 250.0
    orders = strategy._plan_rebalance({"new": candidate}, bundles, 1)
    assert len(orders) == 2
    assert strategy._pending_rebalance_pair_ids == {"old_a"}
    assert strategy._pending_rebalance_release == pytest.approx(89.946)


def test_rebalance_min_holding_uses_pair_local_bar_index():
    from core.modules.models.pipeline_types import RawPairTarget
    from core.modules.strategy.config import RebalanceConfig

    strategy = MultiPairStrategy(
        pairs=[
            PairDefinition(pair_id="old", x_symbol="A", y_symbol="B"),
            PairDefinition(pair_id="new", x_symbol="C", y_symbol="D"),
        ],
        estimator_ctor=TLSEstimator,
        estimator_cfg=EstimatorConfig(regression_method="price", position_update_policy="freeze"),
        rebalance_cfg=RebalanceConfig(
            enabled=True,
            minimum_entry_capital_ratio=0.5,
            eviction_min_holding_bars=5,
        ),
    )
    old = strategy.pipelines["old"]
    old.state.sizing_state.position_side = "long_x"
    old.state.sizing_state.x_quantity = 5.0
    old.state.sizing_state.y_quantity = 5.0
    old.state.sizing_state.target_hedge_ratio = 1.0
    old.state.protection_state.active = True
    old.state.protection_state.side = "long_x"
    old.state.protection_state.entry_bar_index = 5
    old.state.protection_state.entry_x_price = 10.0
    old.state.protection_state.entry_y_price = 10.0
    old.state.protection_state.entry_x_quantity = 5.0
    old.state.protection_state.entry_y_quantity = 5.0
    old.state.protection_state.entry_gross_notional = 100.0
    old.state.last_bar_index = 5

    bundles = {
        "old": PairBarBundle(
            "old", 5, 5,
            BarSnapshot(1, 8.0, 8.0, 8.0, 8.0, 1, "binance", "A"),
            BarSnapshot(1, 10.0, 10.0, 10.0, 10.0, 1, "binance", "B"),
        )
    }
    raw = RawPairTarget(
        pair_id="new", ready=True, side="long_x", x_weight=0.5, y_weight=0.5,
        x_price=10.0, y_price=10.0, gross_notional=100.0, target_capital=100.0,
    )
    candidate = {
        "pair_id": "new",
        "pair_def": strategy.pipelines["new"].pair_def,
        "raw_target": raw,
        "result": SimpleNamespace(estimator=SimpleNamespace(cointegration_pvalue=0.1)),
    }
    strategy.portfolio_state.available_balance = 0.0

    # The global strategy bar is deliberately much later, but this Pair has
    # only held for zero local bars and must not be evicted.
    assert strategy._plan_rebalance({"new": candidate}, bundles, 100) == []

    # Once the Pair-local bar reaches the threshold, eviction is allowed.
    bundles["old"] = replace(bundles["old"], bar_index=10)
    orders = strategy._plan_rebalance({"new": candidate}, bundles, 100)
    assert orders
