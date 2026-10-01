from types import SimpleNamespace

import pytest

from core.modules.estimator.tls import TLSEstimator
from core.modules.models.pipeline_types import BarSnapshot, PairBarBundle, RawPairTarget
from core.modules.portfolio.allocator import PairTargetCapitalAllocator
from core.modules.strategy.config import (
    EstimatorConfig,
    PairDefinition,
    PortfolioConfig,
    ProtectionConfig,
    ProfitablePositionReplacementConfig,
    RebalanceCandidateQualityConfig,
    RebalanceConfig,
)
from core.modules.strategy.multi_pair_strategy import MultiPairStrategy
from core.modules.strategy.rebalance import (
    RebalanceCandidate,
    RebalanceManager,
    RebalancePosition,
)


def _manager(quality_enabled=True, replacement_enabled=True, min_net_return=0.0, min_convergence_ratio=0.80):
    return RebalanceManager(RebalanceConfig(
        enabled=True,
        candidate_quality=RebalanceCandidateQualityConfig(
            enabled=quality_enabled,
            min_theoretical_zero_return_x_move=0.20,
            max_adf_pvalue=0.4,
            min_expected_net_return=0.0,
        ),
        profitable_position_replacement=ProfitablePositionReplacementConfig(
            enabled=replacement_enabled,
            min_net_return=min_net_return,
            min_convergence_ratio=min_convergence_ratio,
        ),
    ))


def _candidate(pair_id, expected, move=0.25, adf=0.2, minimum=50.0):
    return RebalanceCandidate(pair_id, 100.0, minimum, adf, move, expected, 3.0)


def _position(pair_id, net_return, release, convergence=0.90):
    return RebalancePosition(pair_id, net_return, convergence, release)


def test_rebalance_selects_only_best_qualified_underfunded_candidate():
    plan = _manager().plan(
        [_candidate("bad_adf", 0.20, adf=0.5), _candidate("good", 0.04), _candidate("best", 0.06)],
        [_position("old", 0.05, 50.0)],
        available_capital=0.0,
    )
    assert plan.ready is True
    assert plan.candidate.pair_id == "best"
    assert [position.pair_id for position in plan.evictions] == ["old"]


def test_losing_position_does_not_bypass_candidate_quality_gate():
    plan = _manager().plan(
        [_candidate("unstable", 0.20, move=0.10, adf=0.5)],
        [_position("old_loss", -0.5, 100.0, convergence=1.0)],
        available_capital=0.0,
    )
    assert plan.ready is False
    assert plan.candidate is None


def test_rebalance_excludes_loss_and_low_convergence_positions():
    plan = _manager().plan(
        [_candidate("new", 0.05, minimum=80.0)],
        [_position("loss_but_converged", -0.5, 100.0, convergence=1.0),
         _position("profit_not_converged", 0.05, 100.0, convergence=0.79)],
        available_capital=0.0,
    )
    assert plan.ready is False
    assert plan.evictions == ()


def test_profitable_converged_position_at_threshold_can_be_replaced():
    plan = _manager().plan(
        [_candidate("new", 0.05)],
        [_position("old", 0.000001, 100.0, convergence=0.80)],
        available_capital=0.0,
    )
    assert plan.ready is True
    assert [p.pair_id for p in plan.evictions] == ["old"]


def test_profitable_replacement_min_net_return_rejects_negative_threshold():
    with pytest.raises(ValueError, match="min_net_return must be finite and non-negative"):
        ProfitablePositionReplacementConfig(min_net_return=-0.05)


def test_rebalance_sorts_by_convergence_then_return_then_pair_id():
    plan = _manager().plan(
        [_candidate("new", 0.05, minimum=70.0)],
        [_position("low", 0.50, 40.0, convergence=0.82),
         _position("high", 0.01, 40.0, convergence=0.95),
         _position("same_a", 0.20, 10.0, convergence=0.90),
         _position("same_b", 0.30, 35.0, convergence=0.90)],
        available_capital=0.0,
    )
    assert plan.ready is True
    assert [p.pair_id for p in plan.evictions] == ["high", "same_b"]


def test_rebalance_cancels_entire_batch_when_qualified_releases_are_insufficient():
    plan = _manager().plan(
        [_candidate("new", 0.05, minimum=80.0)],
        [_position("old_a", 0.20, 20.0), _position("old_b", 0.10, 30.0)],
        available_capital=0.0,
    )
    assert plan.ready is False
    assert plan.evictions == ()
    assert plan.planned_release == pytest.approx(50.0)


def _strategy(protection_enabled=False, pair_override=None, model_lookback=10):
    old = PairDefinition(pair_id="old", x_symbol="A", y_symbol="B", model_lookback_bars_override=pair_override)
    return MultiPairStrategy(
        pairs=[old, PairDefinition(pair_id="new_a", x_symbol="C", y_symbol="D"), PairDefinition(pair_id="new_b", x_symbol="E", y_symbol="F")],
        estimator_ctor=TLSEstimator,
        estimator_cfg=EstimatorConfig(regression_method="price", position_update_policy="freeze", model_lookback_bars=model_lookback),
        protection_cfg=ProtectionConfig(enabled=protection_enabled),
        rebalance_cfg=RebalanceConfig(enabled=True, minimum_entry_capital_ratio=0.5),
    )


def _fill_old_position(strategy):
    pipeline = strategy.pipelines["old"]
    pipeline.state.last_bar_index = 0
    pipeline.state.sizing_state.position_side = "long_x"
    pipeline.state.sizing_state.position_id = "position-old"
    trades = [
        SimpleNamespace(pair_id="old", group_id="open-old", action="open", symbol="A", side="buy", quantity=5.0, price=10.0, fee=0.1, slippage=0.0, ts=1, position_id="position-old", para={"pair_position": {"entry_z": 3.3, "exit_z": 0.5}}),
        SimpleNamespace(pair_id="old", group_id="open-old", action="open", symbol="B", side="sell", quantity=5.0, price=10.0, fee=0.1, slippage=0.0, ts=1, position_id="position-old", para={"pair_position": {"entry_z": 3.3, "exit_z": 0.5}}),
    ]
    strategy.on_trades_filled(trades)
    pipeline.state.last_bar_index = 10
    pipeline._last_zscore = 1.0
    return pipeline


def _bundle(pair_id="old", bar_index=10, x_close=12.0, y_close=10.0):
    return PairBarBundle(pair_id, bar_index, bar_index, BarSnapshot(bar_index, x_close, x_close, x_close, x_close, 1, "binance", "A"), BarSnapshot(bar_index, y_close, y_close, y_close, y_close, 1, "binance", "B"))


def _raw_candidate(pair_id, expected_return):
    symbols = ("C", "D") if pair_id == "new_a" else ("E", "F")
    raw = RawPairTarget(pair_id=pair_id, ready=True, side="long_x", x_weight=0.5, y_weight=0.5, x_price=10.0, y_price=10.0, x_notional=50.0, y_notional=50.0, x_quantity=5.0, y_quantity=5.0, gross_notional=100.0, target_capital=100.0, signal_strength=3.0, para={"rebalance_quality": {"adf_pvalue": 0.2, "theoretical_zero_return_x_move": 0.25, "expected_net_return": expected_return, "signal_strength": 3.0}})
    return {"pair_id": pair_id, "pair_def": PairDefinition(pair_id=pair_id, x_symbol=symbols[0], y_symbol=symbols[1]), "raw_target": raw, "result": SimpleNamespace(estimator=SimpleNamespace(cointegration_pvalue=0.2)), "fallback_hedge": 1.0, "position_id": None}


def test_convergence_ratio_formula():
    strategy = _strategy()
    pipeline = _fill_old_position(strategy)
    assert strategy._pair_convergence_ratio(pipeline) == pytest.approx(2.3 / 2.8)
    pipeline._last_zscore = None
    assert strategy._pair_convergence_ratio(pipeline) is None


def test_position_ledger_records_entry_z_when_protection_disabled():
    strategy = _strategy(protection_enabled=False)
    pipeline = _fill_old_position(strategy)
    ledger = pipeline.state.position_ledger
    assert ledger.active is True
    assert ledger.entry_z == pytest.approx(3.3)
    assert ledger.exit_z == pytest.approx(0.5)
    assert pipeline.state.protection_state.active is False


def test_protection_switch_does_not_change_rebalance_choice():
    chosen = []
    for enabled in (False, True):
        strategy = _strategy(protection_enabled=enabled)
        _fill_old_position(strategy)
        candidates = {"new_a": _raw_candidate("new_a", 0.04), "new_b": _raw_candidate("new_b", 0.08)}
        orders, allocation = strategy._plan_rebalance(candidates, {"old": _bundle()}, 10, 0.0)
        assert len(orders) == 2
        chosen.append(allocation.selected_pair_ids)
    assert chosen == [["new_b"], ["new_b"]]


def test_filled_rebalance_close_does_not_freeze_or_lock_pair():
    strategy = _strategy(protection_enabled=False)
    pipeline = _fill_old_position(strategy)
    orders, allocation = strategy._plan_rebalance({"new_a": _raw_candidate("new_a", 0.08)}, {"old": _bundle()}, 10, 0.0)
    assert allocation.selected_pair_ids == ["new_a"]
    close_group = [SimpleNamespace(pair_id="old", group_id="close-old", action="close", symbol=order.symbol, quantity=order.quantity, exit_reason=order.exit_reason, protection_trigger=order.protection_trigger, exit_class=order.exit_class, protection_rule=order.protection_rule, protection_freeze_bars=order.protection_freeze_bars, reopen_lock_pending=order.reopen_lock_pending) for order in orders]
    strategy.on_trades_filled(close_group)
    state = pipeline.state.protection_state
    assert state.freeze_rule is None
    assert state.freeze_bars == 0
    assert state.freeze_until_bar is None
    assert state.reopen_lock_pending is False


def test_affordable_qualified_candidates_are_not_limited_to_one():
    strategy = MultiPairStrategy(
        pairs=[PairDefinition(pair_id="new_a", x_symbol="A", y_symbol="B"), PairDefinition(pair_id="new_b", x_symbol="C", y_symbol="D")],
        estimator_ctor=TLSEstimator,
        estimator_cfg=EstimatorConfig(regression_method="price", position_update_policy="freeze"),
        portfolio_ctor=PairTargetCapitalAllocator,
        portfolio_cfg=PortfolioConfig(method="pair_target_capital", minimum_entry_capital_ratio=0.5),
        rebalance_cfg=RebalanceConfig(enabled=True),
    )
    candidates = {pair_id: _raw_candidate(pair_id, expected_return) for pair_id, expected_return in (("new_a", 0.04), ("new_b", 0.08))}
    strategy.portfolio_state.ready = True
    strategy.portfolio_state.available_balance = 200.0
    allocation = strategy._allocate_entry_targets({pair_id: candidate["raw_target"] for pair_id, candidate in candidates.items()}, bar_index=1)
    assert allocation.selected_pair_ids == ["new_a", "new_b"]


def test_rebalance_uses_exchange_pair_marked_equity_when_context_is_available():
    strategy = _strategy(protection_enabled=False)
    pipeline = _fill_old_position(strategy)
    strategy.set_portfolio_context(pair_marked_equity={"old": 12.5})
    assert strategy._pair_releasable_equity(pipeline, _bundle()) == pytest.approx(12.5)


def test_rebalance_quality_gates_fail_fast_for_non_frozen_models():
    with pytest.raises(ValueError, match="candidate_quality requires a frozen Price/Log-Price model"):
        MultiPairStrategy(
            pairs=[PairDefinition(pair_id="p", x_symbol="A", y_symbol="B")],
            estimator_ctor=TLSEstimator,
            estimator_cfg=EstimatorConfig(regression_method="price", position_update_policy="update"),
            rebalance_cfg=RebalanceConfig(enabled=True),
        )


def test_aborted_rebalance_restores_protection_diagnostics():
    strategy = _strategy(protection_enabled=False)
    pipeline = _fill_old_position(strategy)
    state = pipeline.state.protection_state
    state.protection_rule = "previous_rule"
    state.freeze_rule = "previous_freeze"
    state.freeze_bars = 7
    state.freeze_until_bar = 99
    original_allocate = strategy._allocate_entry_targets
    strategy._allocate_entry_targets = lambda *args, **kwargs: None
    try:
        orders, allocation = strategy._plan_rebalance(
            {"new_a": _raw_candidate("new_a", 0.08)}, {"old": _bundle()}, 10, 0.0
        )
    finally:
        strategy._allocate_entry_targets = original_allocate
    assert orders == []
    assert allocation is None
    assert state.pending_exit is False
    assert state.protection_rule == "previous_rule"
    assert state.freeze_rule == "previous_freeze"
    assert state.freeze_bars == 7
    assert state.freeze_until_bar == 99
