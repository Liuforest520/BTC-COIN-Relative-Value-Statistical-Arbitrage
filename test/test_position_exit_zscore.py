from math import log, sqrt

import pytest

from core.modules.models.pipeline_types import (
    BarSnapshot,
    EstimatorOutput,
    PairBarBundle,
    SignalOutput,
    SignalState,
)
from core.modules.signals.position_exit_zscore import (
    PositionExitZScoreTracker,
    ReturnObservation,
)
from core.modules.signals.zscore import ZScoreSignal
from core.modules.signals.zscore_reversion import ZScoreReversionSignal
from core.modules.strategy.config import (
    EstimatorConfig,
    ExecutionConfig,
    PairDefinition,
    SignalConfig,
)
from core.modules.strategy.pair_pipeline import PairPipeline


def _observation(bar_index, x_return, frozen_zscore):
    alpha, beta, mean, std = 0.1, 2.0, 0.05, 0.5
    y_return = alpha + beta * x_return + mean + frozen_zscore * std
    return ReturnObservation(
        bar_index=bar_index,
        x_return=x_return,
        y_return=y_return,
        alpha=alpha,
        beta=beta,
        residual_mean=mean,
        residual_std=std,
    )


def test_sum_zscore_includes_entry_once_then_adds_each_new_bar():
    tracker = PositionExitZScoreTracker("sum_zscore")
    entry = _observation(10, 0.1, 2.0)
    next_bar = _observation(11, 0.2, -0.5)

    assert tracker.on_bar(entry, has_position=False) is None
    tracker.on_position_opened()
    assert tracker.value == pytest.approx(2.0)

    # Seeing the entry bar again must not count its return twice.
    assert tracker.on_bar(entry, has_position=True) == pytest.approx(2.0)
    assert tracker.on_bar(next_bar, has_position=True) == pytest.approx(1.5)
    assert tracker.details["return_count"] == 2


def test_cumulative_return_starts_with_the_entry_bar_return():
    tracker = PositionExitZScoreTracker("cumulative_return")
    entry = _observation(20, 0.1, 2.0)
    next_bar = _observation(21, 0.2, -0.5)

    tracker.on_bar(entry, has_position=False)
    tracker.on_position_opened()
    assert tracker.value == pytest.approx(entry.zscore)
    assert tracker.details["sum_x_return"] == pytest.approx(entry.x_return)
    assert tracker.details["sum_y_return"] == pytest.approx(entry.y_return)

    value = tracker.on_bar(next_bar, has_position=True)
    assert value == pytest.approx((2.0 - 0.5) / sqrt(2.0))
    assert tracker.details["return_count"] == 2


@pytest.mark.parametrize("signal_class", [ZScoreSignal, ZScoreReversionSignal])
def test_signal_uses_position_statistic_only_for_exit(signal_class):
    signal = signal_class(
        pair_id="pair",
        entry_z=2.0,
        exit_z=0.5,
        reversion_filter_enabled=False,
    )
    estimator = EstimatorOutput(
        pair_id="pair",
        ready=True,
        spread_mean=0.0,
        spread_std=1.0,
        latest_spread=2.5,
    )
    state = SignalState(entry_z_upper=2.0, entry_z_lower=-2.0)
    para = {"position": {
        "has_position": True,
        "exit_zscore_method": "cumulative_return",
        "exit_zscore": 0.2,
    }}

    result = signal.evaluate(state, estimator, bar_index=5, para=para)

    assert result.action == "close"
    assert result.zscore == pytest.approx(0.2)
    assert result.para["signal"]["standard_zscore"] == pytest.approx(2.5)


@pytest.mark.parametrize("signal_class", [ZScoreSignal, ZScoreReversionSignal])
@pytest.mark.parametrize(
    ("position_side", "current_zscore"),
    [("short_x", 3.0), ("long_x", -3.0)],
)
def test_signal_closes_when_zscore_reverses_position_direction(
    signal_class, position_side, current_zscore
):
    signal = signal_class(
        pair_id="pair",
        entry_z=2.0,
        exit_z=0.5,
        reversion_filter_enabled=False,
    )
    estimator = EstimatorOutput(
        pair_id="pair",
        ready=True,
        spread_mean=0.0,
        spread_std=1.0,
        latest_spread=current_zscore,
    )
    state = SignalState(entry_z_upper=2.0, entry_z_lower=-2.0)
    para = {"position": {
        "has_position": True,
        "side": position_side,
        "exit_zscore_method": "standard",
        "exit_zscore": None,
    }}

    result = signal.evaluate(state, estimator, bar_index=5, para=para)

    assert result.action == "close"
    assert "direction reversed" in result.reason


@pytest.mark.parametrize("signal_class", [ZScoreSignal, ZScoreReversionSignal])
def test_signal_does_not_close_outside_band_without_direction_reversal(signal_class):
    signal = signal_class(
        pair_id="pair",
        entry_z=2.0,
        exit_z=0.5,
        reversion_filter_enabled=False,
    )
    estimator = EstimatorOutput(
        pair_id="pair",
        ready=True,
        spread_mean=0.0,
        spread_std=1.0,
        latest_spread=-1.0,
    )
    state = SignalState(entry_z_upper=2.0, entry_z_lower=-2.0)
    para = {"position": {
        "has_position": True,
        "side": "short_x",
        "exit_zscore_method": "standard",
        "exit_zscore": None,
    }}

    result = signal.evaluate(state, estimator, bar_index=5, para=para)

    assert result.action == "none"


@pytest.mark.parametrize("signal_class", [ZScoreSignal, ZScoreReversionSignal])
@pytest.mark.parametrize("exit_method", ["sum_zscore", "cumulative_return"])
def test_nonstandard_exit_zscore_closes_after_direction_reversal(
    signal_class, exit_method
):
    signal = signal_class(
        pair_id="pair",
        entry_z=2.0,
        exit_z=0.5,
        reversion_filter_enabled=False,
    )
    estimator = EstimatorOutput(
        pair_id="pair",
        ready=True,
        spread_mean=0.0,
        spread_std=1.0,
        latest_spread=-3.0,
    )
    state = SignalState(entry_z_upper=2.0, entry_z_lower=-2.0)
    para = {"position": {
        "has_position": True,
        "side": "short_x",
        "exit_zscore_method": exit_method,
        "exit_zscore": 3.0,
    }}

    result = signal.evaluate(state, estimator, bar_index=5, para=para)

    assert result.action == "close"
    assert result.zscore == pytest.approx(3.0)
    assert "direction reversed" in result.reason


class _DummyEstimator:
    def __init__(self, **kwargs):
        pass


class _ReturnIntervalEstimator:
    def __init__(self, return_interval_bars=1, **kwargs):
        self.return_interval_bars = return_interval_bars

    def update(self, state, x_close, y_close, bar_index, para=None):
        state.x_close_history.append(float(x_close))
        state.y_close_history.append(float(y_close))
        return EstimatorOutput(
            pair_id="pair",
            bar_index=bar_index,
            ready=True,
            alpha=0.0,
            beta=1.0,
            spread_beta=1.0,
            spread_mean=0.0,
            spread_std=1.0,
            latest_spread=0.0,
        )


class _CaptureSignal:
    def __init__(self, pair_id="", **kwargs):
        self.pair_id = pair_id
        self.last_para = None

    def evaluate(self, state, estimator, bar_index, para=None):
        self.last_para = para
        return SignalOutput(
            pair_id=self.pair_id,
            bar_index=bar_index,
            ready=True,
            action="none",
            zscore=0.0,
        )


def _pipeline(estimator_method, regression_method, exit_method):
    return PairPipeline(
        PairDefinition(pair_id="pair", x_symbol="X", y_symbol="Y"),
        _DummyEstimator,
        EstimatorConfig(
            method=estimator_method,
            regression_method=regression_method,
        ),
        ZScoreSignal,
        SignalConfig(position_exit_zscore_method=exit_method),
        None,
        None,
        ExecutionConfig(),
    )


@pytest.mark.parametrize(
    "method",
    ["age_weighted_wls", "residual_weighted_wls", "tls", "huber"],
)
def test_four_return_estimators_accept_both_position_exit_methods(method):
    for exit_method in ("sum_zscore", "cumulative_return"):
        pipeline = _pipeline(method, "log_return", exit_method)
        assert pipeline.position_exit_zscore.method == exit_method


def test_nonstandard_position_exit_method_rejects_other_models_and_price_regression():
    with pytest.raises(ValueError, match="supported only"):
        _pipeline("rolling_ols", "log_return", "sum_zscore")
    with pytest.raises(ValueError, match="requires log_return"):
        _pipeline("tls", "price", "cumulative_return")


def test_nonstandard_exit_observation_uses_estimator_return_interval():
    pipeline = PairPipeline(
        PairDefinition(pair_id="pair", x_symbol="X", y_symbol="Y"),
        _ReturnIntervalEstimator,
        EstimatorConfig(
            method="tls",
            regression_method="log_return",
            return_interval_bars=3,
        ),
        None,
        SignalConfig(position_exit_zscore_method="sum_zscore"),
        None,
        None,
        ExecutionConfig(),
    )
    pipeline.state.estimator_state.x_close_history.extend([100.0, 110.0, 120.0])
    pipeline.state.estimator_state.y_close_history.extend([200.0, 210.0, 220.0])
    bundle = PairBarBundle(
        pair_id="pair",
        ts=1,
        bar_index=3,
        x_bar=BarSnapshot(ts=1, open=130.0, high=130.0, low=130.0, close=130.0, volume=1.0),
        y_bar=BarSnapshot(ts=1, open=230.0, high=230.0, low=230.0, close=230.0, volume=1.0),
    )

    pipeline.on_bar(bundle)
    observation = pipeline.position_exit_zscore.latest_observation

    assert observation is not None
    assert observation.x_return == pytest.approx(log(130.0 / 100.0))
    assert observation.y_return == pytest.approx(log(230.0 / 200.0))


def test_compact_pipeline_still_passes_position_side_to_signal():
    pipeline = PairPipeline(
        PairDefinition(pair_id="pair", x_symbol="X", y_symbol="Y"),
        _ReturnIntervalEstimator,
        EstimatorConfig(method="tls", regression_method="log_return"),
        _CaptureSignal,
        SignalConfig(position_exit_zscore_method="standard"),
        None,
        None,
        ExecutionConfig(),
    )
    pipeline.collect_diagnostics = False
    pipeline.state.sizing_state.position_side = "short_x"
    pipeline.state.sizing_state.x_quantity = 1.0
    pipeline.state.sizing_state.y_quantity = 1.0
    bundle = PairBarBundle(
        pair_id="pair",
        ts=1,
        bar_index=1,
        x_bar=BarSnapshot(ts=1, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0),
        y_bar=BarSnapshot(ts=1, open=200.0, high=200.0, low=200.0, close=200.0, volume=1.0),
    )

    pipeline.on_bar(bundle, compute_sizing=False)

    assert pipeline.signal.last_para["position"]["has_position"] is True
    assert pipeline.signal.last_para["position"]["side"] == "short_x"
