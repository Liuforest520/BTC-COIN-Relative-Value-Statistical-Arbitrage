from core.modules.estimator.kalman import KalmanEstimator
from core.modules.models.pipeline_types import EstimatorState


def test_kalman_initializes_from_full_warmup_before_ready():
    estimator = KalmanEstimator(
        pair_id="test_pair",
        regression_method="price",
        model_lookback_bars=5,
        warmup_bars=5,
        zscore_lookback_bars=5,
    )
    state = EstimatorState()

    outputs = [
        estimator.update(state, float(i), 2.0 + 3.0 * float(i), i)
        for i in range(1, 5)
    ]
    assert all(not item.ready for item in outputs)
    assert state.kalman_theta is None

    initialized = estimator.update(state, 5.0, 17.0, 5)
    assert not initialized.ready
    assert state.kalman_theta is not None
    assert state.kalman_P is not None
    assert state.kalman_R is not None
    assert state.spread_sample_count == 5

    tradable = estimator.update(state, 6.0, 20.0, 6)
    assert tradable.ready
    assert abs(tradable.alpha - 2.0) < 1e-6
    assert abs(tradable.beta - 3.0) < 1e-6
    assert tradable.spread_sample_count == 5


def test_kalman_rolling_spread_stats_match_window_stats():
    estimator = KalmanEstimator(
        pair_id="test_pair",
        regression_method="price",
        model_lookback_bars=3,
    )
    state = EstimatorState()

    for spread in [0.0, 1.0, 2.0, 3.0]:
        estimator._append_spread(state, spread)

    assert state.spread_sample_count == 3
    assert all(abs(a - b) < 1e-9 for a, b in zip(state.spread_history, [1.0, 2.0, 3.0]))
    assert abs(state.spread_mean - 2.0) < 1e-9
    assert abs(state.spread_std - 1.0) < 1e-9
