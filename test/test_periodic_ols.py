from core.modules.estimator.periodic_ols import PeriodicOLSEstimator
from core.modules.models.pipeline_types import EstimatorState


def test_periodic_ols_fits_after_warmup_and_trades_next_bar():
    estimator = PeriodicOLSEstimator(
        pair_id="test_pair",
        regression_method="price",
        model_lookback_bars=5,
        warmup_bars=5,
        model_update_interval_bars=10,
        zscore_lookback_bars=5,
    )
    state = EstimatorState()

    outputs = [
        estimator.update(state, float(i), 2.0 + 3.0 * float(i), i)
        for i in range(1, 5)
    ]
    assert all(not item.ready for item in outputs)
    assert state.spread_beta is None

    initialized = estimator.update(state, 5.0, 17.0, 5)
    assert not initialized.ready
    assert abs(state.alpha - 2.0) < 1e-6
    assert abs(state.spread_beta - 3.0) < 1e-6
    assert state.spread_sample_count == 5

    tradable = estimator.update(state, 6.0, 20.0, 6)
    assert tradable.ready
    assert abs(tradable.alpha - 2.0) < 1e-6
    assert abs(tradable.beta - 3.0) < 1e-6
    assert abs(tradable.latest_spread) < 1e-6
