import numpy as np
import pytest
from types import SimpleNamespace

from core.modules.estimator import (
    AgeWeightedWLSEstimator,
    DOLSEstimator,
    EWLSEstimator,
    HuberEstimator,
    ResidualWeightedWLSEstimator,
    RollingOLSEstimator,
    TLSEstimator,
    WinsorizedOLSEstimator,
)
from core.modules.models.pipeline_types import ESTIMATOR_REGISTRY, EstimatorOutput, EstimatorState
from core.modules.strategy.multi_pair_strategy import MultiPairStrategy
from core.modules.strategy.factory import build_strategy


ESTIMATORS = [RollingOLSEstimator, TLSEstimator, DOLSEstimator, EWLSEstimator, HuberEstimator]
RETURN_INTERVAL_ESTIMATORS = [TLSEstimator, HuberEstimator]
NEW_WEIGHTED_ESTIMATORS = [
    AgeWeightedWLSEstimator, WinsorizedOLSEstimator, ResidualWeightedWLSEstimator,
]


def test_five_beta_estimators_are_registered_independently():
    assert {"rolling_ols", "tls", "dols", "ewls", "huber"} <= set(ESTIMATOR_REGISTRY)
    assert len({ESTIMATOR_REGISTRY[name] for name in ("rolling_ols", "tls", "dols", "ewls", "huber")}) == 5


def test_three_additional_estimators_are_registered_independently():
    names = {"age_weighted_wls", "winsorized_ols", "residual_weighted_wls"}
    assert names <= set(ESTIMATOR_REGISTRY)
    assert len({ESTIMATOR_REGISTRY[name] for name in names}) == 3


@pytest.mark.parametrize(
    ("method", "estimator_class"),
    [("rolling_ols", RollingOLSEstimator), ("tls", TLSEstimator), ("dols", DOLSEstimator),
     ("ewls", EWLSEstimator), ("huber", HuberEstimator)],
)
def test_strategy_factory_constructs_each_beta_estimator(method, estimator_class):
    setup = SimpleNamespace(
        pairs=[{"pair_id": "pair", "x_symbol": "X", "y_symbol": "Y"}],
        pipeline={
            "estimator": {
                "method": method, "regression_method": "price",
                "model_lookback_bars": 10, "warmup_bars": 10,
            },
            "signal": {"method": "zscore"},
            "sizing": {"method": "beta_neutral"},
            "portfolio": {"method": "equal_weight"},
            "execution": {},
        },
    )

    strategy = build_strategy(setup, {"X": {}, "Y": {}})

    assert isinstance(strategy.pipelines["pair"].estimator, estimator_class)


@pytest.mark.parametrize(
    ("method", "estimator_class"),
    [
        ("age_weighted_wls", AgeWeightedWLSEstimator),
        ("residual_weighted_wls", ResidualWeightedWLSEstimator),
        ("tls", TLSEstimator),
        ("huber", HuberEstimator),
    ],
)
def test_strategy_factory_passes_return_interval_to_return_estimators(method, estimator_class):
    setup = SimpleNamespace(
        pairs=[{"pair_id": "pair", "x_symbol": "X", "y_symbol": "Y"}],
        pipeline={
            "estimator": {
                "method": method,
                "regression_method": "log_return",
                "return_interval_bars": 5,
                "model_lookback_bars": 60,
                "short_model_lookback_bars": 15,
                "age_bucket_end_bars": [15, 30, 45, 60],
            },
            "signal": {"method": "zscore"},
            "sizing": {"method": "beta_neutral"},
            "portfolio": {"method": "equal_weight"},
            "execution": {},
        },
    )

    strategy = build_strategy(setup, {"X": {}, "Y": {}})
    estimator = strategy.pipelines["pair"].estimator

    assert isinstance(estimator, estimator_class)
    assert estimator.regression_method == "log_return"
    assert estimator.return_interval_bars == 5


@pytest.mark.parametrize(
    ("estimator_class", "extra"),
    [
        (AgeWeightedWLSEstimator, {"age_bucket_end_bars": [150, 300, 450, 600]}),
        (ResidualWeightedWLSEstimator, {"short_model_lookback_bars": 180}),
    ],
)
def test_weighted_return_estimators_use_one_interval_for_fit_and_live_zscore(
    estimator_class, extra,
):
    estimator = estimator_class(
        regression_method="log_return",
        return_interval_bars=60,
        model_lookback_bars=600,
        **extra,
    )
    minute_returns = np.arange(1.0, 601.0) / 1_000_000.0
    prices = 100.0 * np.exp(np.r_[0.0, np.cumsum(minute_returns)])
    state = EstimatorState()
    state.x_close_history.extend(prices)
    state.y_close_history.extend(prices ** 1.5)

    sampled_x, sampled_y = estimator._prepare_xy(state)
    x_return, y_return = estimator._transform_current(
        state, prices[-1] * 1.01, prices[-1] ** 1.5 * 1.02,
    )

    assert len(sampled_x) == len(sampled_y) == 10
    assert x_return == pytest.approx(np.log((prices[-1] * 1.01) / prices[-60]))
    assert y_return == pytest.approx(
        np.log((prices[-1] ** 1.5 * 1.02) / (prices[-60] ** 1.5))
    )


def test_age_weighted_return_buckets_keep_bar_age_units():
    estimator = AgeWeightedWLSEstimator(
        regression_method="log_return",
        return_interval_bars=60,
        model_lookback_bars=600,
        age_bucket_end_bars=[150, 300, 450, 600],
        age_bucket_weights=[0.5, 0.7, 0.85, 1.0],
    )

    assert estimator._age_weights(10) == pytest.approx(
        [1.0, 1.0, 1.0, 0.85, 0.85, 0.7, 0.7, 0.7, 0.5, 0.5]
    )


def test_residual_weighted_return_windows_keep_bar_duration():
    estimator = ResidualWeightedWLSEstimator(
        regression_method="log_return",
        return_interval_bars=300,
        model_lookback_bars=28800,
        short_model_lookback_bars=10080,
    )

    assert estimator._model_sample_count() == 96
    assert estimator._short_sample_count() == 33


@pytest.mark.parametrize(
    ("method", "estimator_class"),
    [("age_weighted_wls", AgeWeightedWLSEstimator),
     ("winsorized_ols", WinsorizedOLSEstimator),
     ("residual_weighted_wls", ResidualWeightedWLSEstimator)],
)
def test_strategy_factory_constructs_each_additional_estimator(method, estimator_class):
    setup = SimpleNamespace(
        pairs=[{"pair_id": "pair", "x_symbol": "X", "y_symbol": "Y"}],
        pipeline={
            "estimator": {
                "method": method, "regression_method": "log_return",
                "model_lookback_bars": 10, "short_model_lookback_bars": 5,
                "age_bucket_end_bars": [2, 5, 8, 10],
            },
            "signal": {"method": "zscore"}, "sizing": {"method": "beta_neutral"},
            "portfolio": {"method": "equal_weight"}, "execution": {},
        },
    )

    strategy = build_strategy(setup, {"X": {}, "Y": {}})

    assert isinstance(strategy.pipelines["pair"].estimator, estimator_class)


def test_age_weighted_wls_uses_documented_segments_and_direction():
    estimator = AgeWeightedWLSEstimator(
        regression_method="price", model_lookback_bars=56, age_wls_profile="medium",
        age_bucket_end_bars=[1, 8, 24, 56],
    )
    weights = estimator._age_weights(56)
    x = np.arange(56, dtype=float)
    y = 1.0 + 2.0 * x
    y[-1] += 200.0
    _, weighted_beta = estimator._age_wls(x, y)
    _, ols_beta = RollingOLSEstimator._ols(x, y)

    assert weights[:32].tolist() == pytest.approx([1.0] * 32)
    assert weights[32:48].tolist() == pytest.approx([0.85] * 16)
    assert weights[48:55].tolist() == pytest.approx([0.70] * 7)
    assert weights[55] == pytest.approx(0.50)
    assert abs(weighted_beta - 2.0) < abs(ols_beta - 2.0)


def test_age_weighted_wls_accepts_arbitrary_bucket_bars_and_weights():
    estimator = AgeWeightedWLSEstimator(
        regression_method="price", model_lookback_bars=10,
        age_bucket_end_bars=[5, 10], age_bucket_weights=[0.2, 1.0],
    )

    assert estimator._age_weights(10).tolist() == pytest.approx(
        [1.0] * 5 + [0.2] * 5
    )


@pytest.mark.parametrize(
    ("lookback", "boundaries"),
    [
        (7200, [180, 1440, 2880, 7200]),
        (14400, [360, 1440, 5760, 14400]),
        (28800, [720, 4320, 11520, 28800]),
        (43200, [720, 5760, 17280, 43200]),
        (86400, [1440, 10080, 36000, 86400]),
    ],
)
def test_age_weighted_wls_uses_lookback_specific_time_buckets(lookback, boundaries):
    estimator = AgeWeightedWLSEstimator(
        regression_method="price", model_lookback_bars=lookback,
        age_bucket_end_bars=boundaries, age_wls_profile="medium",
    )

    weights = estimator._age_weights(lookback)

    assert np.count_nonzero(weights == 0.50) == boundaries[0]
    assert np.count_nonzero(weights == 0.70) == boundaries[1] - boundaries[0]
    assert np.count_nonzero(weights == 0.85) == boundaries[2] - boundaries[1]
    assert np.count_nonzero(weights == 1.00) == boundaries[3] - boundaries[2]


def test_winsorized_ols_clips_each_leg_and_reduces_tail_influence():
    x = np.arange(100, dtype=float)
    y = 1.0 + 2.0 * x
    y[-1] += 1_000.0
    estimator = WinsorizedOLSEstimator(
        regression_method="price", winsor_lower_quantile=0.01, winsor_upper_quantile=0.99
    )

    _, winsor_beta = estimator._winsorized_ols(x, y)
    _, ols_beta = RollingOLSEstimator._ols(x, y)
    clipped_y = estimator._winsorized_values(y)

    assert clipped_y[-1] == pytest.approx(np.quantile(y, 0.99))
    assert abs(winsor_beta - 2.0) < abs(ols_beta - 2.0)


def test_winsorized_ols_parameterizes_quantile_method():
    values = np.array([0.0, 1.0, 2.0, 100.0])
    estimator = WinsorizedOLSEstimator(
        regression_method="price", winsor_lower_quantile=0.25,
        winsor_upper_quantile=0.75, winsor_quantile_method="nearest",
    )

    clipped = estimator._winsorized_values(values)

    assert clipped.tolist() == pytest.approx([1.0, 1.0, 2.0, 2.0])


def test_residual_weight_profiles_match_documented_shock_bands():
    scores = np.array([np.nan, 1.5, 1.7, 2.5, 3.5])

    mild = ResidualWeightedWLSEstimator(
        model_lookback_bars=10, residual_wls_scale_lookback_bars=10,
        residual_wls_profile="mild",
    )
    medium = ResidualWeightedWLSEstimator(
        model_lookback_bars=10, residual_wls_scale_lookback_bars=10,
        residual_wls_profile="medium",
    )
    strong = ResidualWeightedWLSEstimator(
        model_lookback_bars=10, residual_wls_scale_lookback_bars=10,
        residual_wls_profile="strong",
    )

    assert mild._weights_from_scores(scores).tolist() == pytest.approx([1, 1, .9, .75, .5])
    assert medium._weights_from_scores(scores).tolist() == pytest.approx([1, 1, .75, .5, .25])
    assert strong._weights_from_scores(scores).tolist() == pytest.approx([1, 1, .6, .3, .1])


def test_residual_weights_use_current_short_model_for_entire_long_window():
    x = np.linspace(-2.0, 2.0, 40)
    noise = 0.05 * np.sin(np.arange(40, dtype=float))
    y = 0.2 + 1.4 * x + noise
    y[:20] = 0.2 + 2.6 * x[:20] + noise[:20]
    estimator = ResidualWeightedWLSEstimator(
        regression_method="price", model_lookback_bars=40,
        short_model_lookback_bars=10, residual_wls_profile="medium",
    )

    weights, scores = estimator._observation_weights(x, y)

    assert estimator.last_short_beta == pytest.approx(1.4, abs=0.1)
    assert np.mean(scores[:20]) > np.mean(scores[-10:])
    assert np.mean(weights[:20]) < np.mean(weights[-10:])


def test_residual_weights_are_recomputed_when_short_relationship_changes():
    x = np.linspace(-2.0, 2.0, 40)
    noise = 0.1 * np.sin(np.arange(40, dtype=float))
    y = 0.5 + 1.5 * x + noise
    changed_y = y.copy()
    changed_y[-10:] = 0.5 + 3.0 * x[-10:] + noise[-10:]
    estimator = ResidualWeightedWLSEstimator(
        regression_method="price", model_lookback_bars=40,
        short_model_lookback_bars=10,
        shock_score_boundaries=[1.0, 2.0, 3.0],
        shock_weights=[1.0, 0.7, 0.4, 0.1],
    )

    original_weights, original_scores = estimator._observation_weights(x, y)
    original_short_beta = estimator.last_short_beta
    changed_weights, changed_scores = estimator._observation_weights(x, changed_y)

    assert original_short_beta == pytest.approx(1.5, abs=0.2)
    assert estimator.last_short_beta == pytest.approx(3.0, abs=0.2)
    assert not np.allclose(changed_scores, original_scores)
    assert not np.array_equal(changed_weights, original_weights)


def test_residual_weight_boundaries_and_weights_are_fully_parameterized():
    estimator = ResidualWeightedWLSEstimator(
        model_lookback_bars=10, short_model_lookback_bars=5,
        shock_score_boundaries=[1.0, 4.0], shock_weights=[1.0, 0.6, 0.2],
    )

    assert estimator._weights_from_scores(
        np.array([np.nan, 1.0, 1.1, 4.0, 4.1])
    ).tolist() == pytest.approx([1.0, 1.0, 0.6, 0.6, 0.2])


@pytest.mark.parametrize("estimator_class", NEW_WEIGHTED_ESTIMATORS)
def test_additional_estimators_support_log_return_and_next_bar_activation(estimator_class):
    estimator = estimator_class(
        pair_id="weighted_pair", regression_method="log_return",
        model_lookback_bars=20, model_update_interval_bars=100,
        residual_wls_scale_lookback_bars=10,
        age_bucket_end_bars=[5, 10, 15, 20],
        residual_adf_min_samples=10,
    )
    rng = np.random.default_rng(29)
    return_count = estimator.warmup_bars
    x_returns = rng.normal(scale=0.01, size=return_count)
    noise = rng.normal(scale=0.001, size=return_count)
    y_returns = 0.001 + 1.6 * x_returns + noise
    x_prices = 100.0 * np.exp(np.r_[0.0, np.cumsum(x_returns)])
    y_prices = 80.0 * np.exp(np.r_[0.0, np.cumsum(y_returns)])
    state = EstimatorState()

    initialization = [
        estimator.update(state, x_prices[index], y_prices[index], index)
        for index in range(estimator.warmup_bars)
    ]
    tradable = estimator.update(
        state, x_prices[estimator.warmup_bars], y_prices[estimator.warmup_bars],
        estimator.warmup_bars,
    )

    assert all(not output.ready for output in initialization)
    assert tradable.ready
    assert tradable.spread_sample_count == 20
    assert tradable.beta == pytest.approx(1.6, abs=0.15)
    assert "residual ADF" in state.cointegration_reason


@pytest.mark.parametrize("estimator_class", NEW_WEIGHTED_ESTIMATORS)
@pytest.mark.parametrize("policy", ["freeze", "update"])
def test_additional_estimators_apply_position_update_policy(estimator_class, policy):
    estimator = estimator_class(
        regression_method="price", model_lookback_bars=10,
        model_update_interval_bars=3, position_update_policy=policy,
        residual_wls_scale_lookback_bars=10,
        age_bucket_end_bars=[2, 4, 7, 10],
        residual_adf_min_samples=10,
    )
    state = EstimatorState()
    for index in range(estimator.warmup_bars):
        x = float(index + 1)
        estimator.update(state, x, 1.0 + 2.0 * x, index)
    initial_update = state.last_model_update_index
    due_index = initial_update + 3
    for index in range(initial_update + 1, due_index):
        x = float(index + 1)
        estimator.update(state, x, 1.0 + 4.0 * x, index)

    estimator.update(
        state, float(due_index + 1), 1.0 + 4.0 * float(due_index + 1), due_index,
        para={"position": {"has_position": True}},
    )

    if policy == "freeze":
        assert state.last_model_update_index == initial_update
        assert state.last_model_update_skip_index == due_index
        assert state.next_model_update_index == due_index + 3
    else:
        assert state.last_model_update_index == due_index
        assert state.last_model_update_skip_index is None


def test_residual_short_and_long_models_run_on_the_same_update_schedule(monkeypatch):
    estimator = ResidualWeightedWLSEstimator(
        regression_method="price", model_lookback_bars=10,
        short_model_lookback_bars=5, model_update_interval_bars=3,
        position_update_policy="update",
    )
    state = EstimatorState()
    calls = []
    original = estimator._observation_weights

    def recording_weights(x_arr, y_arr):
        calls.append(len(x_arr))
        return original(x_arr, y_arr)

    monkeypatch.setattr(estimator, "_observation_weights", recording_weights)
    rng = np.random.default_rng(101)
    for index in range(13):
        x = float(index + 1)
        y = 1.0 + 2.0 * x + float(rng.normal(scale=0.1))
        estimator.update(state, x, y, index)

    assert calls == [10, 10]
    assert state.last_model_update_index == 12


def test_ols_and_tls_use_different_error_directions():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 4.0])

    ols_alpha, ols_beta = RollingOLSEstimator._ols(x, y)
    tls_alpha, tls_beta = TLSEstimator._tls(x, y)

    assert ols_beta == pytest.approx(1.5)
    assert tls_beta == pytest.approx(1.5387619780)
    assert tls_beta != pytest.approx(ols_beta)
    assert tls_alpha == pytest.approx(np.mean(y) - tls_beta * np.mean(x))
    assert ols_alpha == pytest.approx(np.mean(y) - ols_beta * np.mean(x))


def test_dols_recovers_long_run_beta_and_drops_unavailable_edge_rows():
    rng = np.random.default_rng(7)
    x = 100.0 + np.cumsum(rng.normal(size=120))
    delta_x = np.diff(x)
    y = 2.0 + 1.3 * x
    for t in range(2, len(x) - 1):
        y[t] += 0.4 * delta_x[t - 2] - 0.2 * delta_x[t - 1] + 0.3 * delta_x[t]

    estimator = DOLSEstimator(regression_method="price", dols_lead_lag_order=1)
    alpha, beta = estimator._dols(x, y)
    changed_edge_y = y.copy()
    changed_edge_y[-1] += 1_000_000.0
    edge_alpha, edge_beta = estimator._dols(x, changed_edge_y)

    assert alpha == pytest.approx(2.0, abs=1e-9)
    assert beta == pytest.approx(1.3, abs=1e-9)
    assert edge_alpha == pytest.approx(alpha)
    assert edge_beta == pytest.approx(beta)


def test_ewls_half_life_weights_and_beta():
    estimator = EWLSEstimator(regression_method="price", ewls_half_life_bars=1.0)
    weights = estimator._weights(4)
    alpha, beta = estimator._ewls(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([1.0, 2.0, 3.0, 8.0]),
    )

    assert weights.tolist() == pytest.approx([0.125, 0.25, 0.5, 1.0])
    assert beta == pytest.approx(2.8144329897)
    assert alpha == pytest.approx(-3.7938144330)


def test_huber_reduces_outlier_influence_and_handles_zero_mad():
    x = np.arange(50, dtype=float)
    clean_y = 1.0 + 2.0 * x
    contaminated_y = clean_y.copy()
    contaminated_y[-1] += 500.0
    estimator = HuberEstimator(regression_method="price")

    _, ols_beta = RollingOLSEstimator._ols(x, contaminated_y)
    alpha, huber_beta = estimator._huber(x, contaminated_y)
    exact_alpha, exact_beta = estimator._huber(x, clean_y)

    assert abs(huber_beta - 2.0) < abs(ols_beta - 2.0)
    assert alpha == pytest.approx(1.0, abs=1e-3)
    assert exact_alpha == pytest.approx(1.0, abs=1e-8)
    assert exact_beta == pytest.approx(2.0, abs=1e-8)


def test_huber_runs_configured_iteration_count():
    x = np.arange(30, dtype=float)
    y = 1.0 + 2.0 * x
    y[-1] += 100.0
    estimator = HuberEstimator(regression_method="price", huber_iterations=3)

    estimator._huber(x, y)

    assert estimator.last_iterations == 3


@pytest.mark.parametrize("estimator_class", ESTIMATORS)
def test_each_estimator_initializes_then_becomes_tradable_next_bar(estimator_class):
    rng = np.random.default_rng(11)
    x = 100.0 + np.cumsum(rng.normal(scale=0.3, size=61))
    residual = np.zeros(61)
    for index in range(1, len(residual)):
        residual[index] = 0.45 * residual[index - 1] + rng.normal(scale=0.1)
    y = 2.0 + 1.5 * x + residual
    estimator = estimator_class(
        pair_id="pair",
        regression_method="price",
        model_lookback_bars=60,
        warmup_bars=60,
        model_update_interval_bars=100,
        residual_adf_min_samples=20,
        residual_adf_max_pvalue=0.40,
    )
    state = EstimatorState()

    outputs = [estimator.update(state, x[i], y[i], i) for i in range(60)]
    tradable = estimator.update(state, x[60], y[60], 60)

    assert all(not output.ready for output in outputs)
    assert state.spread_beta is not None
    assert tradable.ready
    assert tradable.beta == pytest.approx(outputs[-1].beta)
    assert "residual ADF" in state.cointegration_reason
    assert state.cointegration_pvalue is not None


@pytest.mark.parametrize("estimator_class", ESTIMATORS)
def test_each_estimator_supports_log_return_regression(estimator_class):
    rng = np.random.default_rng(19)
    x_returns = rng.normal(loc=0.0002, scale=0.01, size=41)
    y_returns = 0.001 + 1.7 * x_returns
    x_prices = 100.0 * np.exp(np.r_[0.0, np.cumsum(x_returns)])
    y_prices = 80.0 * np.exp(np.r_[0.0, np.cumsum(y_returns)])
    estimator = estimator_class(
        pair_id="return_pair",
        regression_method="log_return",
        model_lookback_bars=40,
        warmup_bars=40,
        model_update_interval_bars=100,
        residual_adf_min_samples=10,
    )
    state = EstimatorState()

    initialization_outputs = [
        estimator.update(state, x_prices[index], y_prices[index], index)
        for index in range(41)
    ]
    tradable = estimator.update(state, x_prices[41], y_prices[41], 41)

    assert estimator.warmup_bars == 41
    assert all(not output.ready for output in initialization_outputs)
    assert tradable.ready
    assert tradable.alpha == pytest.approx(0.001, abs=1e-9)
    assert tradable.beta == pytest.approx(1.7, abs=1e-9)
    assert tradable.latest_spread == pytest.approx(0.0, abs=1e-9)
    assert tradable.spread_sample_count == 40


@pytest.mark.parametrize("estimator_class", RETURN_INTERVAL_ESTIMATORS)
def test_return_interval_uses_non_overlapping_returns_ending_at_latest_close(estimator_class):
    minute_returns = np.arange(1.0, 63.0) / 10_000.0
    prices = 100.0 * np.exp(np.r_[0.0, np.cumsum(minute_returns)])
    estimator = estimator_class(
        regression_method="log_return",
        return_interval_bars=5,
        model_lookback_bars=62,
    )

    sampled = estimator._sample_log_returns(prices)
    expected = np.asarray([
        minute_returns[start:start + 5].sum()
        for start in range(2, 62, 5)
    ])

    assert len(sampled) == 12
    assert sampled == pytest.approx(expected)


@pytest.mark.parametrize("estimator_class", RETURN_INTERVAL_ESTIMATORS)
def test_one_bar_return_interval_matches_original_log_return(estimator_class):
    prices = np.array([100.0, 101.0, 99.0, 102.0, 104.0, 103.0,
                       105.0, 108.0, 107.0, 109.0, 111.0])
    estimator = estimator_class(
        regression_method="log_return",
        return_interval_bars=1,
        model_lookback_bars=10,
    )

    assert estimator._sample_log_returns(prices) == pytest.approx(
        np.diff(np.log(prices))
    )


@pytest.mark.parametrize("estimator_class", RETURN_INTERVAL_ESTIMATORS)
def test_return_interval_sets_live_rolling_return_horizon(estimator_class):
    estimator = estimator_class(
        regression_method="log_return",
        return_interval_bars=60,
        model_lookback_bars=600,
    )
    state = EstimatorState()
    state.x_close_history.extend(np.linspace(100.0, 159.0, 60))
    state.y_close_history.extend(np.linspace(200.0, 259.0, 60))

    x_return, y_return = estimator._transform(state, 160.0, 260.0)

    assert x_return == pytest.approx(np.log(160.0 / 100.0))
    assert y_return == pytest.approx(np.log(260.0 / 200.0))

    state.alpha = 0.0
    state.beta = state.spread_beta = 1.0
    state.spread_mean = 0.0
    state.spread_std = 1.0
    state.next_model_update_index = 1_000
    estimator.update(state, 160.0, 260.0, 0)
    assert estimator.latest_return_values == pytest.approx((x_return, y_return))

    estimator.update(state, 161.0, 261.0, 1)
    assert estimator.latest_return_values == pytest.approx((
        np.log(161.0 / 101.0),
        np.log(261.0 / 201.0),
    ))


@pytest.mark.parametrize("estimator_class", RETURN_INTERVAL_ESTIMATORS)
def test_return_interval_fit_uses_expected_sample_count_and_activates_next_bar(estimator_class):
    rng = np.random.default_rng(71)
    x_returns = rng.normal(loc=0.0001, scale=0.002, size=601)
    y_returns = 1.8 * x_returns
    y_returns[-1] += 0.10
    x_prices = 100.0 * np.exp(np.r_[0.0, np.cumsum(x_returns)])
    y_prices = 80.0 * np.exp(np.r_[0.0, np.cumsum(y_returns)])
    estimator = estimator_class(
        pair_id="interval_pair",
        regression_method="log_return",
        return_interval_bars=60,
        model_lookback_bars=600,
        model_update_interval_bars=1_000,
    )
    state = EstimatorState()

    initialization = [
        estimator.update(state, x_prices[index], y_prices[index], index)
        for index in range(estimator.warmup_bars)
    ]
    tradable = estimator.update(
        state,
        x_prices[estimator.warmup_bars],
        y_prices[estimator.warmup_bars],
        estimator.warmup_bars,
    )

    assert estimator.warmup_bars == 601
    assert all(not output.ready for output in initialization)
    # Fit, residual normalization, and live signal all use 60-minute returns.
    assert state.spread_sample_count == 10
    assert tradable.ready
    assert tradable.beta == pytest.approx(1.8)
    assert tradable.latest_spread == pytest.approx(0.10)
    assert tradable.para["estimator"]["return_interval_bars"] == 60


@pytest.mark.parametrize("estimator_class", RETURN_INTERVAL_ESTIMATORS)
def test_return_interval_uses_same_return_horizon_for_residual_scale(estimator_class):
    rng = np.random.default_rng(91)
    x_returns = rng.normal(0.0, 0.002, 600)
    block_noise = np.asarray([
        -0.012, 0.008, -0.004, 0.014, -0.009,
        0.003, 0.011, -0.006, 0.005, -0.010,
    ])
    minute_noise = np.repeat(block_noise / 60.0, 60)
    y_returns = 1.5 * x_returns + minute_noise
    x_prices = 100.0 * np.exp(np.r_[0.0, np.cumsum(x_returns)])
    y_prices = 80.0 * np.exp(np.r_[0.0, np.cumsum(y_returns)])
    estimator = estimator_class(
        pair_id="minute_scale",
        regression_method="log_return",
        return_interval_bars=60,
        model_lookback_bars=600,
        model_update_interval_bars=1_000,
    )
    state = EstimatorState()

    for index in range(estimator.warmup_bars):
        estimator.update(state, x_prices[index], y_prices[index], index)

    sampled_x, sampled_y = estimator._prepare_xy(state)
    expected_residuals = sampled_y - (state.alpha + state.beta * sampled_x)

    assert state.spread_sample_count == 10
    assert state.spread_std == pytest.approx(
        np.std(expected_residuals, ddof=1), rel=1e-9
    )
    assert state.spread_std != pytest.approx(np.std(minute_noise, ddof=1), rel=1e-3)


@pytest.mark.parametrize("estimator_class", RETURN_INTERVAL_ESTIMATORS)
def test_return_interval_rejects_invalid_or_insufficient_configuration(estimator_class):
    with pytest.raises(ValueError, match="positive integer"):
        estimator_class(
            regression_method="log_return",
            return_interval_bars=0,
            model_lookback_bars=600,
        )
    with pytest.raises(ValueError, match="positive integer"):
        estimator_class(
            regression_method="log_return",
            return_interval_bars=2.5,
            model_lookback_bars=600,
        )
    with pytest.raises(ValueError, match="at least 10 complete return intervals"):
        estimator_class(
            regression_method="log_return",
            return_interval_bars=60,
            model_lookback_bars=599,
        )


def test_residual_adf_threshold_controls_pass_and_open_block(monkeypatch):
    def fake_adfuller(*args, **kwargs):
        return -1.0, 0.30, 0, len(args[0]) - 1, {"1%": -3.0, "5%": -2.0, "10%": -1.5}, None

    monkeypatch.setattr("statsmodels.tsa.stattools.adfuller", fake_adfuller)
    residuals = np.linspace(-1.0, 1.0, 30)

    permissive = TLSEstimator(residual_adf_max_pvalue=0.40, residual_adf_min_samples=10)
    permissive_state = EstimatorState()
    permissive._update_residual_adf(permissive_state, residuals, 10)

    strict = TLSEstimator(residual_adf_max_pvalue=0.20, residual_adf_min_samples=10)
    strict_state = EstimatorState()
    strict._update_residual_adf(strict_state, residuals, 10)

    assert permissive_state.cointegration_pass
    assert not permissive_state.cointegration_block_open
    assert not strict_state.cointegration_pass
    assert strict_state.cointegration_block_open
    blocked = MultiPairStrategy._cointegration_entry_decision(
        type("Result", (), {"estimator": EstimatorOutput(
            pair_id="pair",
            cointegration_block_open=True,
            cointegration_pvalue=0.30,
            cointegration_reason=strict_state.cointegration_reason,
        )})()
    )
    assert not blocked["allowed"]


def _feed_initial_relation(estimator, state):
    for index in range(10):
        estimator.update(state, float(index + 1), 1.0 + 2.0 * float(index + 1), index)


@pytest.mark.parametrize("estimator_class", ESTIMATORS)
def test_freeze_policy_skips_and_waits_a_full_update_cycle(estimator_class):
    estimator = estimator_class(
        regression_method="price",
        model_lookback_bars=10,
        warmup_bars=10,
        model_update_interval_bars=3,
        position_update_policy="freeze",
        residual_adf_min_samples=10,
    )
    state = EstimatorState()
    _feed_initial_relation(estimator, state)
    initial_beta = state.spread_beta
    for index in (10, 11):
        estimator.update(state, float(index + 1), 1.0 + 4.0 * float(index + 1), index)

    skipped = estimator.update(
        state, 13.0, 53.0, 12, para={"position": {"has_position": True}}
    )
    estimator.update(state, 14.0, 57.0, 13, para={"position": {"has_position": False}})
    estimator.update(state, 15.0, 61.0, 14, para={"position": {"has_position": False}})

    assert skipped.beta == pytest.approx(initial_beta)
    assert state.last_model_update_index == 9
    assert state.next_model_update_index == 15

    due_output = estimator.update(state, 16.0, 65.0, 15, para={"position": {"has_position": False}})
    updated_beta = state.spread_beta
    next_output = estimator.update(state, 17.0, 69.0, 16)

    assert due_output.beta == pytest.approx(initial_beta)
    assert updated_beta != pytest.approx(initial_beta)
    assert next_output.beta == pytest.approx(updated_beta)


@pytest.mark.parametrize("estimator_class", ESTIMATORS)
def test_update_policy_refits_on_schedule_while_position_is_open(estimator_class):
    estimator = estimator_class(
        regression_method="price",
        model_lookback_bars=10,
        warmup_bars=10,
        model_update_interval_bars=3,
        position_update_policy="update",
        residual_adf_min_samples=10,
    )
    state = EstimatorState()
    _feed_initial_relation(estimator, state)
    initial_beta = state.spread_beta
    estimator.update(state, 11.0, 45.0, 10)
    estimator.update(state, 12.0, 49.0, 11)

    due_output = estimator.update(
        state, 13.0, 53.0, 12, para={"position": {"has_position": True}}
    )
    updated_beta = state.spread_beta
    next_output = estimator.update(
        state, 14.0, 57.0, 13, para={"position": {"has_position": True}}
    )

    assert state.last_model_update_index == 12
    assert due_output.beta == pytest.approx(initial_beta)
    assert updated_beta != pytest.approx(initial_beta)
    assert next_output.beta == pytest.approx(updated_beta)
