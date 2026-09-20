import numpy as np
import pytest
from types import SimpleNamespace

from core.modules.estimator import (
    DOLSEstimator,
    EWLSEstimator,
    HuberEstimator,
    KalmanEstimator,
    PeriodicOLSEstimator,
    RLSEstimator,
    RollingOLSEstimator,
    TLSEstimator,
    WinsorizedOLSEstimator,
)
from core.modules.models.pipeline_types import ESTIMATOR_REGISTRY, EstimatorOutput, EstimatorState
from core.modules.strategy.multi_pair_strategy import MultiPairStrategy
from core.modules.strategy.factory import build_strategy


ESTIMATORS = [RollingOLSEstimator, TLSEstimator, DOLSEstimator, EWLSEstimator, HuberEstimator]
ADDITIONAL_ESTIMATORS = [WinsorizedOLSEstimator]
ALL_ESTIMATORS = [
    DOLSEstimator,
    EWLSEstimator,
    HuberEstimator,
    KalmanEstimator,
    PeriodicOLSEstimator,
    RLSEstimator,
    RollingOLSEstimator,
    TLSEstimator,
    WinsorizedOLSEstimator,
]


@pytest.mark.parametrize("estimator_class", ALL_ESTIMATORS)
def test_estimators_reject_unknown_regression_method(estimator_class):
    with pytest.raises(ValueError, match="unsupported regression_method"):
        estimator_class(regression_method="unsupported")


def test_five_beta_estimators_are_registered_independently():
    assert {"rolling_ols", "tls", "dols", "ewls", "huber"} <= set(ESTIMATOR_REGISTRY)
    assert len({ESTIMATOR_REGISTRY[name] for name in ("rolling_ols", "tls", "dols", "ewls", "huber")}) == 5


def test_winsorized_estimator_is_registered():
    assert ESTIMATOR_REGISTRY["winsorized_ols"] is WinsorizedOLSEstimator


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
                "model_lookback_bars": 10,
            },
            "signal": {"method": "zscore"},
            "sizing": {"method": "beta_neutral"},
            "portfolio": {"method": "equal_weight"},
            "execution": {},
        },
    )

    strategy = build_strategy(setup, {"X": {}, "Y": {}})

    assert isinstance(strategy.pipelines["pair"].estimator, estimator_class)


def test_factory_maps_legacy_warmup_bars_to_model_lookback():
    setup = SimpleNamespace(
        pairs=[{"pair_id": "pair", "x_symbol": "X", "y_symbol": "Y"}],
        pipeline={
            "estimator": {
                "method": "tls", "regression_method": "price",
                "warmup_bars": 12,
            },
            "signal": {"method": "zscore"},
            "sizing": {"method": "beta_neutral"},
            "portfolio": {"method": "equal_weight"},
            "execution": {},
        },
    )

    strategy = build_strategy(setup, {"X": {}, "Y": {}})

    assert strategy.pipelines["pair"].estimator.model_lookback_bars == 12


@pytest.mark.parametrize(
    ("method", "estimator_class"),
    [("winsorized_ols", WinsorizedOLSEstimator)],
)
def test_strategy_factory_constructs_each_additional_estimator(method, estimator_class):
    setup = SimpleNamespace(
        pairs=[{"pair_id": "pair", "x_symbol": "X", "y_symbol": "Y"}],
        pipeline={
            "estimator": {
                "method": method, "regression_method": "log_price",
                "model_lookback_bars": 10,
            },
            "signal": {"method": "zscore"}, "sizing": {"method": "beta_neutral"},
            "portfolio": {"method": "equal_weight"}, "execution": {},
        },
    )

    strategy = build_strategy(setup, {"X": {}, "Y": {}})

    assert isinstance(strategy.pipelines["pair"].estimator, estimator_class)


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


@pytest.mark.parametrize("estimator_class", ADDITIONAL_ESTIMATORS)
@pytest.mark.parametrize("policy", ["freeze", "update"])
def test_additional_estimators_apply_position_update_policy(estimator_class, policy):
    estimator = estimator_class(
        regression_method="price", model_lookback_bars=10,
        model_update_interval_bars=3, position_update_policy=policy,
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
