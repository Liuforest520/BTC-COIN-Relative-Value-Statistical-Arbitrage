"""Scheduled residual-shock-weighted rolling WLS estimator."""
from __future__ import annotations

from collections import deque
from math import isfinite, log

import numpy as np

from core.modules.models.pipeline_types import EstimatorOutput, EstimatorState, register_estimator
from core.modules.data.rolling_window import ensure_deque, tail_values


@register_estimator("residual_weighted_wls")
class ResidualWeightedWLSEstimator:
    """Use a short current relationship to weight a longer final WLS fit."""

    method_name = "residual_weighted_wls"
    PROFILE_WEIGHTS = {
        "mild": (1.00, 0.90, 0.75, 0.50),
        "medium": (1.00, 0.75, 0.50, 0.25),
        "strong": (1.00, 0.60, 0.30, 0.10),
    }

    def __init__(
        self, pair_id="", regression_method="log_price", return_interval_bars=1,
        model_lookback_bars=10080,
        model_update_interval_bars=240, position_update_policy="freeze",
        residual_adf_max_pvalue=0.40,
        residual_adf_maxlag=1, residual_adf_regression="c",
        residual_wls_profile="medium", short_model_lookback_bars=None,
        shock_score_boundaries=None, shock_weights=None,
        residual_wls_scale_lookback_bars=None, **kwargs,
    ):
        self.pair_id = pair_id
        self.regression_method = regression_method
        try:
            self.return_interval_bars = int(return_interval_bars)
        except (TypeError, ValueError) as exc:
            raise ValueError("return_interval_bars must be a positive integer") from exc
        if (
            isinstance(return_interval_bars, bool)
            or self.return_interval_bars != return_interval_bars
            or self.return_interval_bars <= 0
        ):
            raise ValueError("return_interval_bars must be a positive integer")
        self.model_lookback_bars = max(10, int(model_lookback_bars))
        if (
            regression_method in {"log_return", "log_returns"}
            and self.model_lookback_bars // self.return_interval_bars < 10
        ):
            raise ValueError(
                "model_lookback_bars must contain at least 10 complete return intervals"
            )
        self.model_update_interval_bars = max(1, int(model_update_interval_bars))
        if short_model_lookback_bars is None:
            short_model_lookback_bars = residual_wls_scale_lookback_bars
        if short_model_lookback_bars is None:
            short_model_lookback_bars = min(10080, self.model_lookback_bars)
        self.short_model_lookback_bars = max(3, int(short_model_lookback_bars))
        if self.short_model_lookback_bars > self.model_lookback_bars:
            raise ValueError(
                "short_model_lookback_bars must not exceed model_lookback_bars"
            )
        if (
            regression_method in {"log_return", "log_returns"}
            and self.short_model_lookback_bars // self.return_interval_bars < 3
        ):
            raise ValueError(
                "short_model_lookback_bars must contain at least 3 complete return intervals"
            )
        self.price_history_bars = self.model_lookback_bars + (
            1 if regression_method in {"log_return", "log_returns"} else 0
        )
        self.warmup_bars = self.price_history_bars
        if position_update_policy not in {"update", "freeze"}:
            raise ValueError("position_update_policy must be 'update' or 'freeze'")
        self.position_update_policy = position_update_policy
        self.residual_adf_max_pvalue = float(residual_adf_max_pvalue)
        if not 0 <= self.residual_adf_max_pvalue <= 1:
            raise ValueError("residual_adf_max_pvalue must be between 0 and 1")
        self.residual_adf_maxlag = max(0, int(residual_adf_maxlag))
        self.residual_adf_regression = str(residual_adf_regression or "c")
        self.residual_wls_profile = str(residual_wls_profile).lower()
        if shock_weights is None:
            if self.residual_wls_profile not in self.PROFILE_WEIGHTS:
                raise ValueError("residual_wls_profile must be mild, medium, or strong")
            shock_weights = self.PROFILE_WEIGHTS[self.residual_wls_profile]
        if shock_score_boundaries is None:
            shock_score_boundaries = (1.5, 2.0, 3.0)
        boundaries = tuple(float(value) for value in shock_score_boundaries)
        weights = tuple(float(value) for value in shock_weights)
        if any(not isfinite(value) or value <= 0 for value in boundaries):
            raise ValueError("shock_score_boundaries must be finite and positive")
        if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
            raise ValueError("shock_score_boundaries must be strictly increasing")
        if len(weights) != len(boundaries) + 1:
            raise ValueError("shock_weights must have one more value than the boundaries")
        if any(not isfinite(value) or value <= 0 for value in weights):
            raise ValueError("shock_weights must be finite and positive")
        self.shock_score_boundaries = boundaries
        self.shock_weights = weights
        # Backward-compatible names used by old reports/configuration.
        self.residual_wls_scale_lookback_bars = self.short_model_lookback_bars
        self.residual_wls_weights = self.shock_weights
        self.last_observation_weights = np.array([], dtype=float)
        self.last_shock_scores = np.array([], dtype=float)
        self.last_short_alpha = None
        self.last_short_beta = None
        self.last_short_residual_std = None
        self._return_lag_state = None
        self._x_return_lag = deque(maxlen=self.return_interval_bars)
        self._y_return_lag = deque(maxlen=self.return_interval_bars)
        self.latest_return_values = None
        self.collect_diagnostics = True

    def update(self, state, x_close, y_close, bar_index, para=None):
        has_position = bool(((para or {}).get("position") or {}).get("has_position", False))
        state.x_close_history = ensure_deque(state.x_close_history, self.warmup_bars)
        state.y_close_history = ensure_deque(state.y_close_history, self.warmup_bars)
        self._sync_return_lag(state)
        self.latest_return_values = None
        ready = self._ready(state)
        spread = None
        if ready:
            x_t, y_t = self._transform_current(state, x_close, y_close)
            if self.regression_method in {"log_return", "log_returns"}:
                self.latest_return_values = (x_t, y_t)
            spread = float(y_t - (float(state.alpha or 0) + float(state.spread_beta) * x_t))
        output = self._output(state, bar_index, ready, spread, has_position)
        state.x_close_history.append(float(x_close))
        state.y_close_history.append(float(y_close))
        if self.regression_method in {"log_return", "log_returns"}:
            self._x_return_lag.append(float(x_close))
            self._y_return_lag.append(float(y_close))
        if len(state.x_close_history) < self.warmup_bars:
            output.ready = False
            output.reason = f"warmup {len(state.x_close_history)}/{self.warmup_bars}"
            if getattr(self, "collect_diagnostics", True):
                output.para["estimator"]["phase"] = "warmup"
            return output
        if state.spread_beta is None:
            fitted = self._fit_model(state, bar_index)
            self._sync(output, state)
            output.ready = False
            output.reason = (
                "residual_weighted_wls initialized; first tradable bar is next bar"
                if fitted else "residual_weighted_wls initialization failed"
            )
            if getattr(self, "collect_diagnostics", True):
                output.para["estimator"].update({
                    "phase": "initialized" if fitted else "init_failed",
                    "residual_adf": self._adf_para(state),
                })
            return output
        if self._due(state, bar_index):
            if has_position and self.position_update_policy == "freeze":
                state.last_model_update_skip_index = bar_index
                state.next_model_update_index = bar_index + self.model_update_interval_bars
                output.next_model_update_index = state.next_model_update_index
                output.last_model_update_skip_index = bar_index
                if getattr(self, "collect_diagnostics", True):
                    output.para["estimator"]["model_update_skipped"] = True
            elif self._fit_model(state, bar_index):
                output.next_model_update_index = state.next_model_update_index
                output.last_model_update_skip_index = state.last_model_update_skip_index
                if getattr(self, "collect_diagnostics", True):
                    output.para["estimator"].update({
                        "refit_for_next_bar": True, "next_alpha": state.alpha,
                        "next_beta": state.spread_beta,
                        "residual_adf": self._adf_para(state),
                    })
        return output

    def _fit_model(self, state, bar_index):
        x_arr, y_arr = self._prepare_xy(state)
        if x_arr is None:
            return False
        try:
            weights, scores = self._observation_weights(x_arr, y_arr)
            target_sample_count = self._model_sample_count()
            target_x = x_arr[-target_sample_count:]
            target_y = y_arr[-target_sample_count:]
            alpha, beta = self._weighted_ols(target_x, target_y, weights)
        except (ValueError, np.linalg.LinAlgError):
            return False
        self.last_observation_weights, self.last_shock_scores = weights, scores
        residuals = target_y - (alpha + beta * target_x)
        self._save_fit(state, alpha, beta, residuals, bar_index)
        return True

    def _observation_weights(self, x_arr, y_arr):
        target_sample_count = self._model_sample_count()
        short_sample_count = self._short_sample_count()
        x_arr = np.asarray(x_arr, dtype=float)[-target_sample_count:]
        y_arr = np.asarray(y_arr, dtype=float)[-target_sample_count:]
        if len(x_arr) < target_sample_count or len(x_arr) != len(y_arr):
            raise ValueError("insufficient history for residual weights")

        short_x = x_arr[-short_sample_count:]
        short_y = y_arr[-short_sample_count:]
        short_alpha, short_beta = self._weighted_ols(
            short_x, short_y, np.ones(len(short_x), dtype=float)
        )
        short_residuals = short_y - (short_alpha + short_beta * short_x)
        short_std = float(np.std(short_residuals, ddof=1))
        if not isfinite(short_std) or short_std <= 1e-12:
            short_std = 1e-8

        scores = np.abs(y_arr - (short_alpha + short_beta * x_arr)) / short_std
        self.last_short_alpha = float(short_alpha)
        self.last_short_beta = float(short_beta)
        self.last_short_residual_std = float(short_std)
        return self._weights_from_scores(scores), scores

    # Compatibility shim for callers of the previous implementation.
    def _causal_observation_weights(self, x_arr, y_arr):
        return self._observation_weights(x_arr, y_arr)

    def _weights_from_scores(self, scores):
        scores = np.asarray(scores, dtype=float)
        weights = np.full(len(scores), self.shock_weights[0], dtype=float)
        finite = np.isfinite(scores)
        band_indices = np.searchsorted(
            np.asarray(self.shock_score_boundaries), scores[finite], side="left"
        )
        weights[finite] = np.asarray(self.shock_weights)[band_indices]
        return weights

    @staticmethod
    def _weighted_ols(x_arr, y_arr, weights):
        total = float(np.sum(weights))
        x_mean = float(np.dot(weights, x_arr) / total)
        y_mean = float(np.dot(weights, y_arr) / total)
        x_centered, y_centered = x_arr - x_mean, y_arr - y_mean
        denominator = float(np.dot(weights, x_centered * x_centered))
        if denominator <= 1e-12:
            raise ValueError("residual-weighted WLS requires non-zero weighted X variance")
        beta = float(np.dot(weights, x_centered * y_centered) / denominator)
        return y_mean - beta * x_mean, beta

    def _save_fit(self, state, alpha, beta, residuals, bar_index):
        residuals = np.asarray(residuals, dtype=float)
        residuals = residuals[np.isfinite(residuals)]
        mean, std = float(np.mean(residuals)), float(np.std(residuals, ddof=1))
        if not isfinite(std) or std <= 1e-12:
            std = 1e-8
        state.alpha, state.beta, state.spread_beta = float(alpha), float(beta), float(beta)
        state.spread_mean, state.spread_std, state.spread_var = mean, std, std * std
        state.spread_sample_count = len(residuals)
        state.spread_history = ensure_deque(residuals.tolist(), self.model_lookback_bars)
        state.zscore_history = ensure_deque(
            ((residuals - mean) / std).tolist(), self.model_lookback_bars
        )
        state.last_model_update_index = bar_index
        state.next_model_update_index = bar_index + self.model_update_interval_bars
        state.last_model_update_skip_index = None
        self._update_adf(state, residuals, bar_index)
        self._update_half_life(state)

    def _update_adf(self, state, residuals, bar_index):
        samples, passed = len(residuals), False
        stat = pvalue = critical = None
        if float(np.std(residuals)) <= 1e-12:
            reason = "residual ADF failed: constant residuals"
        else:
            try:
                from statsmodels.tsa.stattools import adfuller
                result = adfuller(
                    residuals, maxlag=min(self.residual_adf_maxlag, max(0, samples // 2 - 2)),
                    regression=self.residual_adf_regression, autolag=None,
                )
                stat, pvalue = float(result[0]), float(result[1])
                key = "1%" if self.residual_adf_max_pvalue <= .01 else (
                    "5%" if self.residual_adf_max_pvalue <= .05 else "10%"
                )
                critical = float(result[4][key])
                passed = isfinite(pvalue) and pvalue <= self.residual_adf_max_pvalue
                reason = f"residual ADF pvalue={pvalue:.4f} {'<=' if passed else '>'} {self.residual_adf_max_pvalue:.4f}"
            except Exception as exc:
                reason = f"residual ADF failed: {type(exc).__name__}"
        state.cointegration_pass, state.cointegration_block_open = passed, not passed
        state.cointegration_pvalue, state.cointegration_stat = pvalue, stat
        state.cointegration_critical_value = critical
        state.cointegration_window_bars = state.cointegration_sample_count = samples
        state.cointegration_last_check_index, state.cointegration_reason = bar_index, reason

    def _update_half_life(self, state):
        residuals = np.asarray(state.spread_history, dtype=float)
        if len(residuals) < 10 or np.std(residuals[:-1]) <= 1e-12:
            return
        phi = float(np.cov(residuals[:-1], np.diff(residuals), ddof=1)[0, 1] / np.var(residuals[:-1], ddof=1))
        if phi < 0:
            state.residual_phi, state.residual_half_life_bars = phi, -log(2) / phi

    def _prepare_xy(self, state):
        x = np.asarray(tail_values(state.x_close_history, self.price_history_bars), dtype=float)
        y = np.asarray(tail_values(state.y_close_history, self.price_history_bars), dtype=float)
        if len(x) < self.price_history_bars or len(x) != len(y):
            return None, None
        if self.regression_method == "log_price":
            x, y = np.log(np.clip(x, 1e-12, None)), np.log(np.clip(y, 1e-12, None))
        elif self.regression_method in {"log_return", "log_returns"}:
            x = self._sample_log_returns(x)
            y = self._sample_log_returns(y)
        elif self.regression_method not in {"price", "raw_price"}:
            raise ValueError(f"unsupported regression_method: {self.regression_method}")
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        return (x, y) if len(x) >= self._model_sample_count() else (None, None)

    def _model_sample_count(self):
        if self.regression_method in {"log_return", "log_returns"}:
            return self.model_lookback_bars // self.return_interval_bars
        return self.model_lookback_bars

    def _short_sample_count(self):
        if self.regression_method in {"log_return", "log_returns"}:
            return self.short_model_lookback_bars // self.return_interval_bars
        return self.short_model_lookback_bars

    def _sample_log_returns(self, prices):
        prices = np.asarray(prices, dtype=float)
        complete_intervals = (len(prices) - 1) // self.return_interval_bars
        first_endpoint = len(prices) - 1 - complete_intervals * self.return_interval_bars
        sampled_prices = prices[first_endpoint::self.return_interval_bars]
        return np.diff(np.log(np.clip(sampled_prices, 1e-12, None)))

    def _sync_return_lag(self, state):
        if self.regression_method not in {"log_return", "log_returns"}:
            return
        if self._return_lag_state is state:
            return
        self._return_lag_state = state
        self._x_return_lag = deque(
            tail_values(state.x_close_history, self.return_interval_bars),
            maxlen=self.return_interval_bars,
        )
        self._y_return_lag = deque(
            tail_values(state.y_close_history, self.return_interval_bars),
            maxlen=self.return_interval_bars,
        )

    def _transform_current(self, state, x_close, y_close):
        if self.regression_method == "log_price":
            return log(max(float(x_close), 1e-12)), log(max(float(y_close), 1e-12))
        if self.regression_method in {"log_return", "log_returns"}:
            if len(self._x_return_lag) >= self.return_interval_bars:
                base_x = float(self._x_return_lag[0])
                base_y = float(self._y_return_lag[0])
            elif len(state.x_close_history) >= self.return_interval_bars:
                base_x = float(state.x_close_history[-self.return_interval_bars])
                base_y = float(state.y_close_history[-self.return_interval_bars])
            else:
                raise ValueError("insufficient price history for return interval")
            return (
                log(max(float(x_close), 1e-12) / max(base_x, 1e-12)),
                log(max(float(y_close), 1e-12) / max(base_y, 1e-12)),
            )
        return float(x_close), float(y_close)

    def _due(self, state, bar_index):
        return bar_index >= state.next_model_update_index if state.next_model_update_index is not None else True

    @staticmethod
    def _ready(state):
        return state.spread_beta is not None and state.spread_std is not None and state.spread_std > 0

    def _output(self, state, bar_index, ready, spread, has_position):
        para = {}
        if getattr(self, "collect_diagnostics", True):
            para = {"estimator": {
                "method": self.method_name, "regression_method": self.regression_method,
                "return_interval_bars": self.return_interval_bars,
                "model_lookback_bars": self.model_lookback_bars,
                "model_update_interval_bars": self.model_update_interval_bars,
                "residual_wls_profile": self.residual_wls_profile,
                "short_model_lookback_bars": self.short_model_lookback_bars,
                "shock_score_boundaries": list(self.shock_score_boundaries),
                "shock_weights": list(self.shock_weights),
                "position_update_policy": self.position_update_policy,
                "has_position": has_position,
                "residual_adf": self._adf_para(state),
            }}
        return EstimatorOutput(
            pair_id=self.pair_id, bar_index=bar_index, ready=ready, alpha=state.alpha,
            beta=state.beta, spread_beta=state.spread_beta, hedge_beta=state.hedge_beta,
            spread_mean=state.spread_mean, spread_std=state.spread_std,
            spread_var=state.spread_var, spread_sample_count=state.spread_sample_count,
            latest_spread=spread, residual_phi=state.residual_phi,
            residual_half_life_bars=state.residual_half_life_bars,
            cointegration_pass=state.cointegration_pass,
            cointegration_pvalue=state.cointegration_pvalue,
            cointegration_stat=state.cointegration_stat,
            cointegration_critical_value=state.cointegration_critical_value,
            cointegration_window_bars=state.cointegration_window_bars,
            cointegration_sample_count=state.cointegration_sample_count,
            cointegration_last_check_index=state.cointegration_last_check_index,
            cointegration_reason=state.cointegration_reason,
            cointegration_block_open=state.cointegration_block_open,
            last_model_update_index=state.last_model_update_index,
            next_model_update_index=state.next_model_update_index,
            last_model_update_skip_index=state.last_model_update_skip_index,
            reason="ok" if ready else "model not ready", para=para,
        )

    @staticmethod
    def _sync(output, state):
        for name in (
            "alpha", "beta", "spread_beta", "spread_mean", "spread_std", "spread_var",
            "spread_sample_count", "cointegration_pass", "cointegration_pvalue",
            "cointegration_stat", "cointegration_critical_value", "cointegration_window_bars",
            "cointegration_sample_count", "cointegration_last_check_index", "cointegration_reason",
            "cointegration_block_open", "last_model_update_index", "next_model_update_index",
            "last_model_update_skip_index",
        ):
            setattr(output, name, getattr(state, name))

    @staticmethod
    def _adf_para(state):
        return {"pass": state.cointegration_pass, "block_open": state.cointegration_block_open,
                "pvalue": state.cointegration_pvalue, "stat": state.cointegration_stat,
                "reason": state.cointegration_reason}
