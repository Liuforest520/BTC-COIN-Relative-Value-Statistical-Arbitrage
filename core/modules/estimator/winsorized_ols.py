"""Scheduled rolling-window Winsorized OLS estimator."""
from __future__ import annotations

from math import isfinite, log

import numpy as np

from core.modules.models.pipeline_types import EstimatorOutput, EstimatorState, register_estimator
from core.modules.data.rolling_window import ensure_deque, tail_values


@register_estimator("winsorized_ols")
class WinsorizedOLSEstimator:
    """Clip X and Y tails separately, then estimate beta by OLS."""

    method_name = "winsorized_ols"

    def __init__(
        self, pair_id="", regression_method="log_price", model_lookback_bars=10080,
        model_update_interval_bars=240, position_update_policy="freeze",
        residual_adf_max_pvalue=0.40,
        residual_adf_maxlag=1, residual_adf_regression="c",
        winsor_lower_quantile=0.005, winsor_upper_quantile=0.995,
        winsor_quantile_method="linear", **kwargs,
    ):
        self.pair_id = pair_id
        self.regression_method = regression_method
        self.model_lookback_bars = max(10, int(model_lookback_bars))
        self.model_update_interval_bars = max(1, int(model_update_interval_bars))
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
        self.winsor_lower_quantile = float(winsor_lower_quantile)
        self.winsor_upper_quantile = float(winsor_upper_quantile)
        if not 0 <= self.winsor_lower_quantile < self.winsor_upper_quantile <= 1:
            raise ValueError("Winsor quantiles must satisfy 0 <= lower < upper <= 1")
        self.winsor_quantile_method = str(winsor_quantile_method or "linear")
        try:
            np.quantile(
                np.array([0.0, 1.0]), 0.5, method=self.winsor_quantile_method
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"unsupported winsor_quantile_method: {self.winsor_quantile_method}"
            ) from exc

    def update(self, state, x_close, y_close, bar_index, para=None):
        has_position = bool(((para or {}).get("position") or {}).get("has_position", False))
        state.x_close_history = ensure_deque(state.x_close_history, self.warmup_bars)
        state.y_close_history = ensure_deque(state.y_close_history, self.warmup_bars)
        ready = self._ready(state)
        spread = None
        if ready:
            x_t, y_t = self._transform_current(state, x_close, y_close)
            spread = float(y_t - (float(state.alpha or 0) + float(state.spread_beta) * x_t))
        output = self._output(state, bar_index, ready, spread, has_position)
        state.x_close_history.append(float(x_close))
        state.y_close_history.append(float(y_close))
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
                "winsorized_ols initialized; first tradable bar is next bar"
                if fitted else "winsorized_ols initialization failed"
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
                        "next_beta": state.spread_beta, "residual_adf": self._adf_para(state),
                    })
        return output

    def _fit_model(self, state, bar_index):
        x_arr, y_arr = self._prepare_xy(state)
        if x_arr is None:
            return False
        try:
            alpha, beta = self._winsorized_ols(x_arr, y_arr)
        except (ValueError, np.linalg.LinAlgError):
            return False
        # Trading residuals use the observed values, not their clipped substitutes.
        residuals = y_arr - (alpha + beta * x_arr)
        self._save_fit(state, alpha, beta, residuals, bar_index)
        return True

    def _winsorized_values(self, values):
        lower, upper = np.quantile(
            values,
            [self.winsor_lower_quantile, self.winsor_upper_quantile],
            method=self.winsor_quantile_method,
        )
        return np.clip(values, lower, upper)

    def _winsorized_ols(self, x_arr, y_arr):
        x_clipped = self._winsorized_values(x_arr)
        y_clipped = self._winsorized_values(y_arr)
        x_mean, y_mean = float(np.mean(x_clipped)), float(np.mean(y_clipped))
        x_centered = x_clipped - x_mean
        denominator = float(np.dot(x_centered, x_centered))
        if denominator <= 1e-12:
            raise ValueError("Winsorized OLS requires non-zero clipped X variance")
        beta = float(np.dot(x_centered, y_clipped - y_mean) / denominator)
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
            x = np.diff(np.log(np.clip(x, 1e-12, None)))
            y = np.diff(np.log(np.clip(y, 1e-12, None)))
        elif self.regression_method not in {"price", "raw_price"}:
            raise ValueError(f"unsupported regression_method: {self.regression_method}")
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        return (x, y) if len(x) >= self.model_lookback_bars else (None, None)

    def _transform_current(self, state, x_close, y_close):
        if self.regression_method == "log_price":
            return log(max(float(x_close), 1e-12)), log(max(float(y_close), 1e-12))
        if self.regression_method in {"log_return", "log_returns"}:
            return (
                log(max(float(x_close), 1e-12) / max(float(state.x_close_history[-1]), 1e-12)),
                log(max(float(y_close), 1e-12) / max(float(state.y_close_history[-1]), 1e-12)),
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
                "model_lookback_bars": self.model_lookback_bars,
                "model_update_interval_bars": self.model_update_interval_bars,
                "winsor_lower_quantile": self.winsor_lower_quantile,
                "winsor_upper_quantile": self.winsor_upper_quantile,
                "winsor_quantile_method": self.winsor_quantile_method,
                "position_update_policy": self.position_update_policy,
                "has_position": has_position, "residual_adf": self._adf_para(state),
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
