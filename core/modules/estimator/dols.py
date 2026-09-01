"""Scheduled rolling-window dynamic OLS estimator."""
from __future__ import annotations

from math import isfinite, log

import numpy as np
from numpy.linalg import LinAlgError

from core.modules.models.pipeline_types import EstimatorOutput, EstimatorState, register_estimator
from core.modules.data.rolling_window import ensure_deque, tail_values


@register_estimator("dols")
class DOLSEstimator:
    """Estimate long-run beta after absorbing leads/lags of delta X."""

    method_name = "dols"

    def __init__(
        self,
        pair_id: str = "",
        regression_method: str = "log_price",
        model_lookback_bars: int = 10080,
        model_update_interval_bars: int = 240,
        position_update_policy: str = "freeze",
        residual_adf_max_pvalue: float = 0.40,
        residual_adf_maxlag: int = 1,
        residual_adf_regression: str = "c",
        dols_lead_lag_order: int = 1,
        **kwargs,
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
        if not 0.0 <= self.residual_adf_max_pvalue <= 1.0:
            raise ValueError("residual_adf_max_pvalue must be between 0 and 1")
        self.residual_adf_maxlag = max(0, int(residual_adf_maxlag))
        self.residual_adf_regression = str(residual_adf_regression or "c")
        self.dols_lead_lag_order = max(0, int(dols_lead_lag_order))
        self.collect_diagnostics = True

    def update(self, state, x_close, y_close, bar_index, para=None):
        has_position = bool(((para or {}).get("position") or {}).get("has_position", False))
        state.x_close_history = ensure_deque(state.x_close_history, self.warmup_bars)
        state.y_close_history = ensure_deque(state.y_close_history, self.warmup_bars)
        ready = self._model_ready(state)
        latest_spread = None
        if ready:
            x_t, y_t = self._transform(state, x_close, y_close)
            latest_spread = float(y_t - (float(state.alpha or 0.0) + float(state.spread_beta) * x_t))
        output = self._output(state, bar_index, ready, latest_spread, has_position)

        state.x_close_history.append(float(x_close))
        state.y_close_history.append(float(y_close))
        if len(state.x_close_history) < self.warmup_bars:
            output.ready = False
            output.reason = f"warmup {len(state.x_close_history)}/{self.warmup_bars}"
            if self.collect_diagnostics:
                output.para["estimator"]["phase"] = "warmup"
            return output

        if state.spread_beta is None:
            fitted = self._fit_model(state, bar_index)
            self._sync_output(output, state)
            output.ready = False
            output.reason = "dols initialized; first tradable bar is next bar" if fitted else "dols initialization failed"
            if self.collect_diagnostics:
                output.para["estimator"].update({
                    "phase": "initialized" if fitted else "init_failed",
                    "residual_adf": self._adf_para(state),
                })
            return output

        if self._update_due(state, bar_index):
            if has_position and self.position_update_policy == "freeze":
                state.last_model_update_skip_index = bar_index
                state.next_model_update_index = bar_index + self.model_update_interval_bars
                output.next_model_update_index = state.next_model_update_index
                output.last_model_update_skip_index = bar_index
                if self.collect_diagnostics:
                    output.para["estimator"].update({
                        "model_update_skipped": True,
                        "model_update_skip_reason": "position open",
                        "next_model_update_at": state.next_model_update_index,
                    })
            elif self._fit_model(state, bar_index):
                output.next_model_update_index = state.next_model_update_index
                output.last_model_update_skip_index = state.last_model_update_skip_index
                if self.collect_diagnostics:
                    output.para["estimator"].update({
                        "refit_for_next_bar": True,
                        "next_alpha": state.alpha,
                        "next_beta": state.spread_beta,
                        "residual_adf": self._adf_para(state),
                    })
        return output

    def _fit_model(self, state, bar_index):
        x_arr, y_arr = self._prepare_xy(state)
        if x_arr is None:
            return False
        try:
            alpha, beta = self._dols(x_arr, y_arr)
        except (LinAlgError, ValueError):
            return False
        residuals = y_arr - (alpha + beta * x_arr)
        residuals = residuals[np.isfinite(residuals)]
        if len(residuals) < 2:
            return False
        self._save_fit(state, alpha, beta, residuals, bar_index)
        return True

    def _dols(self, x_arr, y_arr):
        order = self.dols_lead_lag_order
        delta_x = np.diff(x_arr)
        indices = np.arange(1 + order, len(x_arr) - order)
        if len(indices) <= 2 * order + 2:
            raise ValueError("insufficient DOLS design rows")
        differences = np.column_stack([
            delta_x[indices + offset - 1]
            for offset in range(-order, order + 1)
        ])
        design = np.column_stack([
            np.ones(len(indices)),
            x_arr[indices],
            differences,
        ])
        coefficients = np.linalg.lstsq(design, y_arr[indices], rcond=None)[0]
        return float(coefficients[0]), float(coefficients[1])

    def _save_fit(self, state, alpha, beta, residuals, bar_index):
        mean = float(np.mean(residuals))
        std = float(np.std(residuals, ddof=1))
        if not isfinite(std) or std <= 1e-12:
            std = 1e-8
        state.alpha, state.beta, state.spread_beta = float(alpha), float(beta), float(beta)
        state.spread_mean, state.spread_std, state.spread_var = mean, std, std * std
        state.spread_sample_count = len(residuals)
        state.spread_history = ensure_deque(residuals.tolist(), self.model_lookback_bars)
        state.spread_rolling_sum = float(np.sum(residuals))
        state.spread_rolling_sumsq = float(np.sum(residuals * residuals))
        state.zscore_history = ensure_deque(
            ((residuals - mean) / std).tolist(), self.model_lookback_bars
        )
        state.last_model_update_index = bar_index
        state.next_model_update_index = bar_index + self.model_update_interval_bars
        state.last_model_update_skip_index = None
        self._update_residual_adf(state, residuals, bar_index)
        self._update_half_life(state)

    def _update_residual_adf(self, state, residuals, bar_index):
        samples = len(residuals)
        passed = False
        stat = pvalue = critical_value = None
        if float(np.std(residuals)) <= 1e-12:
            reason = "residual ADF failed: constant residuals"
        else:
            try:
                from statsmodels.tsa.stattools import adfuller

                result = adfuller(
                    residuals,
                    maxlag=min(self.residual_adf_maxlag, max(0, samples // 2 - 2)),
                    regression=self.residual_adf_regression,
                    autolag=None,
                )
                stat, pvalue = float(result[0]), float(result[1])
                key = "1%" if self.residual_adf_max_pvalue <= 0.01 else (
                    "5%" if self.residual_adf_max_pvalue <= 0.05 else "10%"
                )
                critical_value = float(result[4][key])
                passed = isfinite(pvalue) and pvalue <= self.residual_adf_max_pvalue
                reason = (
                    f"residual ADF pvalue={pvalue:.4f} "
                    f"{'<=' if passed else '>'} {self.residual_adf_max_pvalue:.4f}"
                )
            except Exception as exc:
                reason = f"residual ADF failed: {type(exc).__name__}"
        state.cointegration_pass = passed
        state.cointegration_pvalue, state.cointegration_stat = pvalue, stat
        state.cointegration_critical_value = critical_value
        state.cointegration_window_bars = samples
        state.cointegration_sample_count = samples
        state.cointegration_last_check_index = bar_index
        state.cointegration_reason = reason
        state.cointegration_block_open = not passed

    def _update_half_life(self, state):
        residuals = np.asarray(state.spread_history, dtype=float)
        if len(residuals) < 10 or np.std(residuals[:-1]) <= 1e-12:
            return
        phi = float(np.cov(residuals[:-1], np.diff(residuals), ddof=1)[0, 1] / np.var(residuals[:-1], ddof=1))
        if phi < 0:
            state.residual_phi = phi
            state.residual_half_life_bars = -log(2) / phi

    def _prepare_xy(self, state):
        x_arr = np.asarray(tail_values(state.x_close_history, self.price_history_bars), dtype=float)
        y_arr = np.asarray(tail_values(state.y_close_history, self.price_history_bars), dtype=float)
        if len(x_arr) < 10 or len(x_arr) != len(y_arr):
            return None, None
        if self.regression_method == "log_price":
            x_arr, y_arr = np.log(np.clip(x_arr, 1e-12, None)), np.log(np.clip(y_arr, 1e-12, None))
        elif self.regression_method in {"log_return", "log_returns"}:
            x_arr = np.diff(np.log(np.clip(x_arr, 1e-12, None)))
            y_arr = np.diff(np.log(np.clip(y_arr, 1e-12, None)))
        elif self.regression_method not in {"price", "raw_price"}:
            raise ValueError(f"unsupported regression_method: {self.regression_method}")
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_arr, y_arr = x_arr[mask], y_arr[mask]
        return (x_arr, y_arr) if len(x_arr) >= 10 else (None, None)

    def _transform(self, state, x_close, y_close):
        if self.regression_method == "log_price":
            return log(max(float(x_close), 1e-12)), log(max(float(y_close), 1e-12))
        if self.regression_method in {"log_return", "log_returns"}:
            return (
                log(max(float(x_close), 1e-12) / max(float(state.x_close_history[-1]), 1e-12)),
                log(max(float(y_close), 1e-12) / max(float(state.y_close_history[-1]), 1e-12)),
            )
        return float(x_close), float(y_close)

    def _update_due(self, state, bar_index):
        if state.next_model_update_index is not None:
            return bar_index >= state.next_model_update_index
        return state.last_model_update_index is None or bar_index - state.last_model_update_index >= self.model_update_interval_bars

    @staticmethod
    def _model_ready(state):
        return bool(state.spread_beta is not None and state.spread_mean is not None
                    and state.spread_std is not None and state.spread_std > 0)

    def _output(self, state, bar_index, ready, latest_spread, has_position):
        para = {}
        if self.collect_diagnostics:
            para = {"estimator": {
                "method": self.method_name,
                "regression_method": self.regression_method,
                "dols_lead_lag_order": self.dols_lead_lag_order,
                "position_update_policy": self.position_update_policy,
                "model_updated_at": state.last_model_update_index,
                "next_model_update_at": state.next_model_update_index,
                "has_position": has_position,
                "residual_adf": self._adf_para(state),
            }}
        return EstimatorOutput(
            pair_id=self.pair_id, bar_index=bar_index, ready=ready,
            alpha=state.alpha, beta=state.beta, spread_beta=state.spread_beta,
            hedge_beta=state.hedge_beta, spread_mean=state.spread_mean,
            spread_std=state.spread_std, spread_var=state.spread_var,
            spread_sample_count=state.spread_sample_count, latest_spread=latest_spread,
            residual_phi=state.residual_phi, residual_half_life_bars=state.residual_half_life_bars,
            cointegration_pass=state.cointegration_pass, cointegration_pvalue=state.cointegration_pvalue,
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
    def _sync_output(output, state):
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
        return {
            "pass": state.cointegration_pass, "block_open": state.cointegration_block_open,
            "pvalue": state.cointegration_pvalue, "stat": state.cointegration_stat,
            "critical_value": state.cointegration_critical_value,
            "samples": state.cointegration_sample_count,
            "last_check_index": state.cointegration_last_check_index,
            "reason": state.cointegration_reason,
        }
