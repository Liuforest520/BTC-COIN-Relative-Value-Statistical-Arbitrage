"""
Periodic OLS estimator.

Fit alpha/beta and residual stats on a fixed historical window, then keep those
parameters unchanged until the next scheduled refit. This is the stable baseline
for comparing against dynamic estimators such as Kalman.
"""
from __future__ import annotations

from math import isfinite, log

import numpy as np
from numpy.linalg import LinAlgError

from core.modules.models.pipeline_types import (
    EstimatorOutput,
    EstimatorState,
    register_estimator,
)
from core.modules.data.rolling_window import ensure_deque, tail_values


@register_estimator("periodic_ols")
class PeriodicOLSEstimator:
    """Fixed-window OLS with scheduled refits.

    Data flow per bar:
      1. Use the current saved alpha/beta/mean/std to compute this bar's spread.
      2. Emit output for signal generation.
      3. Append this bar and refit only if the update interval is reached.

    The first fit happens after warmup and is not tradable until the next bar, so
    the strategy does not use the same bar both to fit and to trade.
    """

    def __init__(
        self,
        pair_id: str = "",
        regression_method: str = "log_price",
        model_lookback_bars: int = 86400,
        model_update_interval_bars: int = 1440,
        hedge_model_lookback_bars: int | None = None,
        hedge_model_update_interval_bars: int | None = None,
        hedge_regression_method: str | None = None,
        **kwargs,
    ):
        self.pair_id = pair_id
        self.regression_method = regression_method
        if regression_method not in {"log_price", "price"}:
            raise ValueError(f"unsupported regression_method: {regression_method}")
        self.model_lookback_bars = int(model_lookback_bars)
        self.model_update_interval_bars = max(1, int(model_update_interval_bars))
        self.warmup_bars = self.model_lookback_bars

        self.hedge_model_lookback_bars = int(hedge_model_lookback_bars or self.model_lookback_bars)
        self.hedge_model_update_interval_bars = max(
            1,
            int(hedge_model_update_interval_bars or self.model_update_interval_bars),
        )
        self.hedge_regression_method = hedge_regression_method or regression_method
        if self.hedge_regression_method not in {"log_price", "price"}:
            raise ValueError(
                f"unsupported hedge_regression_method: {self.hedge_regression_method}"
            )

    def update(
        self,
        state: EstimatorState,
        x_close: float,
        y_close: float,
        bar_index: int,
        para: dict | None = None,
    ) -> EstimatorOutput:
        max_history = max(
            self.warmup_bars,
            self.model_lookback_bars,
            self.hedge_model_lookback_bars,
        )
        state.x_close_history = ensure_deque(state.x_close_history, max_history)
        state.y_close_history = ensure_deque(state.y_close_history, max_history)
        state.spread_history = ensure_deque(state.spread_history, self.model_lookback_bars)
        state.x_close_history.append(float(x_close))
        state.y_close_history.append(float(y_close))

        if state.spread_beta is None:
            if len(state.x_close_history) < self.warmup_bars:
                return EstimatorOutput(
                    pair_id=self.pair_id,
                    bar_index=bar_index,
                    ready=False,
                    reason=f"periodic_ols warmup {len(state.x_close_history)}/{self.warmup_bars}",
                    para={
                        "estimator": {
                            "method": "periodic_ols",
                            "phase": "warmup",
                            "warmup_bars": self.warmup_bars,
                            "samples": len(state.x_close_history),
                        }
                    },
                )
            fitted = self._fit_signal_model(state, bar_index)
            self._fit_hedge_model(state, bar_index)
            return EstimatorOutput(
                pair_id=self.pair_id,
                bar_index=bar_index,
                ready=False,
                alpha=state.alpha,
                beta=state.beta,
                spread_beta=state.spread_beta,
                hedge_beta=state.hedge_beta,
                spread_mean=state.spread_mean,
                spread_std=state.spread_std,
                spread_var=state.spread_var,
                spread_sample_count=state.spread_sample_count,
                last_model_update_index=state.last_model_update_index,
                last_hedge_model_update_index=state.last_hedge_model_update_index,
                reason=(
                    "periodic_ols initialized; first tradable bar is next bar"
                    if fitted
                    else "periodic_ols initialization failed"
                ),
                para={
                    "estimator": {
                        "method": "periodic_ols",
                        "phase": "initialized" if fitted else "init_failed",
                        "model_lookback_bars": self.model_lookback_bars,
                        "model_update_interval_bars": self.model_update_interval_bars,
                    }
                },
            )

        x_t, y_t = self._transform_pair(x_close, y_close, self.regression_method)
        alpha = float(state.alpha or 0.0)
        beta = float(state.spread_beta)
        spread = float(y_t - (alpha + beta * x_t))

        ready = (
            state.spread_mean is not None
            and state.spread_std is not None
            and state.spread_std > 0
            and state.spread_sample_count >= max(2, min(self.model_lookback_bars, len(state.x_close_history)))
        )

        result = EstimatorOutput(
            pair_id=self.pair_id,
            bar_index=bar_index,
            ready=ready,
            alpha=state.alpha,
            beta=state.beta,
            spread_beta=state.spread_beta,
            hedge_beta=state.hedge_beta,
            spread_mean=state.spread_mean,
            spread_std=state.spread_std,
            spread_var=state.spread_var,
            spread_sample_count=state.spread_sample_count,
            latest_spread=spread,
            residual_phi=state.residual_phi,
            residual_half_life_bars=state.residual_half_life_bars,
            last_model_update_index=state.last_model_update_index,
            last_hedge_model_update_index=state.last_hedge_model_update_index,
            reason="ok" if ready else "model not ready",
            para={
                "estimator": {
                    "method": "periodic_ols",
                    "phase": "trade" if ready else "not_ready",
                    "model_updated_at": state.last_model_update_index,
                    "hedge_model_updated_at": state.last_hedge_model_update_index,
                    "model_update_interval_bars": self.model_update_interval_bars,
                    "raw_spread": spread,
                }
            },
        )

        state.spread_history.append(spread)
        if self._should_refit(state.last_model_update_index, bar_index, self.model_update_interval_bars):
            if self._fit_signal_model(state, bar_index):
                result.para["estimator"]["refit_for_next_bar"] = True
                result.para["estimator"]["next_alpha"] = state.alpha
                result.para["estimator"]["next_beta"] = state.spread_beta
        if self._should_refit(
            state.last_hedge_model_update_index,
            bar_index,
            self.hedge_model_update_interval_bars,
        ):
            self._fit_hedge_model(state, bar_index)

        return result

    @staticmethod
    def _should_refit(last_update_idx: int | None, bar_idx: int, interval: int) -> bool:
        if last_update_idx is None:
            return True
        return bar_idx - last_update_idx >= interval

    def _fit_signal_model(self, state: EstimatorState, bar_index: int) -> bool:
        x_arr, y_arr = self._prepare_xy(state, self.regression_method, self.model_lookback_bars)
        if len(x_arr) < max(2, self.model_lookback_bars):
            return False
        try:
            alpha, beta = self._ols(x_arr, y_arr)
        except (LinAlgError, ValueError):
            return False

        residuals = y_arr - (alpha + beta * x_arr)
        residuals = residuals[np.isfinite(residuals)]
        if len(residuals) < 2:
            return False

        mean = float(np.mean(residuals))
        std = float(np.std(residuals, ddof=1))
        if not isfinite(std) or std <= 1e-12:
            std = 1e-8

        state.alpha = float(alpha)
        state.beta = float(beta)
        state.spread_beta = float(beta)
        state.spread_mean = mean
        state.spread_std = std
        state.spread_var = float(std * std)
        state.spread_sample_count = int(len(residuals))
        state.last_model_update_index = bar_index
        state.spread_history = ensure_deque(
            residuals.tolist()[-self.model_lookback_bars:], self.model_lookback_bars
        )
        if state.spread_std > 0:
            state.zscore_history = ensure_deque(
                [(float(s) - state.spread_mean) / state.spread_std for s in state.spread_history],
                self.model_lookback_bars,
            )
        return True

    def _fit_hedge_model(self, state: EstimatorState, bar_index: int) -> bool:
        x_arr, y_arr = self._prepare_xy(
            state,
            self.hedge_regression_method,
            self.hedge_model_lookback_bars,
        )
        if len(x_arr) < max(2, self.hedge_model_lookback_bars):
            return False
        try:
            _alpha, beta = self._ols(x_arr, y_arr)
        except (LinAlgError, ValueError):
            return False
        state.hedge_beta = float(beta)
        state.last_hedge_model_update_index = bar_index
        return True

    def _prepare_xy(self, state: EstimatorState, method: str, lookback: int):
        x_vals = tail_values(state.x_close_history, lookback)
        y_vals = tail_values(state.y_close_history, lookback)
        if len(x_vals) != len(y_vals) or len(x_vals) < 2:
            return np.array([]), np.array([])

        x_arr = np.array(x_vals, dtype=float)
        y_arr = np.array(y_vals, dtype=float)
        if method == "log_price":
            x_arr = np.log(np.clip(x_arr, 1e-12, None))
            y_arr = np.log(np.clip(y_arr, 1e-12, None))
        elif method == "price":
            pass
        else:
            raise ValueError(f"unsupported regression_method: {method}")

        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        return x_arr[mask], y_arr[mask]

    @staticmethod
    def _ols(x_arr, y_arr):
        X = np.column_stack([np.ones(len(x_arr)), x_arr])
        coeff = np.linalg.lstsq(X, y_arr, rcond=None)[0]
        return float(coeff[0]), float(coeff[1])

    @staticmethod
    def _transform_pair(x_close: float, y_close: float, method: str):
        if method == "log_price":
            return log(max(float(x_close), 1e-12)), log(max(float(y_close), 1e-12))
        return float(x_close), float(y_close)
