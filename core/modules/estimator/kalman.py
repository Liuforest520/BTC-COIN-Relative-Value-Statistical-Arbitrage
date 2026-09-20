"""
Kalman Filter estimator for dynamic hedge ratio tracking.

State:  [alpha, beta]   -- random walk
Obs:    y = alpha + beta * x + noise
"""
from __future__ import annotations

from math import isfinite, log

import numpy as np

from core.modules.models.pipeline_types import (
    EstimatorOutput,
    EstimatorState,
    register_estimator,
)
from core.modules.data.rolling_window import ensure_deque, tail_values


@register_estimator("kalman")
class KalmanEstimator:
    """Kalman filter for dynamic hedge-ratio estimation.

    State transition:
        [alpha, beta]_t = [alpha, beta]_{t-1} + N(0, Q)

    Observation:
        y_t = alpha_t + beta_t * x_t + N(0, R)
    """

    def __init__(
        self,
        pair_id: str = "",
        regression_method: str = "log_price",
        model_lookback_bars: int = 10080,
        kalman_delta: float = 1e-5,
        kalman_v0: float = 0.1,
        kalman_q_alpha: float | None = None,
        kalman_q_beta: float | None = None,
        kalman_r_multiplier: float = 1.0,
        spread_std_floor: float | None = None,
        zscore_cap: float | None = None,
        **kwargs,
    ):
        self.pair_id = pair_id
        self.regression_method = regression_method
        if regression_method not in {"log_price", "price"}:
            raise ValueError(f"unsupported regression_method: {regression_method}")
        self.delta = kalman_delta   # Q = delta * I (state noise)
        self.v0 = kalman_v0         # initial P scale
        self.q_alpha = float(kalman_delta if kalman_q_alpha is None else kalman_q_alpha)
        self.q_beta = float(kalman_delta if kalman_q_beta is None else kalman_q_beta)
        self.r_multiplier = max(float(kalman_r_multiplier), 1e-12)
        self.spread_std_floor = (
            None if spread_std_floor is None else max(float(spread_std_floor), 0.0)
        )
        self.zscore_cap = None if zscore_cap is None else abs(float(zscore_cap))

        self.Q = np.diag([self.q_alpha, self.q_beta])
        self.R = None

        self.model_lookback_bars = max(1, int(model_lookback_bars))
        self.r_ewma_alpha = 2.0 / (max(2, self.model_lookback_bars) + 1.0)
        self.warmup_bars = self.model_lookback_bars

    def update(self, state: EstimatorState, x_close: float, y_close: float,
               bar_index: int, para: dict | None = None) -> EstimatorOutput:
        max_history = self.model_lookback_bars * 2
        state.x_close_history = ensure_deque(state.x_close_history, max_history)
        state.y_close_history = ensure_deque(state.y_close_history, max_history)
        state.spread_history = ensure_deque(state.spread_history, self.model_lookback_bars)
        state.x_close_history.append(x_close)
        state.y_close_history.append(y_close)

        x, y = self._transform(x_close, y_close)

        if state.kalman_P is None:
            initialized = self._initialize_from_warmup(state, bar_index)
            if not initialized:
                return EstimatorOutput(
                    pair_id=self.pair_id,
                    bar_index=bar_index,
                    ready=False,
                    reason=(
                        f"kalman warmup {len(state.x_close_history)}/"
                        f"{self.warmup_bars}"
                    ),
                    para={
                        "estimator": {
                            "method": "kalman",
                            "phase": "warmup",
                            "warmup_bars": self.warmup_bars,
                            "samples": len(state.x_close_history),
                        }
                    },
                )

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
                residual_phi=state.residual_phi,
                residual_half_life_bars=state.residual_half_life_bars,
                kalman_P=state.kalman_P,
                kalman_theta=state.kalman_theta,
                reason="kalman initialized; first tradable bar is next bar",
                para={
                    "estimator": {
                        "method": "kalman",
                        "phase": "initialized",
                        "warmup_bars": self.warmup_bars,
                        "P_diag": np.diag(np.asarray(state.kalman_P, dtype=float)).tolist(),
                        "Q_diag": np.diag(np.asarray(state.kalman_Q, dtype=float)).tolist(),
                        "R": float(state.kalman_R),
                    }
                },
            )

        P = np.asarray(state.kalman_P, dtype=float)
        theta = np.asarray(state.kalman_theta, dtype=float)
        Q = np.asarray(state.kalman_Q if state.kalman_Q is not None else self.Q, dtype=float)
        R = float(state.kalman_R if state.kalman_R is not None else 1.0)

        # ---- Step 1: Predict with OLD parameters ----
        old_alpha = float(theta[0])
        old_beta = float(theta[1])
        H = np.array([1.0, x])
        y_pred = H @ theta
        err = y - y_pred  # innovation

        # Spread in transform space (log or raw) — matches regression domain
        spread = y - (old_alpha + old_beta * x)
        self._append_spread(state, spread)
        lookback = state.spread_sample_count

        # ---- Step 2: Compute z-score from old state ----
        ready = lookback >= self.model_lookback_bars
        signal_spread = self._signal_spread(spread, state.spread_mean, state.spread_std)

        # Build output with PRE-update state (z-score isn't dampened by current bar)
        result = EstimatorOutput(
            pair_id=self.pair_id, bar_index=bar_index, ready=ready,
            alpha=old_alpha, beta=old_beta,
            spread_beta=old_beta, hedge_beta=old_beta,
            spread_mean=state.spread_mean, spread_std=state.spread_std,
            spread_var=state.spread_var, spread_sample_count=state.spread_sample_count,
            latest_spread=signal_spread,
            residual_phi=state.residual_phi,
            residual_half_life_bars=state.residual_half_life_bars,
            reason="ok" if ready else "warmup",
        )

        # ---- Step 3: Update parameters with current error for NEXT bar ----
        P_pred = P + Q
        S = H @ P_pred @ H.T + R
        K = None
        if S > 1e-12:
            K = P_pred @ H / S
            theta = theta + K * err
            P = P_pred - np.outer(K, H @ P_pred)

        next_R = self._next_observation_variance(state, R)

        # Store updated params for next call
        state.kalman_P = P
        state.kalman_theta = theta
        state.alpha = float(theta[0])
        state.beta = float(theta[1])
        state.spread_beta = float(theta[1])
        state.hedge_beta = float(theta[1])
        state.kalman_Q = Q
        state.kalman_R = next_R
        result.kalman_P = P
        result.kalman_theta = theta
        result.para = {
            "estimator": {
                "method": "kalman",
                "innovation": float(err),
                "innovation_variance": float(S),
                "kalman_gain": K.tolist() if K is not None else None,
                "P_diag": np.diag(P).tolist(),
                "Q_diag": np.diag(Q).tolist(),
                "R": float(R),
                "next_R": float(next_R),
                "raw_spread": float(spread),
                "signal_spread": float(signal_spread),
                "spread_std_floor": self.spread_std_floor,
                "zscore_cap": self.zscore_cap,
            }
        }

        return result

    def _next_observation_variance(self, state: EstimatorState, current_R: float) -> float:
        spread_var = state.spread_var
        if spread_var is None:
            return current_R
        try:
            target_R = float(spread_var) * self.r_multiplier
        except (TypeError, ValueError):
            return current_R
        if not isfinite(target_R) or target_R <= 1e-12:
            return current_R
        next_R = (1.0 - self.r_ewma_alpha) * float(current_R) + self.r_ewma_alpha * target_R
        lower = max(1e-12, float(current_R) * 0.25)
        upper = max(lower, float(current_R) * 4.0)
        next_R = min(max(next_R, lower), upper)
        return next_R

    def _initialize_from_warmup(self, state: EstimatorState, bar_index: int) -> bool:
        if len(state.x_close_history) < self.warmup_bars:
            return False

        x_arr, y_arr = self._prepare_xy(state, self.model_lookback_bars)
        if len(x_arr) < self.warmup_bars:
            return False

        X = np.column_stack([np.ones(len(x_arr)), x_arr])
        try:
            theta, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
            xtx_inv = np.linalg.pinv(X.T @ X)
        except np.linalg.LinAlgError:
            return False

        residuals = y_arr - X @ theta
        ddof = min(2, max(0, len(residuals) - 1))
        R = float(np.var(residuals, ddof=ddof)) * self.r_multiplier
        if not isfinite(R) or R <= 1e-12:
            R = 1e-8

        P = R * xtx_inv
        if not np.all(np.isfinite(P)):
            P = self.v0 * np.eye(2)

        state.kalman_theta = np.array(theta, dtype=float)
        state.kalman_P = np.array(P, dtype=float)
        state.kalman_Q = np.diag([self.q_alpha, self.q_beta])
        state.kalman_R = R
        state.kalman_initialized_bar_index = bar_index
        state.alpha = float(theta[0])
        state.beta = float(theta[1])
        state.spread_beta = float(theta[1])
        state.hedge_beta = float(theta[1])
        state.last_model_update_index = bar_index
        state.last_hedge_model_update_index = bar_index

        state.spread_history = ensure_deque(
            residuals.tolist()[-self.model_lookback_bars:],
            self.model_lookback_bars,
        )
        self._rebuild_spread_stats(state)
        if state.spread_std and state.spread_std > 0:
            state.zscore_history = ensure_deque([
                (s - state.spread_mean) / state.spread_std
                for s in state.spread_history
            ], self.model_lookback_bars)
        return True

    def _prepare_xy(self, state: EstimatorState, lookback: int):
        xs = tail_values(state.x_close_history, lookback)
        ys = tail_values(state.y_close_history, lookback)
        transformed = [self._transform(xv, yv) for xv, yv in zip(xs, ys)]
        arr = np.array(transformed, dtype=float)
        if arr.size == 0:
            return np.array([]), np.array([])
        mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
        return arr[:, 0][mask], arr[:, 1][mask]

    def _append_spread(self, state: EstimatorState, spread: float):
        spread = float(spread)
        state.spread_history = ensure_deque(state.spread_history, self.model_lookback_bars)
        removed = None
        if len(state.spread_history) == state.spread_history.maxlen:
            removed = float(state.spread_history[0])
        state.spread_history.append(spread)
        state.spread_rolling_sum += spread
        state.spread_rolling_sumsq += spread * spread

        if removed is not None:
            state.spread_rolling_sum -= removed
            state.spread_rolling_sumsq -= removed * removed

        self._update_spread_stats_from_rolling(state)

    def _rebuild_spread_stats(self, state: EstimatorState):
        state.spread_rolling_sum = float(sum(state.spread_history))
        state.spread_rolling_sumsq = float(sum(s * s for s in state.spread_history))
        self._update_spread_stats_from_rolling(state)

    def _update_spread_stats_from_rolling(self, state: EstimatorState):
        n = len(state.spread_history)
        state.spread_sample_count = n
        if n == 0:
            state.spread_mean = None
            state.spread_std = None
            state.spread_var = None
            return

        mean = state.spread_rolling_sum / n
        if n < 2:
            var = 0.0
        else:
            numerator = state.spread_rolling_sumsq - (state.spread_rolling_sum * state.spread_rolling_sum / n)
            var = max(0.0, numerator / (n - 1))
        state.spread_mean = float(mean)
        raw_std = float(var ** 0.5)
        effective_std = max(raw_std, 1e-8)
        if self.spread_std_floor is not None:
            effective_std = max(effective_std, self.spread_std_floor)
        state.spread_std = effective_std
        state.spread_var = float(state.spread_std ** 2)

    def _signal_spread(self, spread: float, mean: float | None, std: float | None) -> float:
        if self.zscore_cap is None or mean is None or std is None or std <= 0:
            return float(spread)
        zscore = (float(spread) - float(mean)) / float(std)
        if not isfinite(zscore):
            return float(spread)
        clipped = float(np.clip(zscore, -self.zscore_cap, self.zscore_cap))
        return float(mean) + clipped * float(std)

    def _transform(self, x, y):
        if self.regression_method == "log_price":
            return log(max(x, 1e-12)), log(max(y, 1e-12))
        if self.regression_method != "price":
            raise ValueError(f"unsupported regression_method: {self.regression_method}")
        return float(x), float(y)
