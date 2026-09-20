"""
Recursive Least Squares estimator with exponential forgetting.
"""
from __future__ import annotations

from math import log

import numpy as np

from core.modules.models.pipeline_types import (
    EstimatorOutput,
    EstimatorState,
    register_estimator,
)
from core.modules.data.rolling_window import ensure_deque, tail_values


@register_estimator("rls")
class RLSEstimator:
    """Recursive Least Squares with forgetting factor lambda.

    Updates beta incrementally per bar — no window, no batch recomputation.
    """

    def __init__(
        self,
        pair_id: str = "",
        regression_method: str = "log_price",
        model_lookback_bars: int = 10080,
        rls_forgetting_factor: float = 0.995,
        **kwargs,
    ):
        self.pair_id = pair_id
        self.regression_method = regression_method
        if regression_method not in {"log_price", "price"}:
            raise ValueError(f"unsupported regression_method: {regression_method}")
        self.lam = rls_forgetting_factor  # 0 < lambda <= 1
        self.model_lookback_bars = max(10, int(model_lookback_bars))

    def update(self, state: EstimatorState, x_close: float, y_close: float,
               bar_index: int, para: dict | None = None) -> EstimatorOutput:
        max_history = self.model_lookback_bars * 2
        state.x_close_history = ensure_deque(state.x_close_history, max_history)
        state.y_close_history = ensure_deque(state.y_close_history, max_history)
        state.spread_history = ensure_deque(state.spread_history, max_history)
        state.x_close_history.append(x_close)
        state.y_close_history.append(y_close)

        x, y = self._transform(x_close, y_close)

        # --- Initialize if first bar ---
        if state.rls_P is None:
            state.rls_P = 1000.0 * np.eye(2)
            state.rls_theta = np.array([y - x, 0.0])
        P = np.array(state.rls_P, dtype=float)
        theta = np.array(state.rls_theta, dtype=float)

        # ---- Step 1: Predict with OLD theta ----
        old_alpha = float(theta[0])
        old_beta = float(theta[1])
        phi = np.array([1.0, x])
        y_pred = phi @ theta
        err = y - y_pred

        # Spread in transform space (log or raw) with pre-update params
        spread = y - (old_alpha + old_beta * x)
        state.spread_history.append(spread)
        lookback = min(len(state.spread_history), self.model_lookback_bars)

        # ---- Step 2: Compute stats from old state ----
        ready = False
        if lookback >= 10:
            recent = tail_values(state.spread_history, lookback)
            mean = float(np.mean(recent))
            std = float(np.std(recent, ddof=1))
            state.spread_mean = mean
            state.spread_std = max(std, 1e-8)
            state.spread_var = std * std
            state.spread_sample_count = lookback
            if lookback >= 60:
                ready = True
            if lookback >= 30:
                lagged = recent[:-1]
                diff = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
                if np.std(lagged) > 1e-12:
                    try:
                        phi_h = float(np.cov(lagged, diff, ddof=1)[0, 1] / np.var(lagged, ddof=1))
                        if phi_h < 0:
                            state.residual_phi = phi_h
                            state.residual_half_life_bars = -log(2) / phi_h
                    except (ValueError, ZeroDivisionError):
                        pass

        result = EstimatorOutput(
            pair_id=self.pair_id, bar_index=bar_index, ready=ready,
            alpha=old_alpha, beta=old_beta,
            spread_beta=old_beta, hedge_beta=old_beta,
            spread_mean=state.spread_mean, spread_std=state.spread_std,
            spread_var=state.spread_var, spread_sample_count=state.spread_sample_count,
            latest_spread=spread,
            residual_phi=state.residual_phi,
            residual_half_life_bars=state.residual_half_life_bars,
            reason="ok" if ready else "warmup",
        )

        # ---- Step 3: Update theta with current error for NEXT bar ----
        P_phi = P @ phi
        denom = self.lam + phi @ P_phi
        K = None
        if denom > 1e-12:
            K = P_phi / denom
            theta = theta + K * err
            P = (P - np.outer(K, P_phi)) / self.lam

        state.rls_P = P
        state.rls_theta = theta
        state.alpha = float(theta[0])
        state.beta = float(theta[1])
        state.spread_beta = float(theta[1])
        state.hedge_beta = float(theta[1])
        result.rls_P = P
        result.rls_theta = theta
        result.para = {
            "estimator": {
                "method": "rls",
                "innovation": float(err),
                "forgetting_factor": float(self.lam),
                "gain": K.tolist() if K is not None else None,
                "P_diag": np.diag(P).tolist(),
            }
        }

        return result

    def _transform(self, x, y):
        if self.regression_method == "log_price":
            return log(max(x, 1e-12)), log(max(y, 1e-12))
        if self.regression_method != "price":
            raise ValueError(f"unsupported regression_method: {self.regression_method}")
        return float(x), float(y)
