from math import log

import numpy as np

from core.modules.logger import logger
from core.modules.signals.zscore import ZScoreSignal


class CointegrationZScoreSignal(ZScoreSignal):
    def __init__(
        self,
        model_lookback_bars=10080,
        model_update_interval_bars=240,
        regression_method="log_price",
        deming_delta=1.0,
        hedge_model_lookback_bars=None,
        hedge_model_update_interval_bars=1,
        hedge_regression_method=None,
        hedge_deming_delta=None,
        beta_stability_enabled=False,
        beta_stability_lookback_bars=1440,
        beta_stability_max_cv=0.10,
        beta_stability_min_samples=240,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_lookback_bars = int(model_lookback_bars)
        self.model_update_interval_bars = int(model_update_interval_bars)
        self.regression_method = regression_method
        self.deming_delta = float(deming_delta)
        self.hedge_model_lookback_bars = int(hedge_model_lookback_bars or model_lookback_bars)
        self.hedge_model_update_interval_bars = int(hedge_model_update_interval_bars)
        self.hedge_regression_method = hedge_regression_method or regression_method
        self.hedge_deming_delta = float(hedge_deming_delta if hedge_deming_delta is not None else self.deming_delta)
        self.beta_stability_enabled = bool(beta_stability_enabled)
        self.beta_stability_lookback_bars = int(beta_stability_lookback_bars)
        self.beta_stability_max_cv = float(beta_stability_max_cv)
        self.beta_stability_min_samples = int(beta_stability_min_samples)
        self.alpha = None
        self.beta = None
        self.spread_beta = None
        self.hedge_beta = None
        self.last_model_update_index = None
        self.last_hedge_model_update_index = None
        self.hedge_beta_history = []
        self.beta_stability_state = {}
        self._validate_regression_method()
        self._validate_regression_method(self.hedge_regression_method)
        self._validate_positive_number(self.deming_delta, "deming_delta")
        self._validate_positive_number(self.hedge_deming_delta, "hedge_deming_delta")
        self._validate_positive_int(self.model_lookback_bars, "model_lookback_bars")
        self._validate_positive_int(self.model_update_interval_bars, "model_update_interval_bars")
        self._validate_positive_int(self.hedge_model_lookback_bars, "hedge_model_lookback_bars")
        self._validate_positive_int(self.hedge_model_update_interval_bars, "hedge_model_update_interval_bars")
        self._validate_positive_int(self.beta_stability_lookback_bars, "beta_stability_lookback_bars")
        self._validate_positive_int(self.beta_stability_min_samples, "beta_stability_min_samples")
        self._validate_positive_number(self.beta_stability_max_cv, "beta_stability_max_cv")
        requested_warmup_bars = int(self.warmup_bars)
        self.warmup_bars = max(requested_warmup_bars, self.model_lookback_bars, self.hedge_model_lookback_bars)
        if self.warmup_bars != requested_warmup_bars:
            logger.warning(
                "warmup_bars {} is shorter than model lookbacks; using {}",
                requested_warmup_bars,
                self.warmup_bars,
            )

    def _prepare_model(self):
        if self._should_update_hedge_model():
            self._update_hedge_model()
        if self._should_update_model():
            self._update_model()
        return self.alpha is not None and self.spread_beta is not None and self.hedge_beta is not None

    def _spread(self, x_close, y_close):
        if self._uses_return_residual(self.regression_method):
            if len(self.x_closes) < 2 or len(self.y_closes) < 2:
                return None
            x_prev = self.x_closes[-2]
            y_prev = self.y_closes[-2]
            if x_prev <= 0 or y_prev <= 0:
                return None
            x_return = log(x_close) - log(x_prev)
            y_return = log(y_close) - log(y_prev)
            return y_return - self.spread_beta * x_return
        return log(y_close) - self.alpha - self.spread_beta * log(x_close)

    def _hedge_ratio(self):
        return self.hedge_beta if self.hedge_beta is not None else 1.0

    def _should_update_model(self):
        if len(self.x_closes) <= self.model_lookback_bars:
            return False
        if self.last_model_update_index is None:
            return True
        return self.bar_count - self.last_model_update_index >= self.model_update_interval_bars

    def _should_update_hedge_model(self):
        if len(self.x_closes) <= self.hedge_model_lookback_bars:
            return False
        if self.last_hedge_model_update_index is None:
            return True
        return self.bar_count - self.last_hedge_model_update_index >= self.hedge_model_update_interval_bars

    def _update_model(self):
        beta_alpha = self._fit_window(
            lookback_bars=self.model_lookback_bars,
            regression_method=self.regression_method,
            deming_delta=self.deming_delta,
            include_alpha=True,
        )
        if beta_alpha is None:
            return
        beta, alpha = beta_alpha
        if not np.isfinite(alpha) or not np.isfinite(beta):
            return
        self.alpha = float(alpha)
        self.spread_beta = float(beta)
        self.beta = self.spread_beta
        self.last_model_update_index = self.bar_count

    def _update_hedge_model(self):
        beta = self._fit_window(
            lookback_bars=self.hedge_model_lookback_bars,
            regression_method=self.hedge_regression_method,
            deming_delta=self.hedge_deming_delta,
            include_alpha=False,
        )
        if beta is None or not np.isfinite(beta):
            return
        self.hedge_beta = float(beta)
        self.hedge_beta_history.append(self.hedge_beta)
        self.last_hedge_model_update_index = self.bar_count

    def _fit_window(self, lookback_bars, regression_method, deming_delta, include_alpha):
        x_prices = np.array(self.x_closes[-lookback_bars - 1 : -1], dtype=float)
        y_prices = np.array(self.y_closes[-lookback_bars - 1 : -1], dtype=float)
        if (
            len(x_prices) < lookback_bars
            or len(y_prices) < lookback_bars
            or not np.isfinite(x_prices).all()
            or not np.isfinite(y_prices).all()
            or (x_prices <= 0).any()
            or (y_prices <= 0).any()
        ):
            return None
        log_x = np.log(x_prices)
        log_y = np.log(y_prices)
        beta_alpha = self._fit_beta_alpha(log_x, log_y, regression_method, deming_delta)
        if beta_alpha is None:
            return None
        beta, alpha = beta_alpha
        if include_alpha:
            return beta, alpha
        return beta

    def _fit_beta_alpha(self, log_x, log_y, regression_method=None, deming_delta=None):
        regression_method = regression_method or self.regression_method
        deming_delta = self.deming_delta if deming_delta is None else float(deming_delta)
        try:
            if regression_method == "log_price":
                beta, alpha = np.polyfit(log_x, log_y, 1)
            elif regression_method == "returns":
                x_returns = np.diff(log_x)
                y_returns = np.diff(log_y)
                beta = self._ols_slope(x_returns, y_returns)
                if beta is None:
                    return None
                alpha = 0.0
            elif regression_method == "deming_returns":
                x_returns = np.diff(log_x)
                y_returns = np.diff(log_y)
                beta = self._deming_slope(x_returns, y_returns, deming_delta)
                if beta is None:
                    return None
                alpha = 0.0
            else:
                raise ValueError(f"unsupported regression_method: {regression_method}")
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return None
        return beta, alpha

    def _ols_slope(self, x_values, y_values):
        x_values = np.array(x_values, dtype=float)
        y_values = np.array(y_values, dtype=float)
        valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[valid_mask]
        y_values = y_values[valid_mask]
        if len(x_values) < 2:
            return None
        denominator = float(np.dot(x_values, x_values))
        if not np.isfinite(denominator) or denominator <= 1e-24:
            return None
        beta = float(np.dot(x_values, y_values) / denominator)
        if not np.isfinite(beta):
            return None
        return beta

    def _deming_slope(self, x_values, y_values, deming_delta=None):
        deming_delta = self.deming_delta if deming_delta is None else float(deming_delta)
        x_values = np.array(x_values, dtype=float)
        y_values = np.array(y_values, dtype=float)
        valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[valid_mask]
        y_values = y_values[valid_mask]
        if len(x_values) < 3:
            return None

        x_centered = x_values - float(x_values.mean())
        y_centered = y_values - float(y_values.mean())
        sample_count = len(x_values) - 1
        s_xx = float(np.dot(x_centered, x_centered) / sample_count)
        s_yy = float(np.dot(y_centered, y_centered) / sample_count)
        s_xy = float(np.dot(x_centered, y_centered) / sample_count)
        if (
            not np.isfinite(s_xx)
            or not np.isfinite(s_yy)
            or not np.isfinite(s_xy)
            or s_xx <= 0
            or s_yy <= 0
            or abs(s_xy) < 1e-12
        ):
            return None

        numerator = s_yy - deming_delta * s_xx
        discriminant = numerator * numerator + 4.0 * deming_delta * s_xy * s_xy
        if not np.isfinite(discriminant):
            return None
        return (numerator + np.sqrt(max(discriminant, 0.0))) / (2.0 * s_xy)

    def _entry_filter(self, side, zscore):
        if not self.beta_stability_enabled:
            self.beta_stability_state = {"enabled": False}
            self._entry_filter_block_reason = None
            return True

        passed, state = self._beta_stability_check()
        self.beta_stability_state = state
        self._entry_filter_block_reason = None if passed else state.get("reason", "beta unstable")
        return passed

    def _beta_stability_check(self):
        recent = np.array(self.hedge_beta_history[-self.beta_stability_lookback_bars :], dtype=float)
        recent = recent[np.isfinite(recent)]
        if len(recent) < self.beta_stability_min_samples:
            return False, {
                "enabled": True,
                "passed": False,
                "sample_count": int(len(recent)),
                "reason": "beta stability warmup",
            }

        beta_mean = float(recent.mean())
        beta_std = float(recent.std(ddof=1))
        if not np.isfinite(beta_mean) or not np.isfinite(beta_std) or abs(beta_mean) < 1e-12:
            return False, {
                "enabled": True,
                "passed": False,
                "sample_count": int(len(recent)),
                "beta_mean": beta_mean,
                "beta_std": beta_std,
                "reason": "invalid beta stability statistics",
            }

        beta_cv = beta_std / abs(beta_mean)
        passed = beta_cv <= self.beta_stability_max_cv
        return passed, {
            "enabled": True,
            "passed": passed,
            "sample_count": int(len(recent)),
            "beta_mean": beta_mean,
            "beta_std": beta_std,
            "beta_cv": beta_cv,
            "max_cv": self.beta_stability_max_cv,
            "reason": "beta stable" if passed else "beta unstable",
        }

    def _validate_regression_method(self, regression_method=None):
        regression_method = regression_method or self.regression_method
        if regression_method not in {"log_price", "returns", "deming_returns"}:
            raise ValueError("regression_method must be 'log_price', 'returns', or 'deming_returns'")

    def _uses_return_residual(self, regression_method):
        return regression_method in {"returns", "deming_returns"}

    def _validate_positive_number(self, value, name):
        if not np.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be a positive finite number")

    def _validate_positive_int(self, value, name):
        if int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")

    def _record_decision(self, state, decision):
        if self.beta_stability_enabled:
            _, beta_state = self._beta_stability_check()
            self.beta_stability_state = beta_state
        else:
            beta_state = {"enabled": False}
            self.beta_stability_state = beta_state
        state.update(
            {
                "alpha": self.alpha,
                "beta": self.beta,
                "spread_beta": self.spread_beta,
                "hedge_beta": self.hedge_beta,
                "regression_method": self.regression_method,
                "spread_residual_method": "returns" if self._uses_return_residual(self.regression_method) else "log_price",
                "deming_delta": self.deming_delta,
                "hedge_regression_method": self.hedge_regression_method,
                "hedge_deming_delta": self.hedge_deming_delta,
                "last_model_update_index": self.last_model_update_index,
                "last_hedge_model_update_index": self.last_hedge_model_update_index,
                "beta_stability_enabled": beta_state.get("enabled", self.beta_stability_enabled),
                "beta_stability_passed": beta_state.get("passed"),
                "beta_stability_sample_count": beta_state.get("sample_count"),
                "beta_stability_mean": beta_state.get("beta_mean"),
                "beta_stability_std": beta_state.get("beta_std"),
                "beta_stability_cv": beta_state.get("beta_cv"),
                "beta_stability_max_cv": beta_state.get("max_cv", self.beta_stability_max_cv),
                "beta_stability_reason": beta_state.get("reason"),
            }
        )
        return super()._record_decision(state, decision)
