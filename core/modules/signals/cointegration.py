from math import log

import numpy as np

from core.modules.signals.zscore import ZScoreSignal


class CointegrationZScoreSignal(ZScoreSignal):
    def __init__(
        self,
        model_lookback_bars=10080,
        model_update_interval_bars=240,
        regression_method="log_price",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_lookback_bars = model_lookback_bars
        self.model_update_interval_bars = model_update_interval_bars
        self.regression_method = regression_method
        self.alpha = None
        self.beta = None
        self.last_model_update_index = None
        self._validate_regression_method()

    def _prepare_model(self):
        if self._should_update_model():
            self._update_model()
        return self.alpha is not None and self.beta is not None

    def _spread(self, x_close, y_close):
        return log(y_close) - self.alpha - self.beta * log(x_close)

    def _hedge_ratio(self):
        return self.beta if self.beta is not None else 1.0

    def _should_update_model(self):
        if len(self.x_closes) < self.model_lookback_bars:
            return False
        if self.last_model_update_index is None:
            return True
        return self.bar_count - self.last_model_update_index >= self.model_update_interval_bars

    def _update_model(self):
        x_prices = np.array(self.x_closes[-self.model_lookback_bars :], dtype=float)
        y_prices = np.array(self.y_closes[-self.model_lookback_bars :], dtype=float)
        log_x = np.log(x_prices)
        log_y = np.log(y_prices)
        if self.regression_method == "log_price":
            beta, alpha = np.polyfit(log_x, log_y, 1)
        elif self.regression_method == "returns":
            x_returns = np.diff(log_x)
            y_returns = np.diff(log_y)
            beta, _ = np.polyfit(x_returns, y_returns, 1)
            alpha = np.mean(log_y - beta * log_x)
        else:
            raise ValueError(f"unsupported regression_method: {self.regression_method}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.last_model_update_index = self.bar_count

    def _validate_regression_method(self):
        if self.regression_method not in {"log_price", "returns"}:
            raise ValueError("regression_method must be 'log_price' or 'returns'")

    def _record_decision(self, state, decision):
        state.update(
            {
                "alpha": self.alpha,
                "beta": self.beta,
                "regression_method": self.regression_method,
                "last_model_update_index": self.last_model_update_index,
            }
        )
        return super()._record_decision(state, decision)
