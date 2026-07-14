from math import isfinite, log

import numpy as np

from core.modules.signals.base import BasePairSignal, SignalDecision


class ZScoreSignal(BasePairSignal):
    def __init__(
        self,
        entry_rule_method="fixed_z",
        entry_rule_lookback_bars=10080,
        entry_rule_update_interval_bars=60,
        entry_rule_upper_percentile=97.5,
        entry_rule_lower_percentile=2.5,
        entry_rule_min_samples=None,
        entry_rule_min_abs_entry_z=None,
        entry_rule_max_abs_entry_z=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.x_closes = []
        self.y_closes = []
        self.spreads = []
        self.zscores = []
        self.entry_rule_method = entry_rule_method
        self.entry_rule_lookback_bars = int(entry_rule_lookback_bars)
        self.entry_rule_update_interval_bars = int(entry_rule_update_interval_bars)
        self.entry_rule_upper_percentile = float(entry_rule_upper_percentile)
        self.entry_rule_lower_percentile = float(entry_rule_lower_percentile)
        self.entry_rule_min_samples = int(entry_rule_min_samples or self.entry_rule_lookback_bars)
        self.entry_rule_min_abs_entry_z = (
            None if entry_rule_min_abs_entry_z is None else float(entry_rule_min_abs_entry_z)
        )
        self.entry_rule_max_abs_entry_z = (
            None if entry_rule_max_abs_entry_z is None else float(entry_rule_max_abs_entry_z)
        )
        self.entry_z_upper = float(self.entry_z)
        self.entry_z_lower = -float(self.entry_z)
        self.last_entry_rule_update_index = None
        self.entry_rule_reason = None
        self._validate_entry_rule()

    def on_bar(self, bars, position_side=None):
        ts = self._bar_ts(bars, self.x_exchange, self.x_symbol)
        x_close = self._close_price(bars, self.x_exchange, self.x_symbol)
        y_close = self._close_price(bars, self.y_exchange, self.y_symbol)
        state = {
            "ts": ts,
            "x_symbol": self.x_symbol,
            "y_symbol": self.y_symbol,
            "x_close": x_close,
            "y_close": y_close,
            "position_side": position_side,
            "entry_z": self.entry_z,
            "entry_rule_method": self.entry_rule_method,
            "exit_z": self.exit_z,
        }
        if not isfinite(x_close) or not isfinite(y_close) or x_close <= 0 or y_close <= 0:
            return self._record_decision(state, SignalDecision("none", reason="invalid price"))

        self.x_closes.append(x_close)
        self.y_closes.append(y_close)

        if self.bar_count < self.warmup_bars:
            return self._record_decision(state, SignalDecision("none", reason="warmup"))
        if not self._prepare_model():
            return self._record_decision(state, SignalDecision("none", reason="model not ready"))

        spread = self._spread(x_close, y_close)
        if spread is None:
            return self._record_decision(state, SignalDecision("none", reason="no spread"))
        self.spreads.append(spread)
        state["spread"] = spread

        zscore = self._zscore()
        if zscore is None:
            return self._record_decision(state, SignalDecision("none", reason="zscore not ready"))
        state["zscore"] = zscore

        entry_thresholds_ready = self._prepare_entry_thresholds()
        state.update(self._entry_rule_state())
        if position_side is not None and abs(zscore) < self.exit_z:
            decision = SignalDecision("close", position_side, reason=f"zscore={zscore:.4f}")
        elif entry_thresholds_ready:
            decision = self._decision_from_zscore(zscore, position_side)
        else:
            decision = SignalDecision("none", reason=self.entry_rule_reason or "entry threshold not ready")
        decision.hedge_ratio = self._hedge_ratio()
        decision.long_vol = self._rolling_vol(self.x_closes)
        decision.short_vol = self._rolling_vol(self.y_closes)
        self.zscores.append(zscore)
        state.update(
            {
                "hedge_ratio": decision.hedge_ratio,
                "long_vol": decision.long_vol,
                "short_vol": decision.short_vol,
            }
        )
        return self._record_decision(state, decision)

    def _prepare_model(self):
        return True

    def _spread(self, x_close, y_close):
        return log(y_close) - log(x_close)

    def _hedge_ratio(self):
        return 1.0

    def _entry_filter(self, side, zscore):
        return True

    def _decision_from_zscore(self, zscore, position_side):
        upper = self.entry_z_upper
        lower = self.entry_z_lower
        if position_side is None:
            if zscore > upper:
                if self._entry_filter("short_spread", zscore):
                    return SignalDecision("open", "short_spread", reason=f"zscore={zscore:.4f}")
                return SignalDecision("none", reason=self._entry_filter_reason(zscore))
            if zscore < lower:
                if self._entry_filter("long_spread", zscore):
                    return SignalDecision("open", "long_spread", reason=f"zscore={zscore:.4f}")
                return SignalDecision("none", reason=self._entry_filter_reason(zscore))
            return SignalDecision("none", reason=f"zscore={zscore:.4f}")

        if abs(zscore) < self.exit_z:
            return SignalDecision("close", position_side, reason=f"zscore={zscore:.4f}")

        if position_side == "short_spread" and zscore > upper:
            if self._entry_filter(position_side, zscore):
                return SignalDecision("open", position_side, reason=f"zscore={zscore:.4f}")
            return SignalDecision("none", reason=self._entry_filter_reason(zscore))
        if position_side == "long_spread" and zscore < lower:
            if self._entry_filter(position_side, zscore):
                return SignalDecision("open", position_side, reason=f"zscore={zscore:.4f}")
            return SignalDecision("none", reason=self._entry_filter_reason(zscore))
        return SignalDecision("none", reason=f"zscore={zscore:.4f}")

    def _entry_filter_reason(self, zscore):
        reason = getattr(self, "_entry_filter_block_reason", None)
        return reason or f"zscore={zscore:.4f}"

    def _prepare_entry_thresholds(self):
        if self.entry_rule_method in {"fixed", "fixed_z"}:
            self.entry_z_upper = float(self.entry_z)
            self.entry_z_lower = -float(self.entry_z)
            self.entry_rule_reason = "fixed z threshold"
            return True

        if self.entry_rule_method != "rolling_quantile":
            raise ValueError(f"unsupported entry_rule method: {self.entry_rule_method}")

        history = np.array(self.zscores[-self.entry_rule_lookback_bars :], dtype=float)
        history = history[np.isfinite(history)]
        if len(history) < self.entry_rule_min_samples:
            self.entry_z_upper = None
            self.entry_z_lower = None
            self.entry_rule_reason = "entry threshold warmup"
            return False

        if (
            self.last_entry_rule_update_index is None
            or self.bar_count - self.last_entry_rule_update_index >= self.entry_rule_update_interval_bars
        ):
            upper = float(np.percentile(history, self.entry_rule_upper_percentile))
            lower = float(np.percentile(history, self.entry_rule_lower_percentile))
            if self.entry_rule_min_abs_entry_z is not None:
                upper = max(upper, self.entry_rule_min_abs_entry_z)
                lower = min(lower, -self.entry_rule_min_abs_entry_z)
            if self.entry_rule_max_abs_entry_z is not None:
                upper = min(upper, self.entry_rule_max_abs_entry_z)
                lower = max(lower, -self.entry_rule_max_abs_entry_z)
            if not isfinite(upper) or not isfinite(lower) or upper <= 0 or lower >= 0:
                self.entry_rule_reason = "invalid entry threshold"
                return False
            self.entry_z_upper = upper
            self.entry_z_lower = lower
            self.last_entry_rule_update_index = self.bar_count

        self.entry_rule_reason = "rolling quantile threshold"
        return True

    def _entry_rule_state(self):
        return {
            "entry_rule_method": self.entry_rule_method,
            "entry_z_upper": self.entry_z_upper,
            "entry_z_lower": self.entry_z_lower,
            "entry_rule_lookback_bars": self.entry_rule_lookback_bars,
            "entry_rule_update_interval_bars": self.entry_rule_update_interval_bars,
            "entry_rule_upper_percentile": self.entry_rule_upper_percentile,
            "entry_rule_lower_percentile": self.entry_rule_lower_percentile,
            "entry_rule_sample_count": len(self.zscores[-self.entry_rule_lookback_bars :]),
            "entry_rule_min_samples": self.entry_rule_min_samples,
            "entry_rule_last_update_index": self.last_entry_rule_update_index,
            "entry_rule_reason": self.entry_rule_reason,
        }

    def _validate_entry_rule(self):
        if self.entry_rule_method not in {"fixed", "fixed_z", "rolling_quantile"}:
            raise ValueError("entry_rule method must be 'fixed_z' or 'rolling_quantile'")
        if self.entry_rule_lookback_bars <= 0:
            raise ValueError("entry_rule lookback_bars must be positive")
        if self.entry_rule_update_interval_bars <= 0:
            raise ValueError("entry_rule update_interval_bars must be positive")
        if self.entry_rule_min_samples <= 0:
            raise ValueError("entry_rule min_samples must be positive")
        if not 0 < self.entry_rule_lower_percentile < self.entry_rule_upper_percentile < 100:
            raise ValueError("entry_rule percentiles must satisfy 0 < lower < upper < 100")
        if self.entry_rule_min_abs_entry_z is not None and self.entry_rule_min_abs_entry_z <= 0:
            raise ValueError("entry_rule min_abs_entry_z must be positive")
        if self.entry_rule_max_abs_entry_z is not None and self.entry_rule_max_abs_entry_z <= 0:
            raise ValueError("entry_rule max_abs_entry_z must be positive")
        if (
            self.entry_rule_min_abs_entry_z is not None
            and self.entry_rule_max_abs_entry_z is not None
            and self.entry_rule_min_abs_entry_z > self.entry_rule_max_abs_entry_z
        ):
            raise ValueError("entry_rule min_abs_entry_z cannot exceed max_abs_entry_z")

    def _zscore(self):
        if len(self.spreads) <= self.zscore_lookback_bars:
            return None
        current = float(self.spreads[-1])
        window = np.array(self.spreads[-self.zscore_lookback_bars - 1 : -1], dtype=float)
        if not np.isfinite(window).all():
            return None
        mean = float(window.mean())
        std = float(window.std(ddof=1))
        if not isfinite(current) or not isfinite(mean) or not isfinite(std) or std <= 0:
            return None
        return (current - mean) / std

    def _rolling_vol(self, prices):
        if len(prices) < 3:
            return None
        window_size = min(len(prices), self.zscore_lookback_bars)
        prices = np.array(prices[-window_size:], dtype=float)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            return None
        returns = np.diff(np.log(prices))
        if len(returns) < 2:
            return None
        return float(returns.std(ddof=1))

    def _record_decision(self, state, decision):
        state.update(
            {
                "action": decision.action,
                "side": decision.side,
                "reason": decision.reason,
                "hedge_ratio": decision.hedge_ratio,
                "long_vol": decision.long_vol,
                "short_vol": decision.short_vol,
                "entry_rule_method": self.entry_rule_method,
                "entry_z_upper": self.entry_z_upper,
                "entry_z_lower": self.entry_z_lower,
                "entry_rule_reason": self.entry_rule_reason,
            }
        )
        self.last_state = state
        return decision

    def _bar_ts(self, bars, exchange, symbol):
        bar = bars[exchange][symbol]
        if isinstance(bar, dict):
            return bar["ts"]
        return bar[0]
