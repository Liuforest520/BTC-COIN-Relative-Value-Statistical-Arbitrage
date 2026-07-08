from math import log

import numpy as np

from core.modules.signals.base import BasePairSignal, SignalDecision


class ZScoreSignal(BasePairSignal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.x_closes = []
        self.y_closes = []
        self.spreads = []

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
            "exit_z": self.exit_z,
        }
        if x_close <= 0 or y_close <= 0:
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

        decision = self._decision_from_zscore(zscore, position_side)
        decision.hedge_ratio = self._hedge_ratio()
        decision.long_vol = self._rolling_vol(self.x_closes)
        decision.short_vol = self._rolling_vol(self.y_closes)
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
        if position_side is None:
            if zscore > self.entry_z and self._entry_filter("short_spread", zscore):
                return SignalDecision("open", "short_spread", reason=f"zscore={zscore:.4f}")
            if zscore < -self.entry_z and self._entry_filter("long_spread", zscore):
                return SignalDecision("open", "long_spread", reason=f"zscore={zscore:.4f}")
            return SignalDecision("none", reason=f"zscore={zscore:.4f}")

        if abs(zscore) < self.exit_z:
            return SignalDecision("close", position_side, reason=f"zscore={zscore:.4f}")

        if position_side == "short_spread" and zscore > self.entry_z and self._entry_filter(position_side, zscore):
            return SignalDecision("open", position_side, reason=f"zscore={zscore:.4f}")
        if position_side == "long_spread" and zscore < -self.entry_z and self._entry_filter(position_side, zscore):
            return SignalDecision("open", position_side, reason=f"zscore={zscore:.4f}")
        return SignalDecision("none", reason=f"zscore={zscore:.4f}")

    def _zscore(self):
        if len(self.spreads) < self.zscore_lookback_bars:
            return None
        window = np.array(self.spreads[-self.zscore_lookback_bars :], dtype=float)
        mean = float(window.mean())
        std = float(window.std(ddof=1))
        if std <= 0:
            return None
        return (self.spreads[-1] - mean) / std

    def _rolling_vol(self, prices):
        if len(prices) < 3:
            return None
        window_size = min(len(prices), self.zscore_lookback_bars)
        prices = np.array(prices[-window_size:], dtype=float)
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
            }
        )
        self.last_state = state
        return decision

    def _bar_ts(self, bars, exchange, symbol):
        bar = bars[exchange][symbol]
        if isinstance(bar, dict):
            return bar["ts"]
        return bar[0]
