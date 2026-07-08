import numpy as np

from core.modules.signals.cointegration import CointegrationZScoreSignal


class MomentumFilterZScoreSignal(CointegrationZScoreSignal):
    def __init__(self, momentum_lookback_bars=30, max_abs_momentum=0.5, **kwargs):
        super().__init__(**kwargs)
        self.momentum_lookback_bars = momentum_lookback_bars
        self.max_abs_momentum = max_abs_momentum

    def _entry_filter(self, side, zscore):
        if len(self.spreads) <= self.momentum_lookback_bars:
            return True

        past_zscore = self._past_zscore(self.momentum_lookback_bars)
        momentum = zscore - past_zscore
        if side == "short_spread" and momentum > self.max_abs_momentum:
            return False
        if side == "long_spread" and momentum < -self.max_abs_momentum:
            return False
        return True

    def _past_zscore(self, lookback):
        window = self.spreads[-self.zscore_lookback_bars - lookback : -lookback]
        if len(window) < self.zscore_lookback_bars:
            return 0.0

        current = self.spreads[-lookback]
        window = np.array(window, dtype=float)
        mean = float(window.mean())
        std = float(window.std(ddof=1))
        if std <= 0:
            return 0.0
        return (current - mean) / std
