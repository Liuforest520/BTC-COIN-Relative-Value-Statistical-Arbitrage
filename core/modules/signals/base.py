from dataclasses import dataclass


@dataclass
class SignalDecision:
    action: str
    side: str | None = None
    hedge_ratio: float = 1.0
    long_vol: float | None = None
    short_vol: float | None = None
    reason: str = ""


class BasePairSignal:
    def __init__(
        self,
        x_exchange="binance",
        x_symbol="BTCUSDT",
        y_exchange="binance",
        y_symbol="COINUSDT",
        warmup_bars=4320,
        zscore_lookback_bars=1440,
        entry_z=2.0,
        exit_z=0.5,
    ):
        self.x_exchange = x_exchange
        self.x_symbol = x_symbol
        self.y_exchange = y_exchange
        self.y_symbol = y_symbol
        self.warmup_bars = warmup_bars
        self.zscore_lookback_bars = zscore_lookback_bars
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.bar_count = 0
        self.last_state = {}

    def __call__(self, bars, position_side=None):
        self.bar_count += 1
        return self.on_bar(bars, position_side)

    def on_bar(self, bars, position_side=None):
        raise NotImplementedError

    def on_funding_rates(self, funding_rates):
        pass

    def _close_price(self, bars, exchange, symbol):
        return float(bars[exchange][symbol][3])
