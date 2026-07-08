from abc import ABC, abstractmethod

from core.modules.models import Order


BAR_COLUMNS = ["ts", "open", "high", "close", "low", "volume"]


class BaseStrategy(ABC):
    def __init__(self):
        self.data = []

    def __call__(self, bars: dict):
        self._check_bars(bars)
        self.data.append(bars)
        orders = self.on_bar(bars)
        return self._check_orders(orders)

    def _check_bars(self, bars: dict):
        if not isinstance(bars, dict):
            raise TypeError("bars must be a dict")

        for exchange, symbols in bars.items():
            if not isinstance(symbols, dict):
                raise TypeError(f"bars[{exchange}] must be a dict")
            for symbol, bar in symbols.items():
                self._check_bar(bar, exchange, symbol)

    def _check_bar(self, bar: list, exchange=None, symbol=None):
        if not isinstance(bar, list):
            raise TypeError("bar must be a list")
        if len(bar) != len(BAR_COLUMNS):
            name = f"{exchange}.{symbol}" if exchange and symbol else "bar"
            raise ValueError(f"{name} must be [ts, open, high, close, low, volume]")

    def _check_orders(self, orders):
        if orders is None:
            return []
        if isinstance(orders, Order):
            return [orders]
        if isinstance(orders, list):
            for order in orders:
                if not isinstance(order, Order):
                    raise TypeError("strategy must return Order, list[Order], or None")
            return orders
        raise TypeError("strategy must return Order, list[Order], or None")

    @abstractmethod
    def on_bar(self, bars: dict):
        pass

    def on_funding_rates(self, funding_rates: dict):
        pass

    def on_orders_accepted(self, orders: list[Order]):
        pass

    def on_orders_rejected(self, orders: list[Order]):
        pass

    def on_trades_filled(self, trades: list):
        pass
