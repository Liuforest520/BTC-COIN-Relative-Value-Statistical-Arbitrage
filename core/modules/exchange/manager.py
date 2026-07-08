from core.modules.exchange.exchange import Exchange
from core.modules.models import Order, OrderStatus


class ExchangeManager:
    def __init__(self, exchanges: dict[str, Exchange]):
        self.exchanges = exchanges
        self.initial_cash = sum(exchange.initial_cash for exchange in exchanges.values())
        self.equity_curve = []
        self.funding_payments = []

    def cancel_all_orders(self):
        for exchange in self.exchanges.values():
            exchange.cancel_all_orders()

    def place_orders(self, orders):
        if orders is None:
            return []
        if isinstance(orders, Order):
            orders = [orders]

        accepted = []
        for order in orders:
            exchange = self.exchanges.get(order.exchange)
            if exchange is None:
                order.status = OrderStatus.REJECTED
                continue
            accepted.extend(exchange.place_order(order))
        return accepted

    def on_bar(self, bars_by_exchange: dict, funding_rates=None):
        results = {}
        new_trades = []
        rejected_orders = []
        positions = {}
        funding_payments = []
        funding_rates = funding_rates or {}

        for exchange_name, exchange in self.exchanges.items():
            bars = bars_by_exchange.get(exchange_name, {})
            result = exchange(bars, funding_rates=funding_rates)
            results[exchange_name] = result
            new_trades.extend(result["new_trades"])
            rejected_orders.extend(result["rejected_orders"])
            funding_payments.extend(result["funding_payments"])
            positions[exchange_name] = result["positions"]

        self.funding_payments.extend(funding_payments)
        portfolio_state = self._portfolio_state(results)
        ts = self._current_ts(bars_by_exchange)
        if ts is not None:
            self.equity_curve.append({"ts": ts, "equity": portfolio_state["equity"]})

        return {
            "results": results,
            "new_trades": new_trades,
            "rejected_orders": rejected_orders,
            "funding_payments": funding_payments,
            "trades": self._all_trades(results),
            "positions": positions,
            "cash": portfolio_state["cash"],
            "equity": portfolio_state["equity"],
            "initial_cash": self.initial_cash,
            "equity_curve": self.equity_curve,
            "has_fill": len(new_trades) > 0,
        }

    @classmethod
    def from_names(cls, exchange_names, initial_cash=100000.0, fee_rate=0.0005, slippage_bps=1.0):
        exchange_names = list(exchange_names)
        if not exchange_names:
            raise ValueError("exchange_names cannot be empty")
        cash_per_exchange = initial_cash / len(exchange_names)
        exchanges = {}
        for name in exchange_names:
            exchanges[name] = Exchange(
                exchange_name=name,
                initial_cash=cash_per_exchange,
                fee_rate=fee_rate,
                slippage_bps=slippage_bps,
            )
        return cls(exchanges)

    def _portfolio_state(self, results):
        cash = 0.0
        equity = 0.0

        for result in results.values():
            cash += result["cash"]
            equity += result["equity"]

        return {
            "cash": cash,
            "equity": equity,
        }

    def _all_trades(self, results):
        trades = []
        for result in results.values():
            trades.extend(result["trades"])
        return trades

    def _current_ts(self, bars_by_exchange):
        for symbols in bars_by_exchange.values():
            for bar in symbols.values():
                if isinstance(bar, dict):
                    return bar.get("ts")
                if isinstance(bar, list) and bar:
                    return bar[0]
        return None
