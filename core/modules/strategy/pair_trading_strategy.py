from uuid import uuid4

from core.modules.models import Order, OrderAction, OrderSide, OrderType
from core.modules.strategy.base import BaseStrategy


class PairTradingStrategy(BaseStrategy):
    def __init__(
        self,
        signal,
        sizing,
        x_exchange="binance",
        x_symbol="BTCUSDT",
        y_exchange="binance",
        y_symbol="COINUSDT",
        max_add_times=0,
        add_interval_bars=1,
        pending_timeout_bars=5,
        name="pair_trading",
    ):
        super().__init__()
        self.signal = signal
        self.sizing = sizing
        self.x_exchange = x_exchange
        self.x_symbol = x_symbol
        self.y_exchange = y_exchange
        self.y_symbol = y_symbol
        self.max_add_times = max_add_times
        self.add_interval_bars = add_interval_bars
        self.pending_timeout_bars = pending_timeout_bars
        self.name = name

        self.position_side = None
        self.position_id = None
        self.x_quantity = 0.0
        self.y_quantity = 0.0
        self.entry_count = 0
        self.last_entry_bar_index = None
        self.last_open_reject_bar_index = None
        self.pending_state = None
        self.pending_fills = []

    def on_bar(self, bars: dict):
        bar_index = len(self.data)
        if self.pending_state is not None:
            timeout_orders = self._pending_timeout_orders(bar_index)
            if timeout_orders:
                self._record_pending_timeout()
                self.pending_state = None
                self.pending_fills = []
                return timeout_orders
            return []

        decision = self.signal(bars, self.position_side)

        if decision.action == "open":
            if self._open_reject_cooldown_active(bar_index):
                return []
            if self.position_side is not None and not self._can_add(decision.side, bar_index):
                return []
            return self._open_orders(bars, decision, bar_index)

        if decision.action == "close" and self.position_side is not None:
            return self._close_orders(bars, bar_index)

        return []

    def on_funding_rates(self, funding_rates):
        self.signal.on_funding_rates(funding_rates)

    def _can_add(self, side, bar_index):
        if side != self.position_side:
            return False
        if self.entry_count - 1 >= self.max_add_times:
            return False
        if self.last_entry_bar_index is not None and bar_index - self.last_entry_bar_index < self.add_interval_bars:
            return False
        return True

    def _open_reject_cooldown_active(self, bar_index):
        if self.last_open_reject_bar_index is None:
            return False
        return bar_index - self.last_open_reject_bar_index < self.add_interval_bars

    def _open_orders(self, bars, decision, bar_index):
        group_id = f"{self.name}-open-{bar_index}-{uuid4().hex[:8]}"
        position_id = self.position_id or f"{self.name}-position-{bar_index}"

        try:
            sizing = self.sizing.notionals(
                bars=bars,
                x_leg=(self.x_exchange, self.x_symbol),
                y_leg=(self.y_exchange, self.y_symbol),
                hedge_ratio=decision.hedge_ratio,
                x_vol=decision.long_vol,
                y_vol=decision.short_vol,
            )
        except ValueError as exc:
            state = getattr(self.signal, "last_state", {}) or {}
            state["action"] = "none"
            state["reason"] = str(exc)
            self.signal.last_state = state
            return []
        x_quantity = sizing["x_quantity"]
        y_quantity = sizing["y_quantity"]

        if decision.side == "short_spread":
            x_side = OrderSide.BUY
            y_side = OrderSide.SELL
        else:
            x_side = OrderSide.SELL
            y_side = OrderSide.BUY
        target_hedge_ratio = self._target_hedge_ratio(decision)

        orders = [
            self._order(
                order_id=f"{group_id}-{self.x_symbol}",
                group_id=group_id,
                exchange=self.x_exchange,
                symbol=self.x_symbol,
                action=OrderAction.OPEN,
                side=x_side,
                quantity=x_quantity,
                price=self._open_price(bars, self.x_exchange, self.x_symbol),
                position_id=position_id,
                target_hedge_ratio=target_hedge_ratio,
            ),
            self._order(
                order_id=f"{group_id}-{self.y_symbol}",
                group_id=group_id,
                exchange=self.y_exchange,
                symbol=self.y_symbol,
                action=OrderAction.OPEN,
                side=y_side,
                quantity=y_quantity,
                price=self._open_price(bars, self.y_exchange, self.y_symbol),
                position_id=position_id,
                target_hedge_ratio=target_hedge_ratio,
            ),
        ]

        self.pending_state = {
            "action": "open",
            "group_id": group_id,
            "position_id": position_id,
            "position_side": decision.side,
            "x_quantity_delta": x_quantity,
            "y_quantity_delta": y_quantity,
            "bar_index": bar_index,
            "orders": self._pending_order_refs(orders),
        }
        self.pending_fills = []

        return orders

    def _close_orders(self, bars, bar_index):
        group_id = f"{self.name}-close-{bar_index}-{uuid4().hex[:8]}"
        position_id = self.position_id

        if self.position_side == "short_spread":
            x_side = OrderSide.SELL
            y_side = OrderSide.BUY
        else:
            x_side = OrderSide.BUY
            y_side = OrderSide.SELL

        orders = [
            self._order(
                order_id=f"{group_id}-{self.x_symbol}",
                group_id=group_id,
                exchange=self.x_exchange,
                symbol=self.x_symbol,
                action=OrderAction.CLOSE,
                side=x_side,
                quantity=self.x_quantity,
                price=self._open_price(bars, self.x_exchange, self.x_symbol),
                position_id=position_id,
            ),
            self._order(
                order_id=f"{group_id}-{self.y_symbol}",
                group_id=group_id,
                exchange=self.y_exchange,
                symbol=self.y_symbol,
                action=OrderAction.CLOSE,
                side=y_side,
                quantity=self.y_quantity,
                price=self._open_price(bars, self.y_exchange, self.y_symbol),
                position_id=position_id,
            ),
        ]
        self.pending_state = {
            "action": "close",
            "group_id": group_id,
            "position_id": position_id,
            "bar_index": bar_index,
            "orders": self._pending_order_refs(orders),
        }
        self.pending_fills = []

        return orders

    def on_orders_accepted(self, orders):
        if not self.pending_state or not orders:
            self.pending_state = None
            self.pending_fills = []
            return

        group_id = self.pending_state["group_id"]
        accepted_group = [order for order in orders if order.group_id == group_id]
        if len(accepted_group) != 2:
            self.pending_state = None
            self.pending_fills = []
        return

    def on_trades_filled(self, trades):
        if not self.pending_state or not trades:
            return

        group_id = self.pending_state.get("group_id")
        new_fills = [trade for trade in trades if trade.group_id == group_id]
        if not new_fills:
            return

        seen_order_ids = {trade.order_id for trade in self.pending_fills}
        self.pending_fills.extend(trade for trade in new_fills if trade.order_id not in seen_order_ids)

        filled_symbols = {trade.symbol for trade in self.pending_fills}
        if self.x_symbol not in filled_symbols or self.y_symbol not in filled_symbols:
            return

        filled_group = self.pending_fills
        if self.pending_state["action"] == "open":
            x_quantity = self._filled_quantity(filled_group, self.x_symbol)
            y_quantity = self._filled_quantity(filled_group, self.y_symbol)
            if x_quantity <= 0 or y_quantity <= 0:
                self.pending_state = None
                self.pending_fills = []
                return

            if self.position_id is None:
                self.position_id = self.pending_state["position_id"]
                self.position_side = self.pending_state["position_side"]
            self.x_quantity += x_quantity
            self.y_quantity += y_quantity
            self.entry_count += 1
            self.last_entry_bar_index = self.pending_state["bar_index"]
        elif self.pending_state["action"] == "close":
            self.position_side = None
            self.position_id = None
            self.x_quantity = 0.0
            self.y_quantity = 0.0
            self.entry_count = 0
            self.last_entry_bar_index = None
            self.last_open_reject_bar_index = None

        self.pending_state = None
        self.pending_fills = []

    def on_orders_rejected(self, orders):
        if not self.pending_state:
            return
        if not orders:
            self.pending_state = None
            self.pending_fills = []
            return

        group_id = self.pending_state["group_id"]
        if any(order.group_id == group_id for order in orders):
            if self.pending_state.get("action") == "open":
                self.last_open_reject_bar_index = self.pending_state.get("bar_index")
            self.pending_state = None
            self.pending_fills = []

    def _filled_quantity(self, trades, symbol):
        return sum(float(trade.quantity) for trade in trades if trade.symbol == symbol)

    def _pending_timeout_orders(self, bar_index):
        if self.pending_timeout_bars is None or self.pending_timeout_bars <= 0:
            return []
        if not self.pending_state:
            return []
        started_at = self.pending_state.get("bar_index")
        if started_at is None or bar_index - started_at < self.pending_timeout_bars:
            return []
        if self.pending_fills:
            return []
        return [
            self._order(
                order_id=f"{item['group_id']}-cancel-{item['symbol']}",
                group_id=item["group_id"],
                exchange=item["exchange"],
                symbol=item["symbol"],
                action=OrderAction.CANCEL,
                side=None,
                quantity=0.0,
                cancel_order_id=item["order_id"],
            )
            for item in self.pending_state.get("orders", [])
        ]

    def _pending_order_refs(self, orders):
        return [
            {
                "order_id": order.order_id,
                "group_id": order.group_id,
                "exchange": order.exchange,
                "symbol": order.symbol,
            }
            for order in orders
        ]

    def _record_pending_timeout(self):
        state = getattr(self.signal, "last_state", {}) or {}
        state["action"] = "none"
        state["reason"] = f"pending {self.pending_state.get('action')} timeout"
        self.signal.last_state = state

    def _order(self, order_id, group_id, exchange, symbol, action, side=None, quantity=None, price=None, position_id=None, target_hedge_ratio=None, cancel_order_id=None):
        return Order(
            order_id=order_id,
            group_id=group_id,
            exchange=exchange,
            symbol=symbol,
            action=action,
            position_id=position_id if position_id is not None else self.position_id,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=price,
            cancel_order_id=cancel_order_id,
            target_hedge_ratio=target_hedge_ratio,
        )

    def _target_hedge_ratio(self, decision):
        method = getattr(self.sizing, "method", "fixed_notional")
        if method == "beta_neutral":
            x_to_y_ratio = abs(float(decision.hedge_ratio)) if decision.hedge_ratio else 1.0
        elif method == "volatility_neutral":
            x_vol = decision.long_vol
            y_vol = decision.short_vol
            x_to_y_ratio = y_vol / x_vol if x_vol and y_vol and x_vol > 0 and y_vol > 0 else 1.0
        else:
            x_to_y_ratio = 1.0

        if x_to_y_ratio <= 0:
            x_to_y_ratio = 1.0

        if decision.side == "short_spread":
            return x_to_y_ratio
        return 1 / x_to_y_ratio

    def _open_price(self, bars, exchange, symbol):
        return float(bars[exchange][symbol][1])
