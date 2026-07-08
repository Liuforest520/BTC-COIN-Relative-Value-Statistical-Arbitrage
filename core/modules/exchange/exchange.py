from core.modules.models import FundingPayment, Order, OrderAction, OrderSide, OrderStatus, OrderType, Trade
from core.modules.strategy import BAR_COLUMNS


class Exchange:
    def __init__(self, exchange_name, initial_cash=100000.0, fee_rate=0.0005, slippage_bps=1.0):
        self.exchange_name = exchange_name
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.fee_rate = fee_rate
        self.slippage = slippage_bps / 10000
        self.positions = {}
        self.orders = []
        self.order_history = []
        self.trade_history = []
        self.funding_history = []
        self.equity = initial_cash

    def place_order(self, orders):
        if orders is None:
            return []
        if isinstance(orders, Order):
            orders = [orders]

        accepted = []
        for order in orders:
            if order.exchange != self.exchange_name:
                order.status = OrderStatus.REJECTED
                self.order_history.append(order)
                continue

            if order.action == OrderAction.CANCEL:
                self.cancel_order(order.cancel_order_id)
                self.order_history.append(order)
            else:
                self.orders.append(order)
                self.order_history.append(order)
                accepted.append(order)
        return accepted

    def cancel_order(self, order_id):
        remaining_orders = []
        for order in self.orders:
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELED
            else:
                remaining_orders.append(order)
        self.orders = remaining_orders

    def cancel_all_orders(self):
        for order in self.orders:
            order.status = OrderStatus.CANCELED
        self.orders = []

    def __call__(self, bars, funding_rates=None):
        bars = self._format_bars(bars)
        funding_rates = funding_rates or {}
        filled_orders = []
        rejected_orders = []
        new_trades = []
        remaining_orders = []

        funding_payments = self._apply_funding(bars, funding_rates)

        for order in self.orders:
            bar = bars.get(order.symbol)
            if bar is None:
                remaining_orders.append(order)
                continue

            trade = self._execute_order(order, bar)
            if trade is None:
                if order.status != OrderStatus.REJECTED:
                    remaining_orders.append(order)
                else:
                    rejected_orders.append(order)
            else:
                filled_orders.append(order)
                self.trade_history.append(trade)
                new_trades.append(trade)

        self.orders = remaining_orders
        self._update_equity(bars)

        return {
            "filled_orders": filled_orders,
            "rejected_orders": rejected_orders,
            "new_trades": new_trades,
            "funding_payments": funding_payments,
            "trades": self.trade_history,
            "funding_history": self.funding_history,
            "has_fill": len(filled_orders) > 0,
            "has_open_orders": len(self.orders) > 0,
            "cash": self.cash,
            "positions": dict(self.positions),
            "equity": self.equity,
        }

    def _execute_order(self, order, bar):
        price = self._get_trade_price(order, bar)
        if price is None:
            return None

        quantity = float(order.quantity)
        notional = price * quantity
        fee = notional * self.fee_rate
        slippage = abs(price - bar["open"]) * quantity
        side = self._value(order.side)
        action = self._value(order.action)

        if not self._can_execute(order, side, action, quantity):
            order.status = OrderStatus.REJECTED
            return None

        self._apply_trade(order.symbol, side, quantity, notional, fee)
        order.status = OrderStatus.FILLED
        return Trade(
            order_id=order.order_id,
            group_id=order.group_id,
            exchange=order.exchange,
            symbol=order.symbol,
            action=action,
            position_id=order.position_id,
            side=side,
            quantity=quantity,
            price=price,
            notional=notional,
            fee=fee,
            slippage=slippage,
            ts=bar["ts"],
            target_hedge_ratio=order.target_hedge_ratio,
        )

    def _can_execute(self, order, side, action, quantity):
        if side not in [OrderSide.BUY.value, OrderSide.SELL.value]:
            return False

        if action == OrderAction.OPEN.value:
            return True

        if action != OrderAction.CLOSE.value:
            return False

        current_position = self.positions.get(order.symbol, 0.0)
        if side == OrderSide.SELL.value:
            return current_position > 0 and quantity <= current_position + 1e-12
        return current_position < 0 and quantity <= abs(current_position) + 1e-12

    def _apply_trade(self, symbol, side, quantity, notional, fee):
        if side == OrderSide.BUY.value:
            self.cash -= notional + fee
            new_position = self.positions.get(symbol, 0.0) + quantity
        else:
            self.cash += notional - fee
            new_position = self.positions.get(symbol, 0.0) - quantity

        if abs(new_position) < 1e-12:
            new_position = 0.0
        self.positions[symbol] = new_position

    def _get_trade_price(self, order, bar):
        order_type = self._value(order.order_type)
        side = self._value(order.side)

        if order.quantity is None or order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            return None

        if order_type in [None, OrderType.MARKET.value]:
            if side == OrderSide.BUY.value:
                return bar["open"] * (1 + self.slippage)
            if side == OrderSide.SELL.value:
                return bar["open"] * (1 - self.slippage)
            return None

        if order_type == OrderType.LIMIT.value:
            if order.price is None:
                order.status = OrderStatus.REJECTED
                return None
            if side == OrderSide.BUY.value and bar["low"] <= order.price:
                return order.price
            if side == OrderSide.SELL.value and bar["high"] >= order.price:
                return order.price

        return None

    def _update_equity(self, bars):
        position_value = 0.0
        for symbol, quantity in self.positions.items():
            bar = bars.get(symbol)
            if bar is not None:
                position_value += quantity * bar["close"]
        self.equity = self.cash + position_value

    def _apply_funding(self, bars, funding_rates):
        payments = []
        for symbol, funding_rate in funding_rates.items():
            quantity = self.positions.get(symbol, 0.0)
            if abs(quantity) < 1e-12:
                continue

            bar = bars.get(symbol)
            if bar is None:
                continue

            mark_price = float(bar["close"])
            notional = abs(float(quantity) * mark_price)
            payment = float(quantity) * mark_price * float(funding_rate)
            self.cash -= payment

            funding_payment = FundingPayment(
                exchange=self.exchange_name,
                symbol=symbol,
                ts=bar["ts"],
                funding_rate=float(funding_rate),
                quantity=float(quantity),
                mark_price=mark_price,
                notional=notional,
                payment=payment,
            )
            self.funding_history.append(funding_payment)
            payments.append(funding_payment)

        return payments

    def _format_bars(self, bars):
        if isinstance(bars, list):
            symbols = {order.symbol for order in self.orders}
            if len(symbols) != 1:
                raise ValueError("single bar can only be used when pending orders have one symbol")
            return {next(iter(symbols)): self._bar_to_dict(bars)}

        return {symbol: self._bar_to_dict(bar) for symbol, bar in bars.items()}

    def _bar_to_dict(self, bar):
        if isinstance(bar, dict):
            return bar
        if not isinstance(bar, list) or len(bar) != len(BAR_COLUMNS):
            raise ValueError("bar must be [ts, open, high, close, low, volume]")
        return dict(zip(BAR_COLUMNS, bar))

    def _value(self, value):
        if hasattr(value, "value"):
            return value.value
        return value
