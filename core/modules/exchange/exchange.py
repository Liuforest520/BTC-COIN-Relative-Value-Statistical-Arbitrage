from core.modules.models import FundingPayment, Order, OrderAction, OrderSide, OrderStatus, OrderType, Trade
from core.modules.strategy import BAR_COLUMNS


class Exchange:
    def __init__(
        self,
        exchange_name,
        initial_cash=100000.0,
        fee_rate=0.0005,
        slippage_bps=1.0,
        max_leverage=1.0,
    ):
        self.exchange_name = exchange_name
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.fee_rate = fee_rate
        self.slippage = slippage_bps / 10000
        self.max_leverage = 1.0 if max_leverage is None else float(max_leverage)
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        self.positions = {}
        self.position_lots = {}
        self.orders = []
        self.order_history = []
        self.trade_history = []
        self.funding_history = []
        self.last_bars = {}
        self.equity = initial_cash
        self.copy_positions_on_bar = True

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
                order.status = OrderStatus.CANCELED
                self.order_history.append(order)
                accepted.append(order)
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

    def __call__(self, bars, funding_rates=None, blocked_group_ids=None):
        bars = self._format_bars(bars)
        self.last_bars.update(bars)
        funding_rates = funding_rates or {}
        blocked_group_ids = blocked_group_ids or set()
        filled_orders = []
        rejected_orders = []
        new_trades = []
        remaining_orders = []

        funding_payments = self._apply_funding(bars, funding_rates)
        margin_rejected_ids = self._margin_rejected_order_ids(bars)

        for group_orders in self._pending_order_groups().values():
            group_id = group_orders[0].group_id
            if group_id in blocked_group_ids:
                remaining_orders.extend(group_orders)
                continue

            if not self._group_has_current_bars(group_orders, bars):
                remaining_orders.extend(group_orders)
                continue

            if any(order.order_id in margin_rejected_ids for order in group_orders):
                self._reject_group(group_orders, rejected_orders)
                continue

            status, prepared = self._prepare_group_execution(group_orders, bars)
            if status == "waiting":
                remaining_orders.extend(group_orders)
                continue
            if status == "rejected":
                self._reject_group(group_orders, rejected_orders)
                continue

            for order, bar, price in prepared:
                trade = self._execute_prepared_order(order, bar, price)
                filled_orders.append(order)
                self.trade_history.append(trade)
                new_trades.append(trade)

        self.orders = remaining_orders
        self._update_equity(self.last_bars)

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
            "positions": dict(self.positions) if self.copy_positions_on_bar else self.positions,
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

        self._apply_trade(order.symbol, side, quantity, notional, fee, action, order.position_id)
        order.status = OrderStatus.FILLED
        return Trade(
            order_id=order.order_id,
            group_id=order.group_id,
            exchange=order.exchange,
            symbol=order.symbol,
            action=action,
            position_id=order.position_id,
            pair_id=order.pair_id,
            side=side,
            quantity=quantity,
            price=price,
            notional=notional,
            fee=fee,
            slippage=slippage,
            ts=bar["ts"],
            target_hedge_ratio=order.target_hedge_ratio,
        )

    def _execute_prepared_order(self, order, bar, price):
        quantity = float(order.quantity)
        notional = price * quantity
        fee = notional * self.fee_rate
        slippage = abs(price - bar["open"]) * quantity
        side = self._value(order.side)
        action = self._value(order.action)

        self._apply_trade(order.symbol, side, quantity, notional, fee, action, order.position_id)
        order.status = OrderStatus.FILLED
        return Trade(
            order_id=order.order_id,
            group_id=order.group_id,
            exchange=order.exchange,
            symbol=order.symbol,
            action=action,
            position_id=order.position_id,
            pair_id=order.pair_id,
            side=side,
            quantity=quantity,
            price=price,
            notional=notional,
            fee=fee,
            slippage=slippage,
            ts=bar["ts"],
            target_hedge_ratio=order.target_hedge_ratio,
        )

    def _pending_order_groups(self):
        groups = {}
        for order in self.orders:
            key = order.group_id or order.order_id
            groups.setdefault(key, []).append(order)
        return groups

    def _group_has_current_bars(self, orders, bars):
        return all(order.symbol in bars for order in orders)

    def _reject_group(self, orders, rejected_orders):
        for order in orders:
            order.status = OrderStatus.REJECTED
            rejected_orders.append(order)

    def _prepare_group_execution(self, orders, bars):
        prepared = []
        positions = dict(self.positions)
        position_lots = {pid: dict(lots) for pid, lots in self.position_lots.items()}

        for order in orders:
            bar = bars.get(order.symbol)
            if bar is None:
                return "waiting", []

            price = self._get_trade_price(order, bar)
            if price is None:
                if order.status == OrderStatus.REJECTED:
                    return "rejected", []
                return "waiting", []

            quantity = float(order.quantity)
            side = self._value(order.side)
            action = self._value(order.action)
            if not self._can_execute_against(order, side, action, quantity, positions, position_lots):
                return "rejected", []

            self._simulate_trade_state(order, side, action, quantity, positions, position_lots)
            prepared.append((order, bar, price))

        return "ready", prepared

    def _can_execute_against(self, order, side, action, quantity, positions, position_lots):
        if side not in [OrderSide.BUY.value, OrderSide.SELL.value]:
            return False

        if action == OrderAction.OPEN.value:
            return True

        if action != OrderAction.CLOSE.value:
            return False

        current_position = positions.get(order.symbol, 0.0)
        if order.position_id:
            lot_position = position_lots.get(order.position_id, {}).get(order.symbol, 0.0)
            if side == OrderSide.SELL.value:
                return lot_position > 0 and quantity <= lot_position + 1e-12
            return lot_position < 0 and quantity <= abs(lot_position) + 1e-12

        if side == OrderSide.SELL.value:
            return current_position > 0 and quantity <= current_position + 1e-12
        return current_position < 0 and quantity <= abs(current_position) + 1e-12

    def _simulate_trade_state(self, order, side, action, quantity, positions, position_lots):
        delta = quantity if side == OrderSide.BUY.value else -quantity
        new_position = positions.get(order.symbol, 0.0) + delta
        positions[order.symbol] = 0.0 if abs(new_position) < 1e-12 else new_position

        if not order.position_id or action not in {OrderAction.OPEN.value, OrderAction.CLOSE.value}:
            return

        lots = position_lots.setdefault(order.position_id, {})
        new_lot_quantity = lots.get(order.symbol, 0.0) + delta
        if abs(new_lot_quantity) < 1e-12:
            lots.pop(order.symbol, None)
        else:
            lots[order.symbol] = new_lot_quantity
        if not lots:
            position_lots.pop(order.position_id, None)

    def _can_execute(self, order, side, action, quantity):
        if side not in [OrderSide.BUY.value, OrderSide.SELL.value]:
            return False

        if action == OrderAction.OPEN.value:
            return True

        if action != OrderAction.CLOSE.value:
            return False

        current_position = self.positions.get(order.symbol, 0.0)
        if order.position_id:
            lot_position = self.position_lots.get(order.position_id, {}).get(order.symbol, 0.0)
            if side == OrderSide.SELL.value:
                return lot_position > 0 and quantity <= lot_position + 1e-12
            return lot_position < 0 and quantity <= abs(lot_position) + 1e-12

        if side == OrderSide.SELL.value:
            return current_position > 0 and quantity <= current_position + 1e-12
        return current_position < 0 and quantity <= abs(current_position) + 1e-12

    def _apply_trade(self, symbol, side, quantity, notional, fee, action=None, position_id=None):
        if side == OrderSide.BUY.value:
            self.cash -= notional + fee
            delta = quantity
        else:
            self.cash += notional - fee
            delta = -quantity

        new_position = self.positions.get(symbol, 0.0) + delta
        if abs(new_position) < 1e-12:
            new_position = 0.0
        self.positions[symbol] = new_position
        self._apply_position_lot(symbol, delta, action, position_id)

    def _apply_position_lot(self, symbol, delta, action, position_id):
        if not position_id or action not in {OrderAction.OPEN.value, OrderAction.CLOSE.value}:
            return

        lots = self.position_lots.setdefault(position_id, {})
        new_quantity = lots.get(symbol, 0.0) + delta
        if abs(new_quantity) < 1e-12:
            lots.pop(symbol, None)
        else:
            lots[symbol] = new_quantity

        if not lots:
            self.position_lots.pop(position_id, None)

    def _margin_rejected_order_ids(self, bars):
        rejected = set()
        groups = {}
        for order in self.orders:
            if self._value(order.action) != OrderAction.OPEN.value:
                continue
            groups.setdefault(order.group_id, []).append(order)

        for group_orders in groups.values():
            if not self._can_open_group(group_orders, bars):
                rejected.update(order.order_id for order in group_orders)
        return rejected

    def _can_open_group(self, orders, bars):
        cash = float(self.cash)
        positions = dict(self.positions)

        for order in orders:
            bar = bars.get(order.symbol)
            if bar is None:
                return False

            price = self._projected_order_price(order, bar)
            if price is None:
                return False

            quantity = float(order.quantity)
            notional = price * quantity
            fee = notional * self.fee_rate
            side = self._value(order.side)

            if side == OrderSide.BUY.value:
                cash -= notional + fee
                positions[order.symbol] = positions.get(order.symbol, 0.0) + quantity
            elif side == OrderSide.SELL.value:
                cash += notional - fee
                positions[order.symbol] = positions.get(order.symbol, 0.0) - quantity
            else:
                return False

        equity, gross_exposure = self._projected_exposure(cash, positions, bars)
        if equity <= 0:
            return False
        return gross_exposure <= equity * self.max_leverage + 1e-9

    def _projected_order_price(self, order, bar):
        order_type = self._value(order.order_type)
        if order.quantity is None or order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            return None
        if order_type == OrderType.LIMIT.value:
            if order.price is None or order.price <= 0:
                order.status = OrderStatus.REJECTED
                return None
            return float(order.price)
        return self._get_trade_price(order, bar)

    def _projected_exposure(self, cash, positions, bars):
        position_value = 0.0
        gross_exposure = 0.0
        for symbol, quantity in positions.items():
            if abs(float(quantity)) < 1e-12:
                continue
            bar = bars.get(symbol)
            if bar is None:
                continue
            mark_price = float(bar["close"])
            value = float(quantity) * mark_price
            position_value += value
            gross_exposure += abs(value)
        return cash + position_value, gross_exposure

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
        for event in self._funding_events(funding_rates):
            payments.extend(self._apply_funding_rates(bars, event["rates"], event.get("ts")))
        return payments

    def _apply_funding_rates(self, bars, funding_rates, event_ts=None):
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
                ts=event_ts if event_ts is not None else bar["ts"],
                funding_rate=float(funding_rate),
                quantity=float(quantity),
                mark_price=mark_price,
                notional=notional,
                payment=payment,
            )
            self.funding_history.append(funding_payment)
            payments.append(funding_payment)

        return payments

    def _funding_events(self, funding_rates):
        if not funding_rates:
            return []
        if isinstance(funding_rates, list):
            return funding_rates
        if isinstance(funding_rates, tuple):
            return list(funding_rates)
        if isinstance(funding_rates, dict) and "rates" in funding_rates:
            return [funding_rates]
        return [{"ts": None, "rates": funding_rates}]

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
