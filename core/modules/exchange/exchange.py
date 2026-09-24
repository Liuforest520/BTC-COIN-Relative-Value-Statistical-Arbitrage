from __future__ import annotations

from copy import deepcopy
from math import isfinite

from core.modules.models import FundingPayment, Order, OrderAction, OrderSide, OrderStatus, OrderType, Trade
from core.modules.strategy import BAR_COLUMNS


class Exchange:
    """Linear perpetual-futures account with explicit margin accounting."""

    def __init__(
        self,
        exchange_name,
        initial_cash=100000.0,
        fee_rate=0.0005,
        slippage_bps=1.0,
        max_leverage=1.0,
    ):
        self.exchange_name = exchange_name
        self.initial_cash = float(initial_cash)
        self.wallet_balance = float(initial_cash)
        # Compatibility alias used by existing reports and callbacks.
        self.cash = self.wallet_balance
        self.fee_rate = float(fee_rate)
        self.slippage = float(slippage_bps) / 10000
        self.max_leverage = 1.0 if max_leverage is None else float(max_leverage)
        if not isfinite(self.max_leverage) or self.max_leverage <= 0:
            raise ValueError("max_leverage must be a finite positive number")

        self.positions: dict[str, float] = {}
        self.position_lots: dict[str, dict[str, float]] = {}
        self.position_entry_prices: dict[str, dict[str, float]] = {}
        # Capital is reserved when a futures lot is opened and released only
        # when that same lot is closed.  Mark-price changes affect equity via
        # unrealized PnL, but must not consume otherwise unallocated cash.
        self.position_reserved_margin: dict[str, dict[str, float]] = {}
        # Isolated capital account for every logical Pair position.  Free cash
        # is removed once at entry; subsequent fees, funding and PnL stay in
        # this position account until the position is closed.
        self.position_capital: dict[str, float] = {}
        self.position_pair_ids: dict[str, str] = {}
        self.orders = []
        self.order_history = []
        self.trade_history = []
        self.funding_history = []
        self.last_bars = {}
        self.equity = self.initial_cash
        self.unrealized_pnl = 0.0
        self.used_margin = 0.0
        self.available_balance = self.initial_cash
        self.gross_exposure = 0.0
        self.net_exposure = 0.0
        self.position_marked_equity: dict[str, float] = {}
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

    def begin_bar(self, bars, funding_rates=None):
        """Mark at the executable open and settle funding once."""
        bars = self._format_bars(bars)
        self.last_bars.update(bars)
        funding_payments = self._apply_funding(bars, funding_rates or {})
        # Funding uses the current executable open directly. Recalculate the
        # marked account once afterwards so open decisions see funding-adjusted
        # capital without repeating the same full account scan.
        self._update_account(self.last_bars, price_field="open")
        return bars, funding_payments

    def forced_liquidation_orders(self, bars, bar_index: int):
        """Create market closes for isolated Pair accounts whose equity is <= 0.

        The account is marked at the executable open before this method is
        called.  A liquidation is therefore queued at the same open and is
        filled by the normal atomic order path, so the loss is realized rather
        than silently capped at the Pair's initial capital.
        """
        insolvent = self._insolvent_position_ids(bars)
        if not insolvent:
            return []

        orders = []
        for position_id in sorted(insolvent):
            symbols = self.position_lots.get(position_id, {})
            if not symbols or any(symbol not in bars for symbol in symbols):
                continue
            # A strategy close already waiting for this position must not be
            # filled alongside the forced close.  Replace it with one complete
            # liquidation group.
            self._cancel_position_orders(position_id)
            pair_id = self.position_pair_ids.get(position_id)
            group_id = f"forced-liquidation-{self.exchange_name}-{position_id}-{bar_index}"
            for leg_index, (symbol, quantity) in enumerate(sorted(symbols.items())):
                quantity = float(quantity)
                if abs(quantity) <= 1e-12:
                    continue
                orders.append(
                    Order(
                        order_id=f"{group_id}-{leg_index}",
                        group_id=group_id,
                        exchange=self.exchange_name,
                        symbol=symbol,
                        action=OrderAction.CLOSE,
                        side=OrderSide.SELL if quantity > 0 else OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=abs(quantity),
                        position_id=position_id,
                        pair_id=pair_id,
                        exit_reason="protective_pair_equity_zero",
                        protection_trigger="pair_equity_zero",
                        exit_class="stop_loss",
                        reopen_lock_pending=True,
                        protection_rule="pair_equity_zero",
                    )
                )
        return orders

    def _insolvent_position_ids(self, bars) -> set[str]:
        insolvent = set()
        for position_id, symbols in self.position_lots.items():
            if not symbols:
                continue
            capital = self._position_capital_value(position_id)
            unrealized = 0.0
            complete = True
            for symbol, quantity in symbols.items():
                mark = self._mark_price(symbol, bars, "open")
                if mark is None:
                    complete = False
                    break
                entry = float(self.position_entry_prices.get(position_id, {}).get(symbol, mark))
                unrealized += float(quantity) * (float(mark) - entry)
            if complete and capital + unrealized <= 1e-9:
                insolvent.add(position_id)
        return insolvent

    def _cancel_position_orders(self, position_id: str) -> None:
        remaining = []
        for order in self.orders:
            if order.position_id == position_id:
                order.status = OrderStatus.CANCELED
            else:
                remaining.append(order)
        self.orders = remaining

    def finish_bar(self):
        self._update_account(self.last_bars, price_field="close")

    def __call__(self, bars, funding_rates=None, blocked_group_ids=None):
        bars, funding_payments = self.begin_bar(bars, funding_rates)
        blocked_group_ids = blocked_group_ids or set()
        filled_orders = []
        rejected_orders = []
        new_trades = []
        remaining_orders = []

        for group_orders in self._pending_order_groups().values():
            group_id = group_orders[0].group_id
            if group_id in blocked_group_ids or not self._group_has_current_bars(group_orders, bars):
                remaining_orders.extend(group_orders)
                continue
            result = self.execute_group(group_orders, bars)
            if result["waiting"]:
                remaining_orders.extend(group_orders)
                continue
            filled_orders.extend(result["filled_orders"])
            rejected_orders.extend(result["rejected_orders"])
            new_trades.extend(result["new_trades"])

        self.orders = remaining_orders
        self.finish_bar()
        return self._result(filled_orders, rejected_orders, new_trades, funding_payments)

    def execute_group(self, orders, bars, forced_scale: float | None = None):
        """Execute a logical order group atomically.

        If an open group does not fully fit, every leg is reduced by the same
        factor so the planned hedge ratio is preserved.
        """
        status, prepared = self._prepare_group_execution(orders, bars)
        if status == "waiting":
            return self._group_result(waiting=True)
        if status == "rejected":
            rejected = []
            self._reject_group(orders, rejected)
            return self._group_result(rejected_orders=rejected)

        is_open = all(self._value(order.action) == OrderAction.OPEN.value for order, _bar, _price in prepared)
        if is_open:
            scale = self.affordable_open_scale(orders, bars) if forced_scale is None else float(forced_scale)
            scale = max(0.0, min(1.0, scale))
            if scale <= 1e-12:
                rejected = []
                self._reject_group(orders, rejected)
                return self._group_result(rejected_orders=rejected)
            if scale < 1.0 - 1e-12:
                for order, _bar, _price in prepared:
                    if getattr(order, "requested_quantity", None) is None:
                        order.requested_quantity = float(order.quantity)
                    order.quantity = float(order.quantity) * scale
                    order.fill_scale = scale
                status, prepared = self._prepare_group_execution(orders, bars)
                if status != "ready" or any(float(order.quantity or 0.0) <= 1e-12 for order, _bar, _price in prepared):
                    rejected = []
                    self._reject_group(orders, rejected)
                    return self._group_result(rejected_orders=rejected)
            else:
                for order, _bar, _price in prepared:
                    if getattr(order, "requested_quantity", None) is None:
                        order.fill_scale = 1.0

        position_ids = {
            order.position_id or order.group_id or order.order_id
            for order, _bar, _price in prepared
        }
        reserved_before = {
            position_id: self._position_reserved_total(position_id)
            for position_id in position_ids
        }
        capital_before = {
            position_id: self._position_capital_value(position_id)
            for position_id in position_ids
        }
        realized_by_position = {position_id: 0.0 for position_id in position_ids}
        fees_by_position = {position_id: 0.0 for position_id in position_ids}

        trades = []
        filled = []
        for order, bar, price in prepared:
            trade, realized = self._execute_prepared_order(order, bar, price)
            position_id = order.position_id or order.group_id or order.order_id
            realized_by_position[position_id] += realized
            fees_by_position[position_id] += float(trade.fee)
            filled.append(order)
            self.trade_history.append(trade)
            trades.append(trade)
        self._settle_group_capital(
            prepared,
            reserved_before,
            capital_before,
            realized_by_position,
            fees_by_position,
            is_open=is_open,
        )
        # The next group on this bar must see the newly reserved/released margin.
        self._update_account(self.last_bars, price_field="open")
        return self._group_result(filled_orders=filled, new_trades=trades)

    def affordable_open_scale(self, orders, bars) -> float:
        """Return the largest common quantity scale affordable at fill prices."""
        open_orders = [order for order in orders if self._value(order.action) == OrderAction.OPEN.value]
        if not open_orders:
            return 1.0
        if len(open_orders) != len(orders):
            return 0.0
        prepared = []
        for order in open_orders:
            bar = bars.get(order.symbol)
            if bar is None:
                return 0.0
            price = self._projected_order_price(order, bar)
            if price is None:
                return 0.0
            prepared.append((order, price))

        if self._projected_available_balance(prepared, 1.0, bars) >= -1e-8:
            return 1.0
        if self._projected_available_balance(prepared, 0.0, bars) <= 0.0:
            return 0.0
        low, high = 0.0, 1.0
        for _ in range(60):
            mid = (low + high) / 2.0
            if self._projected_available_balance(prepared, mid, bars) >= 0.0:
                low = mid
            else:
                high = mid
        return low

    def _projected_available_balance(self, prepared, scale, bars):
        lots = deepcopy(self.position_lots)
        reserved_margin = deepcopy(self.position_reserved_margin)
        reserved_before = self._reserved_margin_total(reserved_margin)
        for order, price in prepared:
            quantity = float(order.quantity) * float(scale)
            side = self._value(order.side)
            delta = quantity if side == OrderSide.BUY.value else -quantity
            position_id = order.position_id or order.group_id or order.order_id
            self._apply_reserved_margin_change_to(
                reserved_margin, lots, position_id, order.symbol, delta, price
            )
            self._apply_lot_change_to(lots, {}, position_id, order.symbol, delta, price)
        reserved_after = self._reserved_margin_total(reserved_margin)
        additional_reserve = max(0.0, reserved_after - reserved_before)
        return float(self.available_balance) - additional_reserve

    def projected_account_after_orders(self, orders, bars, scale=1.0):
        """Project metrics after proportionally executing open or close orders."""
        free_cash = float(self.available_balance)
        lots = deepcopy(self.position_lots)
        entries = deepcopy(self.position_entry_prices)
        reserved_margin = deepcopy(self.position_reserved_margin)
        capital = deepcopy(self.position_capital)
        positions = deepcopy(self.positions)
        scale = max(0.0, min(1.0, float(scale)))
        affected = {}

        for order in orders:
            bar = bars.get(order.symbol)
            if bar is None:
                return None
            price = self._projected_order_price(order, bar)
            if price is None:
                return None
            quantity = float(order.quantity or 0.0) * scale
            if quantity <= 1e-12:
                continue

            side = self._value(order.side)
            action = self._value(order.action)
            delta = quantity if side == OrderSide.BUY.value else -quantity
            position_id = order.position_id or order.group_id or order.order_id
            fee = price * quantity * self.fee_rate
            if action not in {OrderAction.OPEN.value, OrderAction.CLOSE.value}:
                return None

            info = affected.setdefault(
                position_id,
                {
                    "before_reserve": sum(
                        max(0.0, float(value))
                        for value in reserved_margin.get(position_id, {}).values()
                    ),
                    "before_capital": float(
                        capital.get(
                            position_id,
                            sum(
                                max(0.0, float(value))
                                for value in reserved_margin.get(position_id, {}).values()
                            ),
                        )
                    ),
                    "realized": 0.0,
                    "fees": 0.0,
                    "actions": set(),
                },
            )
            info["actions"].add(action)
            info["fees"] += fee
            if action == OrderAction.CLOSE.value:
                info["realized"] += self._realized_pnl_from(
                    lots, entries, position_id, order.symbol, delta, price
                )

            self._apply_reserved_margin_change_to(
                reserved_margin, lots, position_id, order.symbol, delta, price
            )
            self._apply_lot_change_to(
                lots, entries, position_id, order.symbol, delta, price
            )
            positions[order.symbol] = positions.get(order.symbol, 0.0) + delta
            if abs(positions[order.symbol]) < 1e-12:
                positions.pop(order.symbol, None)

        for position_id, info in affected.items():
            after_reserve = sum(
                max(0.0, float(value))
                for value in reserved_margin.get(position_id, {}).values()
            )
            before_reserve = float(info["before_reserve"])
            before_capital = float(info["before_capital"])
            if info["actions"] == {OrderAction.OPEN.value}:
                added = max(0.0, after_reserve - before_reserve)
                free_cash -= added
                capital[position_id] = before_capital + added - float(info["fees"])
                continue

            capital_after = before_capital + float(info["realized"]) - float(info["fees"])
            if after_reserve <= 1e-12:
                free_cash += capital_after
                capital.pop(position_id, None)
                continue
            released = max(0.0, before_reserve - after_reserve)
            fraction = min(1.0, released / before_reserve) if before_reserve > 1e-12 else 0.0
            release = capital_after * fraction
            free_cash += release
            capital[position_id] = capital_after - release

        return self._account_metrics(
            bars,
            price_field="open",
            position_lots=lots,
            entry_prices=entries,
            reserved_margin=reserved_margin,
            positions=positions,
            available_balance=free_cash,
            position_capital=capital,
        )

    def _execute_prepared_order(self, order, bar, price):
        quantity = float(order.quantity)
        notional = price * quantity
        fee = notional * self.fee_rate
        slippage = abs(price - bar["open"]) * quantity
        side = self._value(order.side)
        action = self._value(order.action)
        position_id = order.position_id or order.group_id or order.order_id
        order.position_id = position_id
        realized = self._apply_trade(
            order.symbol, side, quantity, price, fee, action, position_id,
            pair_id=order.pair_id,
        )
        order.status = OrderStatus.FILLED
        return Trade(
            order_id=order.order_id,
            group_id=order.group_id,
            exchange=order.exchange,
            symbol=order.symbol,
            action=action,
            position_id=position_id,
            pair_id=order.pair_id,
            side=side,
            quantity=quantity,
            price=price,
            notional=notional,
            fee=fee,
            slippage=slippage,
            ts=bar["ts"],
            target_hedge_ratio=order.target_hedge_ratio,
            exit_reason=order.exit_reason,
            protection_trigger=order.protection_trigger,
            exit_class=order.exit_class,
            reopen_lock_pending=order.reopen_lock_pending,
            protection_max_holding_bars=order.protection_max_holding_bars,
            protection_max_holding_deadline_bar=order.protection_max_holding_deadline_bar,
            protection_pair_loss_stop_return=order.protection_pair_loss_stop_return,
            protection_pair_loss_stop_freeze_bars=order.protection_pair_loss_stop_freeze_bars,
            protection_pair_loss_stop_freeze_until_bar=order.protection_pair_loss_stop_freeze_until_bar,
            protection_rule=order.protection_rule,
            protection_freeze_bars=order.protection_freeze_bars,
            protection_freeze_until_bar=order.protection_freeze_until_bar,
            requested_quantity=order.requested_quantity,
            fill_scale=order.fill_scale,
            para=order.para,
            rebalance_batch_id=order.rebalance_batch_id,
        ), realized

    def _prepare_group_execution(self, orders, bars):
        prepared = []
        positions = dict(self.positions)
        position_lots = {position_id: dict(lots) for position_id, lots in self.position_lots.items()}
        for order in orders:
            bar = bars.get(order.symbol)
            if bar is None:
                return "waiting", []
            price = self._get_trade_price(order, bar)
            if price is None:
                return ("rejected" if order.status == OrderStatus.REJECTED else "waiting"), []
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
            return quantity > 0
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
        positions[order.symbol] = positions.get(order.symbol, 0.0) + delta
        if abs(positions[order.symbol]) < 1e-12:
            positions.pop(order.symbol, None)
        if not order.position_id or action not in {OrderAction.OPEN.value, OrderAction.CLOSE.value}:
            return
        lots = position_lots.setdefault(order.position_id, {})
        lots[order.symbol] = lots.get(order.symbol, 0.0) + delta
        if abs(lots[order.symbol]) < 1e-12:
            lots.pop(order.symbol, None)
        if not lots:
            position_lots.pop(order.position_id, None)

    def _apply_trade(
        self, symbol, side, quantity, price, fee, action=None,
        position_id=None, pair_id=None,
    ):
        delta = quantity if side == OrderSide.BUY.value else -quantity
        realized = 0.0
        if action == OrderAction.OPEN.value:
            if position_id and pair_id:
                self.position_pair_ids[position_id] = pair_id
            self._apply_reserved_margin_change_to(
                self.position_reserved_margin,
                self.position_lots,
                position_id,
                symbol,
                delta,
                price,
            )
            self._apply_lot_change_to(
                self.position_lots, self.position_entry_prices,
                position_id, symbol, delta, price,
            )
        elif action == OrderAction.CLOSE.value:
            realized = self._realized_pnl(position_id, symbol, delta, price)
            self._apply_reserved_margin_change_to(
                self.position_reserved_margin,
                self.position_lots,
                position_id,
                symbol,
                delta,
                price,
            )
            self._apply_lot_change_to(
                self.position_lots, self.position_entry_prices,
                position_id, symbol, delta, price,
            )
        else:
            raise ValueError(f"unsupported trade action: {action!r}")
        self.positions[symbol] = self.positions.get(symbol, 0.0) + delta
        if abs(self.positions[symbol]) < 1e-12:
            self.positions.pop(symbol, None)
        if position_id and position_id not in self.position_lots:
            self.position_pair_ids.pop(position_id, None)
            self.position_reserved_margin.pop(position_id, None)
        return float(realized)

    def _realized_pnl(self, position_id, symbol, close_delta, close_price):
        return self._realized_pnl_from(
            self.position_lots,
            self.position_entry_prices,
            position_id,
            symbol,
            close_delta,
            close_price,
        )

    @staticmethod
    def _realized_pnl_from(lots, entries, position_id, symbol, close_delta, close_price):
        lot_quantity = lots.get(position_id, {}).get(symbol, 0.0)
        entry_price = entries.get(position_id, {}).get(symbol)
        if abs(lot_quantity) < 1e-12 or entry_price is None:
            return 0.0
        close_quantity = min(abs(float(close_delta)), abs(float(lot_quantity)))
        direction = 1.0 if lot_quantity > 0 else -1.0
        return close_quantity * (float(close_price) - float(entry_price)) * direction

    @staticmethod
    def _apply_lot_change_to(lots, entries, position_id, symbol, delta, price):
        if not position_id:
            return
        position_lot = lots.setdefault(position_id, {})
        entry_lot = entries.setdefault(position_id, {})
        old_quantity = float(position_lot.get(symbol, 0.0))
        new_quantity = old_quantity + float(delta)
        if abs(new_quantity) < 1e-12:
            position_lot.pop(symbol, None)
            entry_lot.pop(symbol, None)
        elif abs(old_quantity) < 1e-12 or old_quantity * delta > 0:
            old_abs = abs(old_quantity)
            delta_abs = abs(float(delta))
            old_entry = float(entry_lot.get(symbol, price))
            entry_lot[symbol] = (old_entry * old_abs + float(price) * delta_abs) / (old_abs + delta_abs)
            position_lot[symbol] = new_quantity
        else:
            position_lot[symbol] = new_quantity
        if not position_lot:
            lots.pop(position_id, None)
        if not entry_lot:
            entries.pop(position_id, None)

    def _apply_reserved_margin_change_to(
        self, reserved_margin, lots, position_id, symbol, delta, price
    ):
        """Reserve entry capital and release it in proportion to closed size.

        Both long and short futures legs consume capital.  The reserve is based
        on the actual execution notional, not on later mark prices.  This is
        the account invariant required by the target-capital portfolio model:
        unrealized PnL changes equity, while free cash remains ring-fenced.
        """
        if not position_id:
            return
        delta = float(delta)
        if abs(delta) <= 1e-12:
            return

        old_quantity = float(lots.get(position_id, {}).get(symbol, 0.0))
        position_margin = reserved_margin.setdefault(position_id, {})
        old_margin = float(position_margin.get(symbol, 0.0) or 0.0)

        if abs(old_quantity) <= 1e-12 or old_quantity * delta > 0:
            position_margin[symbol] = (
                old_margin + abs(delta * float(price)) / self.max_leverage
            )
            return

        close_quantity = min(abs(delta), abs(old_quantity))
        release_fraction = close_quantity / abs(old_quantity)
        remaining_margin = old_margin * max(0.0, 1.0 - release_fraction)
        if remaining_margin <= 1e-12:
            position_margin.pop(symbol, None)
        else:
            position_margin[symbol] = remaining_margin
        if not position_margin:
            reserved_margin.pop(position_id, None)

    @staticmethod
    def _reserved_margin_total(reserved_margin) -> float:
        return sum(
            max(0.0, float(value))
            for symbols in reserved_margin.values()
            for value in symbols.values()
        )

    def _position_reserved_total(self, position_id: str) -> float:
        return sum(
            max(0.0, float(value))
            for value in self.position_reserved_margin.get(position_id, {}).values()
        )

    def _position_capital_value(self, position_id: str) -> float:
        if position_id in self.position_capital:
            return float(self.position_capital[position_id])
        # Compatibility for fixtures/checkpoints created before isolated Pair
        # capital was explicit: the fixed entry reserve is the Pair capital.
        return self._position_reserved_total(position_id)

    def _settle_group_capital(
        self,
        prepared,
        reserved_before,
        capital_before,
        realized_by_position,
        fees_by_position,
        *,
        is_open: bool,
    ) -> None:
        """Apply one atomic group's cash movement to isolated Pair accounts."""
        position_ids = {
            order.position_id or order.group_id or order.order_id
            for order, _bar, _price in prepared
        }
        if is_open:
            total_added = 0.0
            for position_id in position_ids:
                before_reserve = float(reserved_before.get(position_id, 0.0))
                after_reserve = self._position_reserved_total(position_id)
                added = max(0.0, after_reserve - before_reserve)
                total_added += added
                self.position_capital[position_id] = (
                    float(capital_before.get(position_id, 0.0))
                    + added
                    - float(fees_by_position.get(position_id, 0.0))
                )
            if total_added > float(self.available_balance) + 1e-7:
                raise RuntimeError(
                    "open execution exceeded isolated available balance: "
                    f"required={total_added:.12f} available={self.available_balance:.12f}"
                )
            self.available_balance = max(0.0, float(self.available_balance) - total_added)
        else:
            for position_id in position_ids:
                before_reserve = float(reserved_before.get(position_id, 0.0))
                after_reserve = self._position_reserved_total(position_id)
                released_reserve = max(0.0, before_reserve - after_reserve)
                capital_after_pnl = (
                    float(capital_before.get(position_id, before_reserve))
                    + float(realized_by_position.get(position_id, 0.0))
                    - float(fees_by_position.get(position_id, 0.0))
                )
                if after_reserve <= 1e-12:
                    release = capital_after_pnl
                    self.available_balance += release
                    self.position_capital.pop(position_id, None)
                    continue

                fraction = (
                    min(1.0, released_reserve / before_reserve)
                    if before_reserve > 1e-12 else 0.0
                )
                release = capital_after_pnl * fraction
                self.available_balance += release
                self.position_capital[position_id] = capital_after_pnl - release

        self._sync_wallet_balance()

    def _sync_wallet_balance(self) -> None:
        # Pair capital remains visible in the account ledger, including a
        # negative value until the Pair is liquidated/closed.  This prevents a
        # loss beyond the isolated reserve from being silently written off.
        self.wallet_balance = float(self.available_balance) + sum(
            float(value) for value in self.position_capital.values()
        )
        self.cash = self.wallet_balance

    def _account_metrics(
        self, bars, price_field="close", wallet_balance=None,
        position_lots=None, entry_prices=None, reserved_margin=None,
        positions=None, available_balance=None, position_capital=None,
    ):
        wallet = self.wallet_balance if wallet_balance is None else float(wallet_balance)
        free_cash = (
            self.available_balance
            if available_balance is None else float(available_balance)
        )
        lots = self.position_lots if position_lots is None else position_lots
        entries = self.position_entry_prices if entry_prices is None else entry_prices
        margin_lots = (
            self.position_reserved_margin
            if reserved_margin is None else reserved_margin
        )
        capital_map = self.position_capital if position_capital is None else position_capital
        aggregate_positions = self.positions if positions is None else positions
        unrealized = 0.0
        unrealized_by_position: dict[str, float] = {}
        gross = 0.0
        net = 0.0
        compatibility_margin = 0.0
        covered: dict[str, float] = {}
        for position_id, symbols in lots.items():
            for symbol, quantity in symbols.items():
                mark = self._mark_price(symbol, bars, price_field)
                if mark is None:
                    continue
                quantity = float(quantity)
                entry = float(entries.get(position_id, {}).get(symbol, mark))
                position_pnl = quantity * (mark - entry)
                unrealized += position_pnl
                unrealized_by_position[position_id] = (
                    unrealized_by_position.get(position_id, 0.0) + position_pnl
                )
                gross += abs(quantity * mark)
                net += quantity * mark
                covered[symbol] = covered.get(symbol, 0.0) + quantity
                if symbol not in margin_lots.get(position_id, {}):
                    # Legacy fixtures/checkpoints may predate explicit margin
                    # lots. Reconstruct their fixed reserve from entry value,
                    # never from the current mark.
                    compatibility_margin += (
                        abs(quantity * entry) / self.max_leverage
                    )
        # Compatibility for tests/callers that seed aggregate positions only.
        for symbol, quantity in aggregate_positions.items():
            residual = float(quantity) - covered.get(symbol, 0.0)
            if abs(residual) < 1e-12:
                continue
            mark = self._mark_price(symbol, bars, price_field)
            if mark is None:
                continue
            gross += abs(residual * mark)
            net += residual * mark
            compatibility_margin += abs(residual * mark) / self.max_leverage
        equity = wallet + unrealized
        used_margin = sum(
            max(0.0, float(value))
            for symbols in margin_lots.values()
            for value in symbols.values()
        ) + compatibility_margin
        isolated_account = bool(margin_lots or capital_map)
        if isolated_account:
            position_ids = set(lots) | set(margin_lots) | set(capital_map)
            realized_capital = 0.0
            marked_capital = 0.0
            for position_id in position_ids:
                if position_id in capital_map:
                    capital = float(capital_map[position_id])
                else:
                    capital = sum(
                        max(0.0, float(value))
                        for value in margin_lots.get(position_id, {}).values()
                    )
                realized_capital += capital
                marked_capital += capital + float(unrealized_by_position.get(position_id, 0.0))
            wallet = free_cash + realized_capital
            equity = free_cash + marked_capital
            reported_available = free_cash
        else:
            # Compatibility for callers that seed aggregate/legacy positions
            # without the isolated position ledgers.
            equity = wallet + unrealized
            reported_available = max(0.0, wallet - used_margin)
        return {
            "wallet_balance": wallet,
            "unrealized_pnl": unrealized,
            "equity": equity,
            "gross_exposure": gross,
            "net_exposure": net,
            "used_margin": used_margin,
            "available_balance": reported_available,
            "position_marked_equity": {
                position_id: float(capital_map.get(position_id, sum(
                    max(0.0, float(value))
                    for value in margin_lots.get(position_id, {}).values()
                ))) + float(unrealized_by_position.get(position_id, 0.0))
                for position_id in (set(lots) | set(margin_lots) | set(capital_map))
            },
        }

    def _update_account(self, bars, price_field="close"):
        metrics = self._account_metrics(bars, price_field=price_field)
        self.wallet_balance = metrics["wallet_balance"]
        self.cash = self.wallet_balance
        self.unrealized_pnl = metrics["unrealized_pnl"]
        self.equity = metrics["equity"]
        self.gross_exposure = metrics["gross_exposure"]
        self.net_exposure = metrics["net_exposure"]
        self.used_margin = metrics["used_margin"]
        self.available_balance = metrics["available_balance"]
        self.position_marked_equity = metrics["position_marked_equity"]

    def _mark_price(self, symbol, bars, price_field):
        bar = bars.get(symbol) if isinstance(bars, dict) else None
        if bar is None:
            bar = self.last_bars.get(symbol)
        if bar is None:
            return None
        if isinstance(bar, dict):
            return float(bar[price_field])
        return float(bar[1 if price_field == "open" else 3])

    def _apply_funding(self, bars, funding_rates):
        payments = []
        for event in self._funding_events(funding_rates):
            payments.extend(self._apply_funding_rates(bars, event["rates"], event.get("ts")))
        return payments

    def _apply_funding_rates(self, bars, funding_rates, event_ts=None):
        payments = []
        for symbol, funding_rate in funding_rates.items():
            bar = bars.get(symbol)
            if bar is None:
                continue
            # Funding is settled during begin_bar, so the current bar's close is
            # not known yet. Use the executable/open price to avoid look-ahead.
            mark_price = float(bar["open"])
            for position_id, symbols in list(self.position_lots.items()):
                quantity = float(symbols.get(symbol, 0.0) or 0.0)
                if abs(quantity) < 1e-12:
                    continue
                notional = abs(quantity * mark_price)
                payment = quantity * mark_price * float(funding_rate)
                current_capital = self._position_capital_value(position_id)
                self.position_capital[position_id] = current_capital - payment
                funding_payment = FundingPayment(
                    exchange=self.exchange_name,
                    symbol=symbol,
                    ts=event_ts if event_ts is not None else bar["ts"],
                    funding_rate=float(funding_rate),
                    quantity=quantity,
                    mark_price=mark_price,
                    notional=notional,
                    payment=payment,
                    position_id=position_id,
                    pair_id=self.position_pair_ids.get(position_id),
                )
                self.funding_history.append(funding_payment)
                payments.append(funding_payment)
        self._sync_wallet_balance()
        return payments

    @staticmethod
    def _funding_events(funding_rates):
        if not funding_rates:
            return []
        if isinstance(funding_rates, list):
            return funding_rates
        if isinstance(funding_rates, tuple):
            return list(funding_rates)
        if isinstance(funding_rates, dict) and "rates" in funding_rates:
            return [funding_rates]
        return [{"ts": None, "rates": funding_rates}]

    def _pending_order_groups(self):
        groups = {}
        for order in self.orders:
            groups.setdefault(order.group_id or order.order_id, []).append(order)
        return groups

    @staticmethod
    def _group_has_current_bars(orders, bars):
        return all(order.symbol in bars for order in orders)

    @staticmethod
    def _reject_group(orders, rejected_orders):
        for order in orders:
            order.status = OrderStatus.REJECTED
            rejected_orders.append(order)

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

    def _format_bars(self, bars):
        if isinstance(bars, list):
            symbols = {order.symbol for order in self.orders}
            if len(symbols) != 1:
                raise ValueError("single bar can only be used when pending orders have one symbol")
            return {next(iter(symbols)): self._bar_to_dict(bars)}
        return {symbol: self._bar_to_dict(bar) for symbol, bar in bars.items()}

    @staticmethod
    def _bar_to_dict(bar):
        if isinstance(bar, dict):
            return bar
        if not isinstance(bar, list) or len(bar) != len(BAR_COLUMNS):
            raise ValueError("bar must be [ts, open, high, close, low, volume]")
        return dict(zip(BAR_COLUMNS, bar))

    @staticmethod
    def _value(value):
        return value.value if hasattr(value, "value") else value

    def _result(self, filled_orders, rejected_orders, new_trades, funding_payments):
        return {
            "filled_orders": filled_orders,
            "rejected_orders": rejected_orders,
            "new_trades": new_trades,
            "funding_payments": funding_payments,
            "trades": self.trade_history,
            "funding_history": self.funding_history,
            "has_fill": bool(filled_orders),
            "has_open_orders": bool(self.orders),
            "cash": self.wallet_balance,
            "wallet_balance": self.wallet_balance,
            "unrealized_pnl": self.unrealized_pnl,
            "used_margin": self.used_margin,
            "available_balance": self.available_balance,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "positions": dict(self.positions) if self.copy_positions_on_bar else self.positions,
            "equity": self.equity,
            "position_marked_equity": dict(self.position_marked_equity),
        }

    @staticmethod
    def _group_result(waiting=False, filled_orders=None, rejected_orders=None, new_trades=None):
        return {
            "waiting": waiting,
            "filled_orders": filled_orders or [],
            "rejected_orders": rejected_orders or [],
            "new_trades": new_trades or [],
        }
