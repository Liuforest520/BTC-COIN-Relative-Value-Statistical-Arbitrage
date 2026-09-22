from core.modules.exchange.exchange import Exchange
from core.modules.models import (
    Order,
    OrderStatus,
)


class ExchangeManager:
    def __init__(self, exchanges: dict[str, Exchange], pending_timeout_bars: int = 0):
        self.exchanges = exchanges
        self.initial_cash = sum(exchange.initial_cash for exchange in exchanges.values())
        self.equity_curve = []
        self.funding_payments = []
        self.compact_equity_curve = False
        self.include_trade_history_on_bar = True
        self.pending_timeout_bars = max(0, int(pending_timeout_bars))
        self._bar_index = 0
        self._group_first_seen: dict[str, int] = {}

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
        self._bar_index += 1
        results = {}
        new_trades = []
        rejected_orders = []
        positions = {}
        funding_payments = []
        funding_rates = funding_rates or {}
        formatted_bars = {}
        per_exchange = {}
        for exchange_name, exchange in self.exchanges.items():
            bars, payments = exchange.begin_bar(
                bars_by_exchange.get(exchange_name, {}),
                funding_rates=funding_rates,
            )
            formatted_bars[exchange_name] = bars
            funding_payments.extend(payments)
            per_exchange[exchange_name] = {
                "filled_orders": [],
                "rejected_orders": [],
                "new_trades": [],
                "funding_payments": payments,
            }

        # A Pair whose isolated account is already at or below zero must be
        # liquidated before any other pending order is considered.  The
        # exchange marks at the executable open in ``begin_bar`` above, so
        # these orders are filled through the same atomic close path at this
        # bar's open.  This realizes the loss instead of allowing the Pair to
        # remain as a free option until it happens to recover.
        forced_orders_by_exchange = {}
        forced_order_ids = set()
        for exchange_name, exchange in self.exchanges.items():
            forced_orders = exchange.forced_liquidation_orders(
                formatted_bars[exchange_name], self._bar_index
            )
            if not forced_orders:
                continue
            accepted = exchange.place_order(forced_orders)
            forced_orders_by_exchange[exchange_name] = accepted
            forced_order_ids.update(order.order_id for order in accepted)

        # Execute complete logical Pair groups sequentially.  This makes the
        # available margin consumed by an earlier group visible to every later
        # group on the same bar and keeps cross-exchange legs atomic.
        pending_groups = self._pending_order_groups()
        # A rebalance close is a prerequisite for its replacement opens.  If
        # one of its legs has no executable bar yet, keep all newly queued
        # opens pending rather than opening without the planned released
        # margin.  Normal close/open groups are unaffected.
        waiting_rebalance_batches = {
            next((getattr(o, "rebalance_batch_id", None) for o in group if getattr(o, "rebalance_batch_id", None)), None)
            for group in pending_groups.values()
            if {getattr(o.action, "value", o.action) for o in group} == {"close"}
            and any(getattr(o, "protection_trigger", None) == "rebalance_replacement" for o in group)
            and any(o.symbol not in formatted_bars.get(o.exchange, {}) for o in group)
        }
        waiting_rebalance_batches.discard(None)
        failed_rebalance_batches = set()
        for group_id, group_orders in pending_groups.items():
            self._group_first_seen.setdefault(group_id, self._bar_index)
            if self.pending_timeout_bars and self._bar_index - self._group_first_seen[group_id] >= self.pending_timeout_bars:
                rejected = self._reject_group_orders({group_id})
                rejected_orders.extend(rejected)
                for order in rejected:
                    per_exchange[order.exchange]["rejected_orders"].append(order)
                self._group_first_seen.pop(group_id, None)
                if any(getattr(o, "protection_trigger", None) == "rebalance_replacement" for o in group_orders):
                    failed_rebalance_batches.update(
                        batch_id
                        for batch_id in (getattr(o, "rebalance_batch_id", None) for o in group_orders)
                        if batch_id is not None
                    )
                continue
            actions = {getattr(o.action, "value", o.action) for o in group_orders}
            batch_ids = {getattr(o, "rebalance_batch_id", None) for o in group_orders}
            if actions == {"open"} and batch_ids.intersection(failed_rebalance_batches):
                rejected = self._reject_group_orders({group_id})
                rejected_orders.extend(rejected)
                for order in rejected:
                    per_exchange[order.exchange]["rejected_orders"].append(order)
                self._group_first_seen.pop(group_id, None)
                continue
            if actions == {"open"} and batch_ids.intersection(waiting_rebalance_batches):
                continue
            local_groups = {}
            waiting = False
            invalid = False
            for order in group_orders:
                exchange = self.exchanges.get(order.exchange)
                bars = formatted_bars.get(order.exchange, {})
                if exchange is None or order.symbol not in bars:
                    waiting = True
                    break
                local_groups.setdefault(order.exchange, []).append(order)
            if waiting:
                continue

            for exchange_name, local_orders in local_groups.items():
                status, _prepared = self.exchanges[exchange_name]._prepare_group_execution(
                    local_orders, formatted_bars[exchange_name]
                )
                if status != "ready":
                    invalid = True
                    break
            if invalid:
                rejected = self._reject_group_orders({group_id})
                rejected_orders.extend(rejected)
                for order in rejected:
                    per_exchange[order.exchange]["rejected_orders"].append(order)
                if any(getattr(o, "protection_trigger", None) == "rebalance_replacement" for o in group_orders):
                    failed_rebalance_batches.update(
                        batch_id
                        for batch_id in (getattr(o, "rebalance_batch_id", None) for o in group_orders)
                        if batch_id is not None
                    )
                continue

            actions = {
                getattr(order.action, "value", order.action)
                for order in group_orders
            }
            common_scale = None
            if actions == {"open"}:
                pair_ids = {order.pair_id for order in group_orders if order.pair_id}
                pair_id = next(iter(pair_ids)) if len(pair_ids) == 1 else None
                margin_scale = min(
                    self.exchanges[exchange_name].affordable_open_scale(
                        local_orders, formatted_bars[exchange_name]
                    )
                    for exchange_name, local_orders in local_groups.items()
                )
                common_scale = margin_scale
                if common_scale <= 1e-12:
                    rejected = self._reject_group_orders({group_id})
                    rejected_orders.extend(rejected)
                    for order in rejected:
                        per_exchange[order.exchange]["rejected_orders"].append(order)
                    continue
                if common_scale < 1.0 - 1e-12:
                    for order in group_orders:
                        if getattr(order, "requested_quantity", None) is None:
                            order.requested_quantity = float(order.quantity)
                        order.quantity = float(order.quantity) * common_scale
                        order.fill_scale = common_scale

            group_failed = False
            group_results = {}
            for exchange_name, local_orders in local_groups.items():
                result = self.exchanges[exchange_name].execute_group(
                    local_orders,
                    formatted_bars[exchange_name],
                    forced_scale=1.0 if common_scale is not None else None,
                )
                group_results[exchange_name] = result
                if result["waiting"] or result["rejected_orders"]:
                    group_failed = True
                    break
            if group_failed:
                # Pre-validation and a common scale make this path defensive;
                # no group should partially execute here.
                raise RuntimeError(f"atomic order group {group_id} failed after pre-validation")

            self._remove_group_orders(group_id)
            self._group_first_seen.pop(group_id, None)
            for exchange_name, result in group_results.items():
                per_exchange[exchange_name]["filled_orders"].extend(result["filled_orders"])
                per_exchange[exchange_name]["new_trades"].extend(result["new_trades"])
                new_trades.extend(result["new_trades"])

        forced_filled_orders = [
            order
            for exchange_name in forced_orders_by_exchange
            for order in per_exchange[exchange_name]["filled_orders"]
            if order.order_id in forced_order_ids
        ]

        for exchange in self.exchanges.values():
            exchange.finish_bar()

        for exchange_name, exchange in self.exchanges.items():
            result = exchange._result(**per_exchange[exchange_name])
            results[exchange_name] = result
            positions[exchange_name] = result["positions"]

        self.funding_payments.extend(funding_payments)
        portfolio_state = self._portfolio_state(results)
        ts = self._current_ts(bars_by_exchange)
        if ts is not None:
            if self.compact_equity_curve:
                self.equity_curve["ts"].append(ts)
                self.equity_curve["equity"].append(portfolio_state["equity"])
            else:
                self.equity_curve.append({"ts": ts, "equity": portfolio_state["equity"]})

        return {
            "results": results,
            "new_trades": new_trades,
            "rejected_orders": rejected_orders,
            "funding_payments": funding_payments,
            # Detailed backtests expose the cumulative trade snapshot for
            # compatibility. Sweeps disable it because rebuilding the full
            # history on every bar makes a run quadratic in its trade count.
            "trades": self._all_trades(results) if self.include_trade_history_on_bar else [],
            "positions": positions,
            "cash": portfolio_state["cash"],
            "equity": portfolio_state["equity"],
            "wallet_balance": portfolio_state["wallet_balance"],
            "unrealized_pnl": portfolio_state["unrealized_pnl"],
            "used_margin": portfolio_state["used_margin"],
            "available_balance": portfolio_state["available_balance"],
            "gross_exposure": portfolio_state["gross_exposure"],
            "net_exposure": portfolio_state["net_exposure"],
            "open_pair_count": portfolio_state["open_pair_count"],
            "open_pair_ids": portfolio_state["open_pair_ids"],
            "pending_open_pair_count": portfolio_state["pending_open_pair_count"],
            # These fields are retained for report compatibility and now
            # describe actual Pair-equity-zero liquidations.
            "margin_deficit": 0.0,
            "forced_deleveraging_triggered": bool(forced_filled_orders),
            "forced_deleveraging_scale": 1.0 if forced_filled_orders else 0.0,
            "forced_deleveraging_orders": forced_filled_orders,
            "initial_cash": self.initial_cash,
            "equity_curve": self.equity_curve,
            "has_fill": len(new_trades) > 0,
        }

    @classmethod
    def from_names(cls, exchange_names, initial_cash=100000.0, fee_rate=0.0005, slippage_bps=1.0, max_leverage=1.0, pending_timeout_bars=0):
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
                max_leverage=max_leverage,
            )
        return cls(exchanges, pending_timeout_bars=pending_timeout_bars)

    def _portfolio_state(self, results):
        wallet_balance = sum(float(result["wallet_balance"]) for result in results.values())
        unrealized_pnl = sum(float(result["unrealized_pnl"]) for result in results.values())
        used_margin = sum(float(result["used_margin"]) for result in results.values())
        gross_exposure = sum(float(result["gross_exposure"]) for result in results.values())
        net_exposure = sum(float(result["net_exposure"]) for result in results.values())
        # Rebuild from each exchange's isolated-ledger equity.  Pair losses are
        # no longer floored at zero: an insolvent Pair is liquidated at the
        # executable open and its realized shortfall is charged to free cash.
        equity = sum(float(result["equity"]) for result in results.values())
        open_pair_ids = sorted({
            pair_id
            for exchange in self.exchanges.values()
            for position_id, pair_id in exchange.position_pair_ids.items()
            if position_id in exchange.position_lots
        })
        pending_open_pair_ids = sorted({
            order.pair_id
            for exchange in self.exchanges.values()
            for order in exchange.orders
            if order.pair_id
            and getattr(order.action, "value", order.action) == "open"
        })
        return {
            "cash": wallet_balance,
            "wallet_balance": wallet_balance,
            "unrealized_pnl": unrealized_pnl,
            "used_margin": used_margin,
            "available_balance": sum(
                float(result["available_balance"])
                for result in results.values()
            ),
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "equity": equity,
            "open_pair_count": len(open_pair_ids),
            "open_pair_ids": open_pair_ids,
            "pending_open_pair_count": len(set(pending_open_pair_ids).difference(open_pair_ids)),
        }

    def _pair_gross_exposure(self, pair_id, formatted_bars):
        gross = 0.0
        for exchange_name, exchange in self.exchanges.items():
            bars = formatted_bars.get(exchange_name, {})
            for position_id, symbols in exchange.position_lots.items():
                if exchange.position_pair_ids.get(position_id) != pair_id:
                    continue
                for symbol, quantity in symbols.items():
                    mark = exchange._mark_price(symbol, bars, "open")
                    if mark is not None:
                        gross += abs(float(quantity) * mark)
        return gross

    def _current_open_pair_ids(self):
        return {
            pair_id
            for exchange in self.exchanges.values()
            for position_id, pair_id in exchange.position_pair_ids.items()
            if position_id in exchange.position_lots
        }

    def _pending_order_groups(self):
        groups = {}
        for exchange in self.exchanges.values():
            for order in exchange.orders:
                groups.setdefault(order.group_id or order.order_id, []).append(order)

        # Rebalance emits close groups and replacement open groups together.
        # Always release margin first, even when the two legs belong to
        # different exchanges whose local order lists have different layouts.
        def priority(item):
            orders = item[1]
            actions = {getattr(o.action, "value", o.action) for o in orders}
            if any(
                getattr(o, "protection_trigger", None) == "pair_equity_zero"
                for o in orders
            ):
                return -1
            if actions == {"close"}:
                return 0
            if actions == {"open"}:
                return 1
            return 2

        return dict(sorted(groups.items(), key=priority))

    def _remove_group_orders(self, group_id):
        for exchange in self.exchanges.values():
            exchange.orders = [
                order for order in exchange.orders
                if (order.group_id or order.order_id) != group_id
            ]

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

    def _reject_group_orders(self, group_ids):
        rejected = []
        if not group_ids:
            return rejected
        for exchange in self.exchanges.values():
            remaining = []
            for order in exchange.orders:
                order_group_id = order.group_id or order.order_id
                if order_group_id in group_ids:
                    order.status = OrderStatus.REJECTED
                    # place_order() already inserted the same Order object in
                    # order_history. Mutating its status is enough; appending
                    # it again would duplicate rejected rows in reports.
                    rejected.append(order)
                else:
                    remaining.append(order)
            exchange.orders = remaining
        return rejected
