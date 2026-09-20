"""OrderPlanner: convert AllocatedPairTarget to Order objects."""
from __future__ import annotations

from uuid import uuid4

from core.modules.models import Order, OrderAction, OrderSide, OrderType
from core.modules.models.pipeline_types import AllocatedPairTarget
from core.modules.strategy.config import ExecutionConfig, PairDefinition


class OrderPlanner:
    """Convert AllocatedPairTarget objects into Exchange Order objects."""

    def __init__(self, execution_cfg: ExecutionConfig | None = None):
        self.cfg = execution_cfg or ExecutionConfig()
        ot = self.cfg.order_type
        self.order_type = OrderType.LIMIT if ot == "limit" else OrderType.MARKET

    def orders_for_target(self, allocated, pair_def: PairDefinition,
                          x_exchange: str = "binance", y_exchange: str = "binance",
                          hedge_ratio: float = 1.0,
                          action: str = "open",
                          position_id: str | None = None,
                          exit_reason: str | None = None,
                          protection_trigger: str | None = None,
                          exit_class: str | None = None,
                          reopen_lock_pending: bool | None = None,
                          protection_max_holding_bars: int | None = None,
                          protection_max_holding_deadline_bar: int | None = None,
                          protection_pair_loss_stop_return: float | None = None,
                          protection_pair_loss_stop_freeze_bars: int | None = None,
                          protection_pair_loss_stop_freeze_until_bar: int | None = None,
                          protection_rule: str | None = None,
                          protection_freeze_bars: int | None = None,
                          protection_freeze_until_bar: int | None = None,
                          rebalance_batch_id: str | None = None) -> list[Order]:
        if not allocated.selected and action != "close":
            return []

        order_action = OrderAction.OPEN if action == "open" else OrderAction.CLOSE
        side = allocated.side
        target_hedge_ratio = getattr(allocated, "hedge_ratio", None)
        if target_hedge_ratio is None:
            target_hedge_ratio = hedge_ratio
        group_id = str(uuid4())[:8]
        order_position_id = position_id
        if action == "open" and order_position_id is None:
            order_position_id = str(uuid4())[:8]
        orders = []
        reverse = (action == "close")
        raw_para = getattr(allocated, "para", {}) or {}
        portfolio_para = raw_para.get("portfolio", {}) if isinstance(raw_para, dict) else {}
        order_para = {}
        if isinstance(raw_para, dict) and isinstance(raw_para.get("portfolio"), dict):
            rebalance_batch_id = rebalance_batch_id or raw_para["portfolio"].get("rebalance_batch_id")
        if isinstance(raw_para, dict) and raw_para.get("protection"):
            # Keep only the small frozen-model snapshot needed after the
            # signal has become a filled order; do not retain full diagnostic
            # payloads on every Order/Trade object.
            order_para["protection"] = dict(raw_para["protection"])

        # long_x / short_spread: buy X, sell Y. short_x / long_spread: sell X, buy Y.
        if side in ("long_x", "short_spread"):
            x_side = OrderSide.SELL if reverse else OrderSide.BUY
            y_side = OrderSide.BUY if reverse else OrderSide.SELL
        elif side in ("short_x", "long_spread"):
            x_side = OrderSide.BUY if reverse else OrderSide.SELL
            y_side = OrderSide.SELL if reverse else OrderSide.BUY
        else:
            return []

        if allocated.final_x_quantity > 0:
            px = None if self.order_type == OrderType.MARKET else (
                allocated.final_x_notional / abs(allocated.final_x_quantity)
                if abs(allocated.final_x_quantity) > 1e-12 and allocated.final_x_notional > 1e-12 else None
            )
            orders.append(Order(
                order_id=str(uuid4())[:8], group_id=group_id,
                exchange=x_exchange or pair_def.x_exchange,
                symbol=pair_def.x_symbol, side=x_side, action=order_action,
                order_type=self.order_type, quantity=abs(allocated.final_x_quantity),
                price=px, position_id=order_position_id,
                target_hedge_ratio=target_hedge_ratio, pair_id=allocated.pair_id,
                exit_reason=exit_reason, protection_trigger=protection_trigger,
                exit_class=exit_class, reopen_lock_pending=reopen_lock_pending,
                protection_max_holding_bars=protection_max_holding_bars,
                protection_max_holding_deadline_bar=protection_max_holding_deadline_bar,
                protection_pair_loss_stop_return=protection_pair_loss_stop_return,
                protection_pair_loss_stop_freeze_bars=protection_pair_loss_stop_freeze_bars,
                protection_pair_loss_stop_freeze_until_bar=protection_pair_loss_stop_freeze_until_bar,
                protection_rule=protection_rule,
                protection_freeze_bars=protection_freeze_bars,
                protection_freeze_until_bar=protection_freeze_until_bar,
                para=order_para, rebalance_batch_id=rebalance_batch_id,
            ))

        if allocated.final_y_quantity > 0:
            py = None if self.order_type == OrderType.MARKET else (
                allocated.final_y_notional / abs(allocated.final_y_quantity)
                if abs(allocated.final_y_quantity) > 1e-12 and allocated.final_y_notional > 1e-12 else None
            )
            orders.append(Order(
                order_id=str(uuid4())[:8], group_id=group_id,
                exchange=y_exchange or pair_def.y_exchange,
                symbol=pair_def.y_symbol, side=y_side, action=order_action,
                order_type=self.order_type, quantity=abs(allocated.final_y_quantity),
                price=py, position_id=order_position_id,
                target_hedge_ratio=target_hedge_ratio, pair_id=allocated.pair_id,
                exit_reason=exit_reason, protection_trigger=protection_trigger,
                exit_class=exit_class, reopen_lock_pending=reopen_lock_pending,
                protection_max_holding_bars=protection_max_holding_bars,
                protection_max_holding_deadline_bar=protection_max_holding_deadline_bar,
                protection_pair_loss_stop_return=protection_pair_loss_stop_return,
                protection_pair_loss_stop_freeze_bars=protection_pair_loss_stop_freeze_bars,
                protection_pair_loss_stop_freeze_until_bar=protection_pair_loss_stop_freeze_until_bar,
                protection_rule=protection_rule,
                protection_freeze_bars=protection_freeze_bars,
                protection_freeze_until_bar=protection_freeze_until_bar,
                para=order_para, rebalance_batch_id=rebalance_batch_id,
            ))

        return orders
