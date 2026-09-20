from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderAction(str, Enum):
    OPEN = "open"
    CLOSE = "close"
    CANCEL = "cancel"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    NEW = "new"
    CANCELED = "canceled"
    FILLED = "filled"
    REJECTED = "rejected"


@dataclass
class Order:
    order_id: str
    group_id: str
    exchange: str
    symbol: str
    action: OrderAction
    side: OrderSide | None = None
    order_type: OrderType | None = None
    quantity: float | None = None
    # ``requested_quantity`` keeps the pre-margin amount when the exchange
    # proportionally reduces an entire Pair order group at the fill bar.
    requested_quantity: float | None = None
    fill_scale: float = 1.0
    price: float | None = None
    timestamp: datetime | None = None
    status: OrderStatus = OrderStatus.NEW
    cancel_order_id: str | None = None
    position_id: str | None = None
    target_hedge_ratio: float | None = None
    pair_id: str | None = None
    exit_reason: str | None = None
    protection_trigger: str | None = None
    exit_class: str | None = None
    reopen_lock_pending: bool | None = None
    protection_max_holding_bars: int | None = None
    protection_max_holding_deadline_bar: int | None = None
    protection_pair_loss_stop_return: float | None = None
    protection_pair_loss_stop_freeze_bars: int | None = None
    protection_pair_loss_stop_freeze_until_bar: int | None = None
    protection_rule: str | None = None
    protection_freeze_bars: int | None = None
    protection_freeze_until_bar: int | None = None
    para: dict | None = None
    rebalance_batch_id: str | None = None
