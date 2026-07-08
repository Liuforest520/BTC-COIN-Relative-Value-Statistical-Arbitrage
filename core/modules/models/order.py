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
    price: float | None = None
    timestamp: datetime | None = None
    status: OrderStatus = OrderStatus.NEW
    cancel_order_id: str | None = None
    position_id: str | None = None
    target_hedge_ratio: float | None = None
