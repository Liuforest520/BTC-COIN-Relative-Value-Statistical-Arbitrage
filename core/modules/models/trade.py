from dataclasses import dataclass


@dataclass
class Trade:
    order_id: str
    group_id: str
    exchange: str
    symbol: str
    action: str
    side: str
    quantity: float
    price: float
    notional: float
    fee: float
    slippage: float
    ts: object
    funding_fee: float = 0.0
    position_id: str | None = None
    target_hedge_ratio: float | None = None
