from dataclasses import dataclass


@dataclass
class FundingPayment:
    exchange: str
    symbol: str
    ts: int
    funding_rate: float
    quantity: float
    mark_price: float
    notional: float
    payment: float
    position_id: str | None = None
    pair_id: str | None = None
