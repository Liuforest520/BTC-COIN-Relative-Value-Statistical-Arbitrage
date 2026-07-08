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
