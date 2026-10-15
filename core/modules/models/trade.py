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
    requested_quantity: float | None = None
    fill_scale: float = 1.0
    funding_fee: float = 0.0
    position_id: str | None = None
    target_hedge_ratio: float | None = None
    pair_id: str | None = None
    exit_reason: str | None = None
    protection_trigger: str | None = None
    exit_class: str | None = None
    reopen_lock_pending: bool | None = None
    para: dict | None = None
    rebalance_batch_id: str | None = None
    protection_pair_loss_stop_return: float | None = None
    protection_pair_loss_stop_freeze_bars: int | None = None
    protection_pair_loss_stop_freeze_until_bar: int | None = None
    protection_rule: str | None = None
    protection_freeze_bars: int | None = None
    protection_freeze_until_bar: int | None = None
