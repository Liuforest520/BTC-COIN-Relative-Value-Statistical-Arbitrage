from core.modules.risk.base import BaseRisk, RiskResult
from core.modules.risk.utils import deviation, group_by_id, order_notional, split_long_short, trade_notional, value_of


class DollarNeutralRisk(BaseRisk):
    def __init__(self, max_deviation=0.02):
        super().__init__()
        self.max_deviation = max_deviation

    def check(self, orders=None, trades=None, bars=None, **kwargs):
        items = orders if orders is not None else trades
        if not items:
            return RiskResult(True, self.name, "no items")

        for group_id, group in group_by_id(items).items():
            actions = {value_of(getattr(item, "action", None)) for item in group}
            if "open" not in actions:
                continue

            if orders is not None:
                long_value, short_value = split_long_short(group, lambda order: order_notional(order, bars))
            else:
                long_value, short_value = split_long_short(group, trade_notional)

            if long_value is None or short_value is None:
                return RiskResult(False, self.name, f"group {group_id} missing price", True)

            dev = deviation(long_value, short_value)
            if dev > self.max_deviation:
                message = f"group {group_id} deviation {dev:.4f} > {self.max_deviation:.4f}"
                return RiskResult(False, self.name, message, True)

        return RiskResult(True, self.name, "dollar neutral")
