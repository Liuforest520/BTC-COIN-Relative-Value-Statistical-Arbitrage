from core.modules.risk.base import BaseRisk, RiskResult
from core.modules.risk.utils import group_by_id, order_notional, split_long_short, value_of


class HedgeRatioRisk(BaseRisk):
    def __init__(self, hedge_method="dollar_neutral", max_deviation=0.02):
        super().__init__()
        self.hedge_method = hedge_method
        self.max_deviation = max_deviation

    def check(self, orders=None, bars=None, **kwargs):
        if not orders:
            return RiskResult(True, self.name, "no orders")

        for group_id, group in group_by_id(orders).items():
            actions = {value_of(getattr(item, "action", None)) for item in group}
            if "open" not in actions:
                continue

            long_value, short_value = split_long_short(group, lambda order: order_notional(order, bars))
            if long_value is None or short_value is None:
                return RiskResult(False, self.name, f"group {group_id} missing price", True)
            if long_value <= 0 or short_value <= 0:
                return RiskResult(False, self.name, f"group {group_id} missing long or short notional", True)

            target_ratio = self._target_ratio(group)
            if target_ratio is None:
                return RiskResult(False, self.name, f"group {group_id} missing target hedge ratio", True)

            actual_ratio = long_value / short_value
            deviation = abs(actual_ratio - target_ratio) / target_ratio if target_ratio else 0.0
            if deviation > self.max_deviation:
                message = (
                    f"group {group_id} hedge ratio deviation {deviation:.4f} > {self.max_deviation:.4f} "
                    f"actual={actual_ratio:.4f} target={target_ratio:.4f}"
                )
                return RiskResult(False, self.name, message, True)

        return RiskResult(True, self.name, f"{self.hedge_method} hedge ratio")

    def _target_ratio(self, group):
        if self.hedge_method == "dollar_neutral":
            return 1.0

        ratios = [
            float(getattr(order, "target_hedge_ratio"))
            for order in group
            if getattr(order, "target_hedge_ratio", None) is not None
        ]
        if not ratios:
            return None

        target_ratio = ratios[0]
        if target_ratio <= 0:
            return None
        return target_ratio
