from core.modules.models import OrderAction, OrderSide
from core.modules.risk.base import BaseRisk, RiskResult
from core.modules.risk.utils import group_by_id, value_of


class PairCompletenessRisk(BaseRisk):
    def check(self, orders=None, trades=None, **kwargs):
        items = orders if orders is not None else trades
        if not items:
            return RiskResult(True, self.name, "no items")

        for group_id, group in group_by_id(items).items():
            if group_id is None:
                return RiskResult(False, self.name, "missing group_id", True)
            actions = {value_of(getattr(item, "action", None)) for item in group if hasattr(item, "action")}
            actions.discard(None)
            if actions and actions.issubset({OrderAction.CANCEL.value}):
                continue
            if len(group) < 2:
                return RiskResult(False, self.name, f"group {group_id} has single leg", True)

            sides = {value_of(item.side) for item in group}
            if OrderSide.BUY.value not in sides or OrderSide.SELL.value not in sides:
                return RiskResult(False, self.name, f"group {group_id} must have buy and sell", True)

            if actions and not actions.issubset({OrderAction.OPEN.value, OrderAction.CLOSE.value}):
                return RiskResult(False, self.name, f"group {group_id} has invalid action", True)

        return RiskResult(True, self.name, "pair complete")
