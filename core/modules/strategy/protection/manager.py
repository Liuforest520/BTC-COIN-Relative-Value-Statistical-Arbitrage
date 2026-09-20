from __future__ import annotations
from .base import ProtectionContext, ProtectionDecision
from .rules import TheoreticalXStopRule, PairLossStopRule, TakeProfitRule, MaxHoldingTimeRule

class ProtectionManager:
    def __init__(self, config):
        self.config = config
        self.rules = []
        if config.enabled and config.stop_loss_enabled: self.rules.append(TheoreticalXStopRule())
        if config.enabled and config.pair_loss_stop_enabled: self.rules.append(PairLossStopRule())
        if config.enabled and config.take_profit_enabled: self.rules.append(TakeProfitRule())
        if config.max_holding_time_enabled: self.rules.append(MaxHoldingTimeRule())
        self.rules.sort(key=lambda r: r.priority)

    def evaluate(self, context: ProtectionContext) -> ProtectionDecision | None:
        for rule in self.rules:
            decision = rule.evaluate(context)
            if decision is not None:
                return decision
        return None
