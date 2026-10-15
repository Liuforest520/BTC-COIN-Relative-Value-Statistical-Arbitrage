from __future__ import annotations
from .base import ProtectionContext, ProtectionDecision
from .rules import PairLossStopRule

class ProtectionManager:
    """Pair loss stop protection manager."""
    def __init__(self, config):
        self.config = config
        self.rules = [PairLossStopRule()] if config.enabled and config.pair_loss_stop_enabled else []

    def evaluate(self, context: ProtectionContext) -> ProtectionDecision | None:
        for rule in self.rules:
            decision = rule.evaluate(context)
            if decision is not None:
                return decision
        return None
