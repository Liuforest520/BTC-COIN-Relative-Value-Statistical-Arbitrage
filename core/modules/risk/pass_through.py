from core.modules.risk.base import BaseRisk, RiskResult


class PassThroughRisk(BaseRisk):
    def check(self, **kwargs):
        return RiskResult(True, self.name, "pass-through risk")
