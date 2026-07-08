from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RiskResult:
    passed: bool
    risk_name: str
    message: str
    block_open_orders: bool = False
    stage: str | None = None

    @property
    def block_new_orders(self):
        return self.block_open_orders


class BaseRisk(ABC):
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self.history = []

    def __call__(self, **kwargs):
        result = self.check(**kwargs)
        self.history.append(result)
        return result

    @abstractmethod
    def check(self, **kwargs):
        pass
