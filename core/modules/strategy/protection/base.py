from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class ProtectionDecision:
    triggered: bool
    rule_name: str = ""
    exit_reason: str = ""
    exit_class: str = "stop_loss"
    protection_trigger: str = ""
    freeze_bars: int = 0
    wait_for_model_update: bool = False
    diagnostic: dict[str, Any] = field(default_factory=dict)

@dataclass
class ProtectionContext:
    pipeline: Any
    bundle: Any
    state: Any
    sizing_state: Any
    config: Any
    fee_rate: float
    slippage_rate: float

class ProtectionRule:
    name = ""
    priority = 0
    def evaluate(self, context: ProtectionContext) -> ProtectionDecision | None:
        raise NotImplementedError
