from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RebalanceCandidate:
    pair_id: str
    target_capital: float
    minimum_capital: float
    adf_pvalue: float | None
    theoretical_zero_return_x_move: float | None
    expected_net_return: float | None
    signal_strength: float
    payload: Any = None


@dataclass(frozen=True)
class RebalancePosition:
    pair_id: str
    net_return: float
    releasable_equity: float
    held_bars: int
    minimum_holding_bars: int
    payload: Any = None


@dataclass(frozen=True)
class RebalancePlan:
    candidate: RebalanceCandidate | None = None
    evictions: tuple[RebalancePosition, ...] = field(default_factory=tuple)
    available_capital: float = 0.0
    release_needed: float = 0.0
    planned_release: float = 0.0
    reason: str = ""

    @property
    def ready(self) -> bool:
        return self.candidate is not None and bool(self.evictions)
