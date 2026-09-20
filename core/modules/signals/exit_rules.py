"""Shared position exit rules for price-based z-score strategies."""
from __future__ import annotations

from math import isfinite


def resolve_position_exit_decision(
    para: dict | None,
    zscore: float | None,
    exit_z: float,
) -> tuple[bool, str | None]:
    """Resolve a signed exit threshold against the active position.

    A positive ``exit_z`` closes inside the symmetric band and also protects
    against a jump across zero. Zero requires a sign reversal. A negative
    value requires the statistic to reach ``abs(exit_z)`` on the opposite
    side of zero.
    """
    position = ((para or {}).get("position") or {})
    if zscore is None or not isfinite(float(zscore)):
        return False, None

    value = float(zscore)
    configured_exit = float(exit_z)
    has_position = bool(position.get("has_position", False))

    if configured_exit > 0.0 and abs(value) <= configured_exit:
        return True, "within_exit_band"

    if not has_position:
        return False, None

    side = position.get("side")
    if configured_exit >= 0.0:
        if side == "long_x" and value <= 0.0:
            return True, "direction_reversed"
        if side == "short_x" and value >= 0.0:
            return True, "direction_reversed"
        return False, None

    reversal_threshold = abs(configured_exit)
    if side == "long_x" and value <= -reversal_threshold:
        return True, "reversal_threshold_reached"
    if side == "short_x" and value >= reversal_threshold:
        return True, "reversal_threshold_reached"
    return False, None
