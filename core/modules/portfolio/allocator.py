"""Portfolio allocation strategies."""
from __future__ import annotations

from copy import deepcopy
from math import isfinite
from core.modules.models.pipeline_types import (
    AllocatedPairTarget, PortfolioAllocation, PortfolioState, RawPairTarget,
    register_portfolio, PORTFOLIO_REGISTRY,
)


@register_portfolio("pair_target_capital")
class PairTargetCapitalAllocator:
    """Allocate each Pair's configured gross target from free account margin.

    There is deliberately no concurrent-Pair count and no per-symbol cap here.
    Both legs consume gross margin; the exchange performs the final common
    proportional scale at executable prices.
    """
    def __init__(self, equity=100000.0, minimum_entry_capital_ratio=0.5,
                 allocation_mode="target_capital", min_signal_score=0.0,
                 min_pair_weight=0.0, **_kwargs):
        self.equity = float(equity)
        self.minimum_entry_capital_ratio = float(minimum_entry_capital_ratio)
        if not isfinite(self.minimum_entry_capital_ratio) or not 0.0 < self.minimum_entry_capital_ratio <= 1.0:
            raise ValueError("minimum_entry_capital_ratio must be in (0, 1]")
        self.allocation_mode = str(allocation_mode or "target_capital")
        self.min_signal_score = max(0.0, float(min_signal_score or 0.0))
        self.min_pair_weight = max(0.0, float(min_pair_weight or 0.0))

    def allocate(self, state: PortfolioState, targets: dict[str, RawPairTarget], bar_index: int):
        ready = [(pid, raw) for pid, raw in targets.items()
                 if raw.ready and _target_score(raw) >= self.min_signal_score]
        if not ready:
            return PortfolioAllocation(bar_index=bar_index, reason="no ready pairs")
        if self.allocation_mode == "score_weighted":
            ready.sort(key=lambda item: (-_target_score(item[1]), item[0]))

        # Zero available balance is a real account state, not a reason to
        # create buying power from equity.  Fall back only for legacy state
        # objects that do not expose available_balance at all.
        if hasattr(state, "available_balance"):
            available_before = float(getattr(state, "available_balance") or 0.0)
        else:
            available_before = float(getattr(state, "equity", 0.0) or 0.0)
        # A direct allocator fixture may omit account context entirely.  The
        # live strategy always supplies a ready PortfolioState with the
        # exchange-reported balance, so this fallback cannot create live cash.
        if (available_before <= 0 and not getattr(state, "ready", False)
                and float(getattr(state, "equity", 0.0) or 0.0) <= 0):
            available_before = self.equity
        available = max(0.0, available_before)
        pair_targets, selected, skipped = {}, [], []
        for pair_id, raw in ready:
            target = _raw_target_capital(raw)
            if target <= 0:
                target = max(0.0, float(raw.gross_notional or raw.x_notional + raw.y_notional))
            minimum = target * self.minimum_entry_capital_ratio
            if target <= 0 or available < minimum - 1e-9:
                skipped.append(pair_id)
                continue
            gross = min(target, available)
            if gross < minimum - 1e-9:
                skipped.append(pair_id)
                continue
            pair_targets[pair_id] = _allocated_from_gross(
                pair_id, raw, gross, self.equity,
                f"pair_target_capital target={target:.2f} allocated={gross:.2f}",
            )
            selected.append(pair_id)
            available -= gross
            if available <= 1e-9:
                break
        return PortfolioAllocation(
            bar_index=bar_index, selected_pair_ids=selected,
            pair_targets=pair_targets,
            portfolio_gross_exposure=sum(t.final_x_notional + t.final_y_notional for t in pair_targets.values()),
            portfolio_net_exposure=_portfolio_net_exposure(pair_targets),
            reason=f"pair_target_capital selected={len(selected)}/{len(ready)} skipped={len(skipped)}",
            constraint_report={"available_before": available_before,
                               "available_after_plan": available,
                               "skipped_pair_ids": skipped},
        )


# Compatibility labels retain import/config compatibility but all resolve to
# the target-capital allocator; none implements slot/count limits.
EquitySlotAllocator = PairTargetCapitalAllocator
EqualWeightAllocator = PairTargetCapitalAllocator
RiskParityAllocator = PairTargetCapitalAllocator
MinVarianceAllocator = PairTargetCapitalAllocator
ConstrainedQPAllocator = PairTargetCapitalAllocator
MaxSharpeAllocator = PairTargetCapitalAllocator
for _name in ("equity_slot", "equal_weight", "risk_parity", "min_variance",
              "constrained_qp", "max_sharpe"):
    PORTFOLIO_REGISTRY[_name] = PairTargetCapitalAllocator


def _raw_target_capital(raw):
    # For an add-on, gross_notional is the remaining amount for this entry;
    # do not allocate the Pair's full target a second time.
    gross = getattr(raw, "gross_notional", None)
    try:
        if gross is not None and isfinite(float(gross)) and float(gross) > 0:
            return max(0.0, float(gross))
    except (TypeError, ValueError):
        pass
    value = getattr(raw, "target_capital", None)
    if value is not None:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            pass
    para = raw.para if isinstance(raw.para, dict) else {}
    pair = para.get("pair", {}) if isinstance(para.get("pair"), dict) else {}
    try:
        pair_target = float(pair.get("target_capital", 0.0) or 0.0)
        if pair_target > 0:
            return pair_target
    except (TypeError, ValueError):
        pass
    # Legacy callers may provide only the per-entry cap.  This is a sizing
    # compatibility fallback, not a concurrent-Pair limit.
    portfolio = para.get("portfolio", {}) if isinstance(para.get("portfolio"), dict) else {}
    try:
        remaining = float(portfolio.get("remaining_pair_gross", 0.0) or 0.0)
        entry_cap = float(portfolio.get("entry_gross_cap", 0.0) or 0.0)
        return max(0.0, min(v for v in (remaining, entry_cap) if v > 0))
    except (TypeError, ValueError):
        return 0.0

def _portfolio_net_exposure(pair_targets: dict[str, AllocatedPairTarget]) -> float:
    return sum(_signed_net_exposure(target) for target in pair_targets.values())


def _signed_net_exposure(target: AllocatedPairTarget) -> float:
    if target.side == "long_x":
        return target.final_x_notional - target.final_y_notional
    if target.side == "short_x":
        return target.final_y_notional - target.final_x_notional
    return target.final_x_notional - target.final_y_notional


def _allocated_from_gross(
    pair_id: str,
    raw: RawPairTarget,
    gross_notional: float,
    equity: float,
    reason: str,
) -> AllocatedPairTarget:
    x_weight, y_weight = _weights_from_raw(raw)
    final_x_notional = gross_notional * x_weight
    final_y_notional = gross_notional * y_weight
    final_x_quantity = _quantity_from_notional(final_x_notional, raw.x_price)
    final_y_quantity = _quantity_from_notional(final_y_notional, raw.y_price)

    return AllocatedPairTarget(
        pair_id=pair_id,
        selected=True,
        side=raw.side,
        hedge_ratio=raw.hedge_ratio,
        scale_factor=1.0,
        final_x_notional=final_x_notional,
        final_y_notional=final_y_notional,
        final_x_quantity=final_x_quantity,
        final_y_quantity=final_y_quantity,
        portfolio_weight=gross_notional / equity if equity else 0.0,
        allocation_reason=reason,
        para=deepcopy(raw.para),
    )


def _weights_from_raw(raw: RawPairTarget) -> tuple[float, float]:
    x_weight = _float_or_default(getattr(raw, "x_weight", 0.0), 0.0)
    y_weight = _float_or_default(getattr(raw, "y_weight", 0.0), 0.0)
    total = x_weight + y_weight
    if total > 1e-12:
        return x_weight / total, y_weight / total

    raw_gross = raw.gross_notional or (raw.x_notional + raw.y_notional)
    if raw_gross > 1e-12:
        return raw.x_notional / raw_gross, raw.y_notional / raw_gross

    return 0.5, 0.5


def _quantity_from_notional(notional: float, price: float) -> float:
    price = _float_or_default(price, 0.0)
    if price <= 0:
        return 0.0
    return notional / price


def _add_symbol_exposure(
    raw: RawPairTarget,
    target: AllocatedPairTarget,
    symbol_exposure: dict[str, float],
) -> None:
    """Track per-symbol notional for diagnostics only; never cap it."""
    x_symbol = _raw_symbol(raw, "x_symbol")
    y_symbol = _raw_symbol(raw, "y_symbol")
    if x_symbol:
        symbol_exposure[x_symbol] = symbol_exposure.get(x_symbol, 0.0) + target.final_x_notional
    if y_symbol:
        symbol_exposure[y_symbol] = symbol_exposure.get(y_symbol, 0.0) + target.final_y_notional


def _raw_symbol(raw: RawPairTarget, key: str) -> str | None:
    para = raw.para if isinstance(raw.para, dict) else {}
    pair_para = para.get("pair", {}) if isinstance(para.get("pair"), dict) else {}
    value = pair_para.get(key)
    return str(value) if value else None


def _target_score(raw: RawPairTarget) -> float:
    score = _float_or_default(getattr(raw, "signal_strength", 0.0), 0.0)
    if score > 0:
        return score

    para = raw.para if isinstance(raw.para, dict) else {}
    signal_para = para.get("signal", {}) if isinstance(para.get("signal"), dict) else {}
    checks = (
        signal_para.get("entry_quality_checks", {})
        if isinstance(signal_para.get("entry_quality_checks"), dict)
        else {}
    )
    return max(_float_or_default(checks.get("score"), 0.0), 0.0)


def _float_or_default(value, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed
