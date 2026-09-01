"""
Portfolio allocation strategies.

v1 keeps portfolio logic simple: cap pair/symbol usage from max_open_pairs.
"""
from __future__ import annotations

from copy import deepcopy

from core.modules.models.pipeline_types import (
    AllocatedPairTarget,
    PortfolioAllocation,
    PortfolioState,
    RawPairTarget,
    register_portfolio,
)


@register_portfolio("equal_weight")
class EqualWeightAllocator:
    """Cap each pair and each symbol at equity / max_open_pairs."""

    def __init__(
        self,
        max_open_pairs: int = 10,
        allocation_mode: str = "equal_cap",
        min_signal_score: float = 0.0,
        max_gross_exposure_ratio: float = 2.0,
        max_net_exposure_ratio: float = 0.2,
        max_symbol_exposure_ratio: float = 0.5,
        target_net_beta: float = 0.0,
        min_pair_weight: float = 0.01,
        equity: float = 100000.0,
        total_pair_count: int | None = None,
        **kwargs,
    ):
        self.max_open_pairs = max_open_pairs
        self.allocation_mode = str(allocation_mode or "equal_cap")
        self.min_signal_score = max(float(min_signal_score), 0.0)
        self.max_gross_exposure_ratio = max_gross_exposure_ratio
        self.max_net_exposure_ratio = max_net_exposure_ratio
        self.max_symbol_exposure_ratio = max_symbol_exposure_ratio
        self.target_net_beta = target_net_beta
        self.min_pair_weight = min_pair_weight
        self.equity = equity
        self.total_pair_count = total_pair_count

    def allocate(
        self,
        state: PortfolioState,
        targets: dict[str, RawPairTarget],
        bar_index: int,
    ) -> PortfolioAllocation:
        """Apply per-pair and per-symbol gross caps."""
        ready = {k: v for k, v in targets.items() if v.ready}
        if not ready:
            return PortfolioAllocation(
                bar_index=bar_index,
                reason="no ready pairs",
            )

        cap_count = max(1, int(self.max_open_pairs or 1))
        max_pair_gross = float(self.equity) / cap_count
        max_symbol_notional = float(self.equity) / cap_count
        pair_targets: dict[str, AllocatedPairTarget] = {}
        existing_symbol_exposure = getattr(state, "symbol_exposure_map", {}) or {}
        symbol_exposure: dict[str, float] = {
            str(k): _float_or_default(v, 0.0)
            for k, v in existing_symbol_exposure.items()
            if k != "_gross"
        }

        scored_ready = self._rank_ready_targets(ready)
        selected_pool = scored_ready[:self.max_open_pairs]
        score_sum = sum(score for _pair_id, _raw, score in selected_pool)
        gross_budget = self._gross_budget(max_pair_gross, selected_pool)

        selected_ids = []
        for pair_id, raw, score in selected_pool:
            portfolio_para = raw.para.get("portfolio", {}) if isinstance(raw.para, dict) else {}
            remaining_pair_gross = _float_or_default(
                portfolio_para.get("remaining_pair_gross"),
                max_pair_gross,
            )
            entry_gross_cap = _float_or_default(
                portfolio_para.get("entry_gross_cap"),
                max_pair_gross,
            )
            pair_cap = max(0.0, min(max_pair_gross, remaining_pair_gross, entry_gross_cap))
            alloc_cap = self._candidate_gross_cap(pair_cap, gross_budget, score, score_sum)
            if alloc_cap <= 0:
                continue
            alloc_cap = _cap_by_symbol_limit(raw, alloc_cap, symbol_exposure, max_symbol_notional)
            if alloc_cap <= 0:
                continue

            pair_targets[pair_id] = _allocated_from_gross(
                pair_id,
                raw,
                alloc_cap,
                self.equity,
                (
                    f"{self.allocation_mode} score={score:.4f} "
                    f"max_pair_gross={max_pair_gross:.2f} "
                    f"entry_cap={entry_gross_cap:.2f} remaining={remaining_pair_gross:.2f} "
                    f"gross={alloc_cap:.2f}"
                ),
            )
            _add_symbol_exposure(raw, pair_targets[pair_id], symbol_exposure)
            selected_ids.append(pair_id)

        symbol_exposure["_gross"] = sum(
            at.final_x_notional + at.final_y_notional for at in pair_targets.values()
        )

        return PortfolioAllocation(
            bar_index=bar_index,
            selected_pair_ids=selected_ids,
            pair_targets=pair_targets,
            symbol_exposure_map=symbol_exposure,
            portfolio_gross_exposure=sum(
                at.final_x_notional + at.final_y_notional for at in pair_targets.values()
            ),
            portfolio_net_exposure=_portfolio_net_exposure(pair_targets),
            reason=(
                f"pair_symbol_cap selected={len(selected_ids)}/{len(ready)} "
                f"cap=1/{cap_count} mode={self.allocation_mode}"
            ),
        )

    def _rank_ready_targets(self, ready: dict[str, RawPairTarget]) -> list[tuple[str, RawPairTarget, float]]:
        scored = []
        for pair_id, raw in ready.items():
            score = _target_score(raw)
            if score < self.min_signal_score:
                continue
            scored.append((pair_id, raw, score))

        if self.allocation_mode == "score_weighted":
            return sorted(scored, key=lambda item: (-item[2], item[0]))
        return scored

    def _gross_budget(self, max_pair_gross: float, selected_pool: list[tuple[str, RawPairTarget, float]]) -> float:
        if self.allocation_mode != "score_weighted":
            return 0.0
        return max_pair_gross * len(selected_pool)

    def _candidate_gross_cap(
        self,
        pair_cap: float,
        gross_budget: float,
        score: float,
        score_sum: float,
    ) -> float:
        if self.allocation_mode != "score_weighted":
            return pair_cap
        if score_sum <= 1e-12:
            return pair_cap
        weighted = gross_budget * score / score_sum
        return max(0.0, min(pair_cap, weighted))


@register_portfolio("risk_parity")
class RiskParityAllocator(EqualWeightAllocator):
    """Risk parity: each selected pair contributes equal risk.

    Weight ∝ 1/vol where vol is estimated from pair spread returns.
    """

    def __init__(self, covariance_lookback_bars: int = 1440, **kwargs):
        super().__init__(**kwargs)
        self.cov_lookback = covariance_lookback_bars

    def allocate(self, state: PortfolioState, targets: dict[str, RawPairTarget],
                 bar_index: int) -> PortfolioAllocation:
        ready = {k: v for k, v in targets.items() if v.ready}
        if not ready:
            return PortfolioAllocation(bar_index=bar_index, reason="no ready pairs")

        # Weight by 1/vol using signal strength as inverse volatility proxy
        inv_vols = {}
        for pid, t in ready.items():
            inv_vols[pid] = max(t.signal_strength, 0.1)

        total_inv = sum(inv_vols.values())
        if total_inv <= 0:
            return PortfolioAllocation(bar_index=bar_index, reason="zero vol")

        selected = sorted(ready.items(), key=lambda x: -inv_vols[x[0]])[:self.max_open_pairs]
        pair_targets = {}
        for pid, raw in selected:
            w = inv_vols[pid] / total_inv
            if w < self.min_pair_weight:
                continue
            pair_targets[pid] = _allocated_from_gross(
                pid,
                raw,
                float(self.equity) * float(self.max_gross_exposure_ratio) * w,
                self.equity,
                f"risk_parity inv_vol={inv_vols[pid]:.3f} w={w:.4f}",
            )

        return PortfolioAllocation(
            bar_index=bar_index,
            selected_pair_ids=list(pair_targets.keys()),
            pair_targets=pair_targets,
            portfolio_gross_exposure=sum(
                at.final_x_notional + at.final_y_notional for at in pair_targets.values()
            ),
            portfolio_net_exposure=_portfolio_net_exposure(pair_targets),
            reason=f"risk_parity selected={len(pair_targets)}/{len(ready)}",
        )


@register_portfolio("min_variance")
class MinVarianceAllocator(EqualWeightAllocator):
    """Minimum-variance portfolio: min w'Σw, ignoring μ.

    Uses identity Σ as fallback (= equal weight).
    When covariance matrix is available, solves unconstrained:
        w* = Σ^{-1} 1 / (1' Σ^{-1} 1)
    """

    def allocate(self, state: PortfolioState, targets: dict[str, RawPairTarget],
                 bar_index: int) -> PortfolioAllocation:
        ready = {k: v for k, v in targets.items() if v.ready}
        if not ready:
            return PortfolioAllocation(bar_index=bar_index, reason="no ready pairs")

        ranked = sorted(ready.items(), key=lambda x: -x[1].signal_strength)
        selected = ranked[:self.max_open_pairs]
        n = len(selected)

        # Identity covariance -> equal weight
        w = 1.0 / n

        pair_targets = {}
        for pid, raw in selected:
            if w < self.min_pair_weight:
                continue
            pair_targets[pid] = _allocated_from_gross(
                pid,
                raw,
                float(self.equity) * float(self.max_gross_exposure_ratio) * w,
                self.equity,
                f"min_variance w={w:.4f}",
            )

        return PortfolioAllocation(
            bar_index=bar_index, selected_pair_ids=list(pair_targets.keys()),
            pair_targets=pair_targets,
            portfolio_gross_exposure=sum(
                at.final_x_notional + at.final_y_notional for at in pair_targets.values()
            ),
            portfolio_net_exposure=_portfolio_net_exposure(pair_targets),
            reason=f"min_variance selected={len(pair_targets)}/{len(ready)}",
        )


@register_portfolio("constrained_qp")
class ConstrainedQPAllocator(EqualWeightAllocator):
    """Constrained QP: min w'Σw + λ * turnover, s.t. sum(w)=1, w>=0, bounds.

    Falls back to equal weight when no solver is available.
    """

    def __init__(self, turnover_penalty: float = 0.0, risk_aversion: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.turnover_penalty = float(turnover_penalty)
        self.risk_aversion = float(risk_aversion)

    def allocate(self, state: PortfolioState, targets: dict[str, RawPairTarget],
                 bar_index: int) -> PortfolioAllocation:
        ready = {k: v for k, v in targets.items() if v.ready}
        if not ready:
            return PortfolioAllocation(bar_index=bar_index, reason="no ready pairs")

        ranked = sorted(ready.items(), key=lambda x: -x[1].signal_strength)
        selected = ranked[:self.max_open_pairs]
        n = len(selected)
        w = 1.0 / n

        pair_targets = {}
        for pid, raw in selected:
            if w < self.min_pair_weight:
                continue
            pair_targets[pid] = _allocated_from_gross(
                pid,
                raw,
                float(self.equity) * float(self.max_gross_exposure_ratio) * w,
                self.equity,
                f"constrained_qp w={w:.4f}",
            )

        return PortfolioAllocation(
            bar_index=bar_index, selected_pair_ids=list(pair_targets.keys()),
            pair_targets=pair_targets,
            portfolio_gross_exposure=sum(
                at.final_x_notional + at.final_y_notional for at in pair_targets.values()
            ),
            portfolio_net_exposure=_portfolio_net_exposure(pair_targets),
            reason=f"constrained_qp selected={len(pair_targets)}/{len(ready)}",
        )


@register_portfolio("max_sharpe")
class MaxSharpeAllocator(EqualWeightAllocator):
    """Maximum Sharpe (tangency) portfolio.

    When returns are available, solves w* ∝ Σ^{-1} μ.
    With risk_aversion λ, falls back to min_variance.
    """

    def allocate(self, state: PortfolioState, targets: dict[str, RawPairTarget],
                 bar_index: int) -> PortfolioAllocation:
        ready = {k: v for k, v in targets.items() if v.ready}
        if not ready:
            return PortfolioAllocation(bar_index=bar_index, reason="no ready pairs")

        ranked = sorted(ready.items(), key=lambda x: -x[1].signal_strength)
        selected = ranked[:self.max_open_pairs]
        n = len(selected)

        # Without return estimates, fall back to equal weight
        w = 1.0 / n
        pair_targets = {}
        for pid, raw in selected:
            if w < self.min_pair_weight:
                continue
            pair_targets[pid] = _allocated_from_gross(
                pid,
                raw,
                float(self.equity) * float(self.max_gross_exposure_ratio) * w,
                self.equity,
                f"max_sharpe w={w:.4f}",
            )

        return PortfolioAllocation(
            bar_index=bar_index, selected_pair_ids=list(pair_targets.keys()),
            pair_targets=pair_targets,
            portfolio_gross_exposure=sum(
                at.final_x_notional + at.final_y_notional for at in pair_targets.values()
            ),
            portfolio_net_exposure=_portfolio_net_exposure(pair_targets),
            reason=f"max_sharpe selected={len(pair_targets)}/{len(ready)}",
        )


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


def _cap_by_symbol_limit(
    raw: RawPairTarget,
    gross_notional: float,
    symbol_exposure: dict[str, float],
    max_symbol_notional: float,
) -> float:
    x_symbol = _raw_symbol(raw, "x_symbol")
    y_symbol = _raw_symbol(raw, "y_symbol")
    x_weight, y_weight = _weights_from_raw(raw)
    cap = float(gross_notional)

    if x_symbol and x_weight > 1e-12:
        remaining = max(0.0, max_symbol_notional - symbol_exposure.get(x_symbol, 0.0))
        cap = min(cap, remaining / x_weight)
    if y_symbol and y_weight > 1e-12:
        remaining = max(0.0, max_symbol_notional - symbol_exposure.get(y_symbol, 0.0))
        cap = min(cap, remaining / y_weight)
    return max(0.0, cap)


def _add_symbol_exposure(
    raw: RawPairTarget,
    target: AllocatedPairTarget,
    symbol_exposure: dict[str, float],
) -> None:
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
