from __future__ import annotations

from math import isfinite

from .base import RebalanceCandidate, RebalancePlan, RebalancePosition


class RebalanceManager:
    """Choose at most one underfunded candidate and its complete evictions.

    Normal, directly affordable entries never pass through this manager. It
    only considers candidates rejected by the ordinary allocator for lack of
    minimum capital. Candidate gates are always applied before an existing
    losing position may be sacrificed.
    """

    def __init__(self, config):
        self.config = config

    def plan(
        self,
        candidates: list[RebalanceCandidate],
        positions: list[RebalancePosition],
        available_capital: float,
    ) -> RebalancePlan:
        available = max(0.0, float(available_capital or 0.0))
        qualified = [candidate for candidate in candidates if self.quality_allowed(candidate)]
        if not qualified:
            return RebalancePlan(available_capital=available, reason="no qualified underfunded candidate")

        candidate = min(qualified, key=self._candidate_sort_key)
        release_needed = max(0.0, candidate.minimum_capital - available)
        if release_needed <= 1e-9:
            return RebalancePlan(
                candidate=candidate,
                available_capital=available,
                reason="candidate is directly affordable",
            )

        eligible = sorted(
            (
                position
                for position in positions
                if position.net_return < 0.0
                and position.held_bars >= position.minimum_holding_bars
                and position.releasable_equity > 1e-9
            ),
            key=lambda position: (position.net_return, position.pair_id),
        )
        if not eligible:
            return RebalancePlan(
                candidate=candidate,
                available_capital=available,
                release_needed=release_needed,
                reason="no eligible losing position",
            )

        selected = []
        release = 0.0
        for position in eligible:
            selected.append(position)
            release += position.releasable_equity
            if release >= release_needed - 1e-9:
                break

        if release < release_needed - 1e-9:
            return RebalancePlan(
                candidate=candidate,
                available_capital=available,
                release_needed=release_needed,
                planned_release=release,
                reason="eligible losing positions cannot fund candidate minimum",
            )
        return RebalancePlan(
            candidate=candidate,
            evictions=tuple(selected),
            available_capital=available,
            release_needed=release_needed,
            planned_release=release,
            reason="replacement ready",
        )

    def quality_allowed(self, candidate: RebalanceCandidate) -> bool:
        quality = self.config.candidate_quality
        if not quality.enabled:
            return True
        if not self._finite_at_most(candidate.adf_pvalue, quality.max_adf_pvalue):
            return False
        if not self._finite_at_least(
            candidate.theoretical_zero_return_x_move,
            quality.min_theoretical_zero_return_x_move,
        ):
            return False
        if not self._finite_at_least(
            candidate.expected_net_return,
            quality.min_expected_net_return,
        ):
            return False
        return True

    @staticmethod
    def _candidate_sort_key(candidate: RebalanceCandidate):
        """Best means highest fee-adjusted return, then safest/stablest."""
        expected_return = RebalanceManager._finite_or(
            candidate.expected_net_return, float("-inf")
        )
        zero_move = RebalanceManager._finite_or(
            candidate.theoretical_zero_return_x_move, float("-inf")
        )
        adf = RebalanceManager._finite_or(candidate.adf_pvalue, float("inf"))
        signal_strength = RebalanceManager._finite_or(candidate.signal_strength, 0.0)
        return (
            -expected_return,
            -zero_move,
            adf,
            -signal_strength,
            candidate.pair_id,
        )

    @staticmethod
    def _finite_or(value, default) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        return parsed if isfinite(parsed) else float(default)

    @staticmethod
    def _finite_at_least(value, threshold) -> bool:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return False
        return isfinite(parsed) and parsed >= float(threshold)

    @staticmethod
    def _finite_at_most(value, threshold) -> bool:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return False
        return isfinite(parsed) and parsed <= float(threshold)
