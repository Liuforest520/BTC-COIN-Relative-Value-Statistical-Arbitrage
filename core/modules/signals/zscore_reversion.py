"""
Z-score reversion signal: enter only after z-score starts reverting toward mean.

Two-stage gating: arm at trigger_z, fire when z-score crosses back into entry band.
Reversion filter: optionally require z-score MA to confirm reversion direction.
"""
from __future__ import annotations

import numpy as np

from core.modules.models.pipeline_types import (
    EstimatorOutput,
    SignalOutput,
    SignalState,
    register_signal,
)
from core.modules.data.rolling_window import ensure_deque, percentiles_from_tail, tail_values
from core.modules.signals.position_exit_zscore import (
    resolve_position_exit_decision,
    resolve_position_exit_zscore,
)


@register_signal("zscore_reversion")
class ZScoreReversionSignal:
    """Signal that waits for z-score reversion before entering.

    Compared to plain ZScoreSignal, this adds:
      - Two-stage: arm when crossing trigger_z; fire only when reverting back
        across entry_z toward zero.
      - Reversion filter: short MA must move in the expected direction versus
        long MA.
    """

    def __init__(
        self,
        pair_id: str = "",
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        entry_rule_method: str = "fixed_z",
        entry_rule_lookback_bars: int = 10080,
        entry_rule_update_interval_bars: int = 60,
        entry_rule_upper_percentile: float = 97.5,
        entry_rule_lower_percentile: float = 2.5,
        entry_rule_min_samples: int | None = None,
        entry_rule_min_abs_entry_z: float | None = None,
        entry_rule_max_abs_entry_z: float | None = None,
        # Two-stage
        two_stage_enabled: bool = False,
        two_stage_trigger_z: float | None = None,
        two_stage_entry_z: float | None = None,
        two_stage_entry_window_z: float | None = None,
        two_stage_levels: list | None = None,
        # Reversion filter
        reversion_filter_enabled: bool = True,
        reversion_ma_lookback_bars: int = 60,
        reversion_min_samples: int | None = None,
        # Two-stage expiry
        two_stage_max_wait_bars: int = 0,
        pair_quality_filter_enabled: bool = False,
        pair_quality_min_samples: int = 30,
        pair_quality_min_corr: float = 0.0,
        cost_filter_enabled: bool = False,
        cost_fee_rate: float = 0.0005,
        cost_slippage_bps: float = 1.0,
        cost_buffer_multiplier: float = 1.0,
        **kwargs,
    ):
        self.pair_id = pair_id
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.entry_rule_method = entry_rule_method
        self.entry_rule_lookback_bars = entry_rule_lookback_bars
        self.entry_rule_update_interval_bars = entry_rule_update_interval_bars
        self.entry_rule_upper_percentile = entry_rule_upper_percentile
        self.entry_rule_lower_percentile = entry_rule_lower_percentile
        self.entry_rule_min_samples = entry_rule_min_samples or entry_rule_lookback_bars
        self.entry_rule_min_abs_entry_z = entry_rule_min_abs_entry_z
        self.entry_rule_max_abs_entry_z = entry_rule_max_abs_entry_z

        self.two_stage_enabled = bool(two_stage_enabled)
        self.two_stage_trigger_z = two_stage_trigger_z
        self.two_stage_entry_z = two_stage_entry_z
        self.two_stage_entry_window_z = (
            None if two_stage_entry_window_z is None else abs(float(two_stage_entry_window_z))
        )
        self.two_stage_levels = two_stage_levels

        self.reversion_filter_enabled = bool(reversion_filter_enabled)
        self.reversion_ma_lookback_bars = int(reversion_ma_lookback_bars)
        self.reversion_min_samples = int(reversion_min_samples or reversion_ma_lookback_bars)
        self.two_stage_max_wait_bars = int(two_stage_max_wait_bars)
        self.pair_quality_filter_enabled = bool(pair_quality_filter_enabled)
        self.pair_quality_min_samples = max(1, int(pair_quality_min_samples))
        self.pair_quality_min_corr = abs(float(pair_quality_min_corr))
        self.cost_filter_enabled = bool(cost_filter_enabled)
        self.cost_fee_rate = max(0.0, float(cost_fee_rate))
        self.cost_slippage_bps = max(0.0, float(cost_slippage_bps))
        self.cost_buffer_multiplier = max(0.0, float(cost_buffer_multiplier))
        self.max_history = max(
            int(self.entry_rule_lookback_bars),
            int(self.entry_rule_min_samples),
            int(self.reversion_ma_lookback_bars),
            1,
        )

    def evaluate(self, state: SignalState, estimator: EstimatorOutput,
                 bar_index: int, para: dict | None = None) -> SignalOutput:
        if not estimator.ready or estimator.spread_std is None:
            return SignalOutput(pair_id=self.pair_id, bar_index=bar_index,
                               ready=False, reason="estimator not ready",
                               para={"signal": {"method": "zscore_reversion", "blocked_by": "estimator"}})

        # Compute zscore
        std = estimator.spread_std
        zscore = 0.0
        if std > 0 and estimator.spread_mean is not None and estimator.latest_spread is not None:
            zscore = (estimator.latest_spread - estimator.spread_mean) / std

        state.zscore_history = ensure_deque(state.zscore_history, self.max_history)
        state.zscore_history.append(zscore)

        # Update entry thresholds
        should_update = (
            state.last_entry_rule_update_index is None
            or bar_index - state.last_entry_rule_update_index >= self.entry_rule_update_interval_bars
        )
        if should_update:
            self._update_thresholds(state)
            state.last_entry_rule_update_index = bar_index

        state.entry_thresholds_ready = (state.entry_z_upper is not None)

        # --- Decision ---
        side = self._candidate_side(zscore, state)
        action = "none"
        reason = f"zscore={zscore:.4f}"
        close_zscore, close_zscore_method, _ = resolve_position_exit_zscore(para, zscore)

        # Close inside the exit band or after crossing to the opposite Z side.
        should_close, close_reason = resolve_position_exit_decision(
            para, close_zscore, self.exit_z
        )
        if should_close:
            action = "close"
            if close_reason == "direction_reversed":
                position_side = ((para or {}).get("position") or {}).get("side")
                reason = (
                    f"{close_zscore_method} zscore direction reversed for "
                    f"{position_side}: {close_zscore:.4f}"
                )
            else:
                reason = (
                    f"{close_zscore_method} zscore within exit: "
                    f"{close_zscore:.4f}"
                )
            self._clear_two_stage(state)
            return SignalOutput(
                pair_id=self.pair_id, bar_index=bar_index, ready=True,
                action=action, side=None, zscore=close_zscore,
                entry_z_upper=state.entry_z_upper, entry_z_lower=state.entry_z_lower,
                exit_z=self.exit_z, signal_strength=abs(close_zscore),
                entry_thresholds_ready=state.entry_thresholds_ready, reason=reason,
                para={"signal": {
                    "method": "zscore_reversion",
                    "action": action,
                    "zscore": close_zscore,
                    "standard_zscore": zscore,
                    "position_exit_zscore": close_zscore,
                    "position_exit_zscore_method": close_zscore_method,
                    "reason": reason,
                }},
            )

        # Two-stage fire: check BEFORE side=None return (z may have reverted into band)
        if self.two_stage_enabled and state.armed_side is not None:
            return self._two_stage_fire_check(
                state, zscore, bar_index, estimator=estimator, para=para
            )

        if side is None:
            return SignalOutput(
                pair_id=self.pair_id, bar_index=bar_index, ready=True,
                action="none", side=None, zscore=zscore,
                entry_z_upper=state.entry_z_upper, entry_z_lower=state.entry_z_lower,
                exit_z=self.exit_z, signal_strength=abs(zscore),
                entry_thresholds_ready=state.entry_thresholds_ready, reason=reason,
                para={"signal": {"method": "zscore_reversion", "action": "none", "zscore": zscore, "reason": reason}},
            )

        # Two-stage arm or direct reversion
        if self.two_stage_enabled:
            return self._two_stage_arm_check(state, zscore, side, bar_index)
        return self._direct_reversion_evaluate(state, estimator, zscore, side, bar_index, para)

    # ---- Core logic ----

    def _candidate_side(self, zscore: float, state: SignalState) -> str | None:
        if zscore >= state.entry_z_upper:
            return "long_x"
        if zscore <= state.entry_z_lower:
            return "short_x"
        return None

    def _direct_reversion_evaluate(self, state, estimator, zscore, side, bar_index, para):
        """Check reversion filter, then fire."""
        checks = self._reversion_checks(state, zscore, side, bar_index)
        if not checks["passed"]:
            return SignalOutput(
                pair_id=self.pair_id, bar_index=bar_index, ready=True,
                action="none", side=side, zscore=zscore,
                entry_z_upper=state.entry_z_upper, entry_z_lower=state.entry_z_lower,
                exit_z=self.exit_z, signal_strength=abs(zscore),
                entry_thresholds_ready=state.entry_thresholds_ready,
                reason=checks["reason"],
                para={"signal": {"method": "zscore_reversion", "reversion_checks": checks}},
            )
        entry_checks = self._entry_checks(estimator, zscore, para)
        if not entry_checks["passed"]:
            return self._blocked_entry_output(
                state, zscore, side, bar_index, entry_checks, checks
            )
        state.reversion_state = {"enabled": True, "reason": "ok"}
        return SignalOutput(
            pair_id=self.pair_id, bar_index=bar_index, ready=True,
            action="open", side=side, zscore=zscore,
            entry_z_upper=state.entry_z_upper, entry_z_lower=state.entry_z_lower,
            exit_z=self.exit_z, signal_strength=abs(zscore),
            entry_thresholds_ready=state.entry_thresholds_ready,
            reason=f"reversion open {side} z={zscore:.4f}",
            para={
                "signal": {
                    "method": "zscore_reversion",
                    "reversion_checks": checks,
                }
            },
        )

    def _two_stage_arm_check(self, state, zscore, side, bar_index):
        """Check if zscore crossed the trigger to arm the two-stage mechanism."""
        trigger = self._effective_trigger_z(side)
        entry_z = self._effective_entry_z(side, state)
        if trigger is None:
            # No trigger configured, arm at entry_z immediately
            if side is not None and state.armed_side is None:
                state.armed_side = side
                state.armed_bar_index = bar_index
                state.armed_zscore = zscore
                state.armed_entry_z = entry_z
            return SignalOutput(
                pair_id=self.pair_id, bar_index=bar_index, ready=True,
                action="none", side=side, zscore=zscore,
                entry_z_upper=state.entry_z_upper, entry_z_lower=state.entry_z_lower,
                exit_z=self.exit_z, signal_strength=abs(zscore),
                entry_thresholds_ready=state.entry_thresholds_ready,
                reason=f"two-stage arm {side} z={zscore:.4f}",
                para={"signal": {"method": "zscore_reversion", "two_stage": "armed"}},
            )

        if side == "long_x" and zscore >= trigger and state.armed_side is None:
            state.armed_side = side
            state.armed_bar_index = bar_index
            state.armed_zscore = zscore
            state.armed_trigger_z = trigger
            state.armed_entry_z = entry_z
        elif side == "short_x" and zscore <= trigger and state.armed_side is None:
            state.armed_side = side
            state.armed_bar_index = bar_index
            state.armed_zscore = zscore
            state.armed_trigger_z = trigger
            state.armed_entry_z = entry_z

        return SignalOutput(
            pair_id=self.pair_id, bar_index=bar_index, ready=True,
            action="none", side=side, zscore=zscore,
            entry_z_upper=state.entry_z_upper, entry_z_lower=state.entry_z_lower,
            exit_z=self.exit_z, signal_strength=abs(zscore),
            entry_thresholds_ready=state.entry_thresholds_ready,
            reason=f"two-stage {'armed' if state.armed_side else 'idle'} z={zscore:.4f}",
            para={"signal": {"method": "zscore_reversion", "two_stage": "armed" if state.armed_side else "idle"}},
        )

    def _two_stage_fire_check(
        self, state, zscore, bar_index, estimator=None, para=None
    ):
        """Check if armed zscore has reverted into the entry band (fire condition)."""
        armed_side = state.armed_side
        self._refresh_armed_extreme(state, zscore)
        entry_z = state.armed_entry_z
        if entry_z is None:
            entry_z = self._effective_entry_z(armed_side, state)

        if self._crossed_opposite_entry_band(state, armed_side, zscore):
            old_side = armed_side
            self._clear_two_stage(state)
            return SignalOutput(
                pair_id=self.pair_id, bar_index=bar_index, ready=True,
                action="none", side=None, zscore=zscore,
                entry_z_upper=state.entry_z_upper, entry_z_lower=state.entry_z_lower,
                exit_z=self.exit_z, signal_strength=abs(zscore),
                entry_thresholds_ready=state.entry_thresholds_ready,
                reason=f"two-stage reset {old_side}: crossed opposite entry band z={zscore:.4f}",
                para={
                    "signal": {
                        "method": "zscore_reversion",
                        "two_stage": "reset_opposite_band",
                        "old_side": old_side,
                    }
                },
            )

        if self._crossed_zero_after_arm(armed_side, zscore):
            old_side = armed_side
            self._clear_two_stage(state)
            return SignalOutput(
                pair_id=self.pair_id, bar_index=bar_index, ready=True,
                action="none", side=None, zscore=zscore,
                entry_z_upper=state.entry_z_upper, entry_z_lower=state.entry_z_lower,
                exit_z=self.exit_z, signal_strength=abs(zscore),
                entry_thresholds_ready=state.entry_thresholds_ready,
                reason=f"two-stage reset {old_side}: crossed zero z={zscore:.4f}",
                para={
                    "signal": {
                        "method": "zscore_reversion",
                        "two_stage": "reset_crossed_zero",
                        "old_side": old_side,
                    }
                },
            )

        if self._two_stage_expired(state, bar_index):
            old_side = armed_side
            elapsed = bar_index - (state.armed_bar_index or bar_index)
            self._clear_two_stage(state)
            return SignalOutput(
                pair_id=self.pair_id, bar_index=bar_index, ready=True,
                action="none", side=None, zscore=zscore,
                entry_z_upper=state.entry_z_upper, entry_z_lower=state.entry_z_lower,
                exit_z=self.exit_z, signal_strength=abs(zscore),
                entry_thresholds_ready=state.entry_thresholds_ready,
                reason=f"two-stage reset {old_side}: expired after {elapsed} bars",
                para={
                    "signal": {
                        "method": "zscore_reversion",
                        "two_stage": "reset_expired",
                        "old_side": old_side,
                        "elapsed_bars": elapsed,
                        "max_wait_bars": self.two_stage_max_wait_bars,
                    }
                },
            )

        # Determine if zscore has reverted into the entry zone
        if armed_side == "long_x" and zscore <= entry_z:
            pass  # Positive z-score reverted downward; check filters below.
        elif armed_side == "short_x" and zscore >= entry_z:
            pass  # Negative z-score reverted upward; check filters below.
        else:
            return SignalOutput(
                pair_id=self.pair_id, bar_index=bar_index, ready=True,
                action="none", side=armed_side, zscore=zscore,
                entry_z_upper=state.entry_z_upper, entry_z_lower=state.entry_z_lower,
                exit_z=self.exit_z, signal_strength=abs(zscore),
                entry_thresholds_ready=state.entry_thresholds_ready,
                reason=f"two-stage armed waiting: z={zscore:.4f} entry_z={entry_z:.4f}",
                para={"signal": {"method": "zscore_reversion", "two_stage": "waiting"}},
            )

        if self._missed_entry_window(armed_side, zscore, entry_z):
            old_side = armed_side
            lower_bound = self._entry_window_lower_bound(entry_z)
            self._clear_two_stage(state)
            return SignalOutput(
                pair_id=self.pair_id, bar_index=bar_index, ready=True,
                action="none", side=None, zscore=zscore,
                entry_z_upper=state.entry_z_upper, entry_z_lower=state.entry_z_lower,
                exit_z=self.exit_z, signal_strength=abs(zscore),
                entry_thresholds_ready=state.entry_thresholds_ready,
                reason=(
                    f"two-stage reset {old_side}: missed entry window "
                    f"z={zscore:.4f} entry_z={entry_z:.4f} lower_bound={lower_bound:.4f}"
                ),
                para={
                    "signal": {
                        "method": "zscore_reversion",
                        "two_stage": "reset_missed_entry_window",
                        "old_side": old_side,
                        "entry_z": entry_z,
                        "entry_window_lower_bound": lower_bound,
                    }
                },
            )

        # Reversion filter
        checks = self._reversion_checks(state, zscore, armed_side, bar_index)
        if not checks["passed"]:
            return SignalOutput(
                pair_id=self.pair_id, bar_index=bar_index, ready=True,
                action="none", side=armed_side, zscore=zscore,
                entry_z_upper=state.entry_z_upper, entry_z_lower=state.entry_z_lower,
                exit_z=self.exit_z, signal_strength=abs(zscore),
                entry_thresholds_ready=state.entry_thresholds_ready,
                reason=checks["reason"],
                para={"signal": {"method": "zscore_reversion", "two_stage": "blocked", "reversion_checks": checks}},
            )

        entry_checks = self._entry_checks(estimator, zscore, para)
        if not entry_checks["passed"]:
            return self._blocked_entry_output(
                state, zscore, armed_side, bar_index, entry_checks, checks,
                two_stage=True,
            )

        self._clear_two_stage(state)
        return SignalOutput(
            pair_id=self.pair_id, bar_index=bar_index, ready=True,
            action="open", side=armed_side, zscore=zscore,
            entry_z_upper=state.entry_z_upper, entry_z_lower=state.entry_z_lower,
            exit_z=self.exit_z, signal_strength=abs(zscore),
            entry_thresholds_ready=state.entry_thresholds_ready,
            reason=f"two-stage fire {armed_side} z={zscore:.4f}",
            para={
                "signal": {
                    "method": "zscore_reversion",
                    "two_stage": "fire",
                    "reversion_checks": checks,
                }
            },
        )

    # ---- Filters ----

    def _reversion_checks(self, state, zscore, side, bar_index) -> dict:
        """Run the optional MA reversion filter; return {passed: bool, reason: str}."""
        # Reversion MA filter
        # long_x comes from positive z-score, so require downward reversion.
        # short_x comes from negative z-score, so require upward reversion.
        if self.reversion_filter_enabled and len(state.zscore_history) >= self.reversion_min_samples:
            recent = tail_values(state.zscore_history, self.reversion_ma_lookback_bars)
            short_n = max(5, len(recent) // 4)
            short_ma = float(np.mean(recent[-short_n:]))
            long_ma = float(np.mean(recent))
            gap = short_ma - long_ma
            if side == "long_x" and gap >= 0.0:
                return {"passed": False, "reason": f"reversion MA gap={gap:.4f} >= 0 (need negative)"}
            if side == "short_x" and gap <= 0.0:
                return {"passed": False, "reason": f"reversion MA gap={gap:.4f} <= 0 (need positive)"}

        return {"passed": True, "reason": "ok"}

    def _entry_checks(self, estimator, zscore: float, para: dict | None) -> dict:
        if self.pair_quality_filter_enabled:
            quality = (para or {}).get("pair_quality") or {}
            samples = int(quality.get("samples") or 0)
            corr = quality.get("return_corr")
            ready = bool(quality.get("ready"))
            if (
                not ready
                or samples < self.pair_quality_min_samples
                or corr is None
                or abs(float(corr)) < self.pair_quality_min_corr
            ):
                return {
                    "passed": False,
                    "blocked_by": "pair_quality",
                    "reason": (
                        f"pair quality corr blocked: ready={ready} samples={samples} "
                        f"corr={corr}"
                    ),
                }

        if self.cost_filter_enabled:
            spread_std = float(getattr(estimator, "spread_std", None) or 0.0)
            expected_move = max(0.0, (abs(float(zscore)) - self.exit_z) * spread_std)
            round_trip_cost = 2.0 * (
                self.cost_fee_rate + self.cost_slippage_bps / 10000.0
            )
            required_move = round_trip_cost * self.cost_buffer_multiplier
            if expected_move <= required_move:
                return {
                    "passed": False,
                    "blocked_by": "cost",
                    "reason": (
                        f"expected move {expected_move:.6g} does not cover "
                        f"required cost {required_move:.6g}"
                    ),
                    "expected_move": expected_move,
                    "required_move": required_move,
                }
        return {"passed": True, "reason": "ok"}

    def _blocked_entry_output(
        self, state, zscore, side, bar_index, entry_checks, reversion_checks,
        two_stage: bool = False,
    ):
        return SignalOutput(
            pair_id=self.pair_id, bar_index=bar_index, ready=True,
            action="none", side=side, zscore=zscore,
            entry_z_upper=state.entry_z_upper, entry_z_lower=state.entry_z_lower,
            exit_z=self.exit_z, signal_strength=abs(zscore),
            entry_thresholds_ready=state.entry_thresholds_ready,
            reason=entry_checks["reason"],
            para={"signal": {
                "method": "zscore_reversion",
                "two_stage": "blocked" if two_stage else None,
                "entry_checks": entry_checks,
                "reversion_checks": reversion_checks,
            }},
        )

    # ---- Helpers ----

    def _update_thresholds(self, state: SignalState):
        if self.entry_rule_method == "fixed_z":
            state.entry_z_upper = self.entry_z
            state.entry_z_lower = -self.entry_z
        elif (self.entry_rule_method == "percentile"
              and len(state.zscore_history) >= self.entry_rule_min_samples):
            lower, upper = percentiles_from_tail(
                state.zscore_history,
                self.entry_rule_lookback_bars,
                [self.entry_rule_lower_percentile, self.entry_rule_upper_percentile],
            )
            if upper is None or lower is None:
                state.entry_z_upper = self.entry_z
                state.entry_z_lower = -self.entry_z
                return
            if self.entry_rule_min_abs_entry_z:
                upper = max(upper, self.entry_rule_min_abs_entry_z)
                lower = min(lower, -self.entry_rule_min_abs_entry_z)
            if self.entry_rule_max_abs_entry_z:
                upper = min(upper, self.entry_rule_max_abs_entry_z)
                lower = max(lower, -self.entry_rule_max_abs_entry_z)
            state.entry_z_upper = upper
            state.entry_z_lower = lower
        else:
            state.entry_z_upper = self.entry_z
            state.entry_z_lower = -self.entry_z

    def _effective_trigger_z(self, side: str) -> float | None:
        if self.two_stage_trigger_z is not None:
            trigger = float(self.two_stage_trigger_z)
            return trigger if side == "long_x" else -trigger
        if self.two_stage_levels:
            for level in self.two_stage_levels:
                if isinstance(level, (list, tuple)) and len(level) >= 2:
                    tz, _ez = float(level[0]), float(level[1])
                    return tz if side == "long_x" else -tz
        return None

    def _effective_entry_z(self, side: str | None, state: SignalState) -> float:
        if self.two_stage_entry_z is not None:
            entry = abs(float(self.two_stage_entry_z))
            return entry if side == "long_x" else -entry
        if self.two_stage_levels:
            for level in self.two_stage_levels:
                if isinstance(level, (list, tuple)) and len(level) >= 2:
                    entry = abs(float(level[1]))
                    return entry if side == "long_x" else -entry
        return state.entry_z_upper if side == "long_x" else state.entry_z_lower

    def _entry_window_lower_bound(self, entry_z: float) -> float:
        if self.two_stage_entry_window_z is None:
            return 0.0
        return max(0.0, abs(float(entry_z)) - self.two_stage_entry_window_z)

    def _missed_entry_window(self, side: str | None, zscore: float, entry_z: float) -> bool:
        if self.two_stage_entry_window_z is None:
            return False
        lower_bound = self._entry_window_lower_bound(entry_z)
        if side == "long_x":
            return zscore < lower_bound
        if side == "short_x":
            return zscore > -lower_bound
        return False

    def _crossed_opposite_entry_band(self, state: SignalState, armed_side: str | None, zscore: float) -> bool:
        if armed_side == "long_x":
            return zscore <= state.entry_z_lower
        if armed_side == "short_x":
            return zscore >= state.entry_z_upper
        return False

    def _crossed_zero_after_arm(self, armed_side: str | None, zscore: float) -> bool:
        if armed_side == "long_x":
            return zscore <= 0.0
        if armed_side == "short_x":
            return zscore >= 0.0
        return False

    def _two_stage_expired(self, state: SignalState, bar_index: int) -> bool:
        if self.two_stage_max_wait_bars <= 0 or state.armed_bar_index is None:
            return False
        return bar_index - state.armed_bar_index > self.two_stage_max_wait_bars

    def _refresh_armed_extreme(self, state: SignalState, zscore: float) -> None:
        if state.armed_side == "long_x":
            if state.armed_zscore is None or zscore > state.armed_zscore:
                state.armed_zscore = zscore
        elif state.armed_side == "short_x":
            if state.armed_zscore is None or zscore < state.armed_zscore:
                state.armed_zscore = zscore

    def _clear_two_stage(self, state: SignalState):
        state.armed_side = None
        state.armed_bar_index = None
        state.armed_zscore = None
        state.armed_trigger_z = None
        state.armed_entry_z = None
