"""
Z-score based signal generator: simple threshold crossing.

Selecting ``signal.method: simple_zscore`` means "hit ±entry_z and enter
immediately", with no two-stage arming and no MA reversion confirmation.  The
two confirmation variants live in their own modules:

* ``zscore_reversion_two_stage`` -- arm at the trigger, enter on the pullback
* ``zscore_reversion_ma``        -- require the z-score MA to turn first
"""
from __future__ import annotations

from core.modules.models.pipeline_types import (
    EstimatorOutput,
    SignalOutput,
    SignalState,
    register_signal,
)
from core.modules.data.rolling_window import ensure_deque, percentiles_from_tail
from core.modules.signals.exit_rules import resolve_position_exit_decision


@register_signal("simple_zscore")
@register_signal("zscore")  # legacy alias, identical behaviour
class SimpleZScoreSignal:
    """Simple z-score threshold crossing signal.

    Open (long X / short Y) as soon as ``zscore >= entry_z_upper``.
    Open (short X / long Y) as soon as ``zscore <= entry_z_lower``.
    There is no confirmation layer: hitting the threshold *is* the entry.

    Exit semantics: positive ``exit_z`` is a symmetric band, zero requires a
    sign reversal, and negative ``exit_z`` requires a reversal beyond its
    absolute value.
    """

    signal_method = "simple_zscore"

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
        self.max_history = max(1, int(self.entry_rule_lookback_bars), int(self.entry_rule_min_samples))

    def evaluate(self, state: SignalState, estimator: EstimatorOutput,
                 bar_index: int, para: dict | None = None) -> SignalOutput:
        if not estimator.ready or estimator.spread_std is None:
            return SignalOutput(pair_id=self.pair_id, bar_index=bar_index,
                               ready=False, reason="estimator not ready",
                               para={"signal": {"method": self.signal_method, "blocked_by": "estimator"}})

        # Compute zscore from estimator output
        std = estimator.spread_std
        if std <= 0 or estimator.spread_mean is None:
            zscore = 0.0
        else:
            latest = estimator.latest_spread
            if latest is None:
                zscore = 0.0
            else:
                zscore = (latest - estimator.spread_mean) / std

        # Track zscore history for percentile-based thresholds
        state.zscore_history = ensure_deque(state.zscore_history, self.max_history)
        state.zscore_history.append(zscore)

        # Update dynamic thresholds
        should_update = (
            state.last_entry_rule_update_index is None
            or bar_index - state.last_entry_rule_update_index >= self.entry_rule_update_interval_bars
        )
        if should_update:
            self._update_thresholds(state)
            state.last_entry_rule_update_index = bar_index

        state.entry_thresholds_ready = (state.entry_z_upper is not None)

        # --- Decision ---
        exit_z = self.exit_z
        side = None
        action = "none"
        reason = f"zscore={zscore:.4f}"

        should_close, close_reason = resolve_position_exit_decision(
            para, zscore, exit_z
        )
        if should_close:
            action = "close"
            if close_reason in {"direction_reversed", "reversal_threshold_reached"}:
                position_side = ((para or {}).get("position") or {}).get("side")
                if close_reason == "reversal_threshold_reached":
                    reason = (
                        f"zscore reached opposite exit {self.exit_z:.4f} "
                        f"for {position_side}: {zscore:.4f}"
                    )
                else:
                    reason = (
                        f"zscore direction reversed for {position_side}: {zscore:.4f}"
                    )
            else:
                reason = (
                    f"zscore within exit band: {zscore:.4f}"
                )
        # z >= upper: Y overvalued relative to X, so buy X and sell Y.
        elif zscore >= state.entry_z_upper:
            action = "open"
            side = "long_x"
            reason = f"zscore {zscore:.4f} >= upper {state.entry_z_upper:.4f}"
        # z <= lower: Y undervalued relative to X, so sell X and buy Y.
        elif zscore <= state.entry_z_lower:
            action = "open"
            side = "short_x"
            reason = f"zscore {zscore:.4f} <= lower {state.entry_z_lower:.4f}"

        return SignalOutput(
            pair_id=self.pair_id, bar_index=bar_index, ready=True,
            action=action, side=side, zscore=zscore,
            entry_z_upper=state.entry_z_upper, entry_z_lower=state.entry_z_lower,
            exit_z=exit_z, signal_strength=abs(zscore),
            entry_thresholds_ready=state.entry_thresholds_ready,
            reason=reason,
            para={
                "signal": {
                    "method": self.signal_method,
                    "action": action,
                    "side": side,
                    "zscore": zscore,
                    "reason": reason,
                }
            },
        )

    def _update_thresholds(self, state: SignalState):
        if self.entry_rule_method == "fixed_z":
            state.entry_z_upper = self.entry_z
            state.entry_z_lower = -self.entry_z
        elif self.entry_rule_method == "percentile" and len(state.zscore_history) >= self.entry_rule_min_samples:
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
            state.entry_z_upper = upper
            state.entry_z_lower = lower
        else:
            state.entry_z_upper = self.entry_z
            state.entry_z_lower = -self.entry_z


# Backwards-compatible class name for imports written before the signal split.
ZScoreSignal = SimpleZScoreSignal

__all__ = ["SimpleZScoreSignal", "ZScoreSignal"]
