"""Fixed-threshold MA-reversion signal."""
from __future__ import annotations

import numpy as np

from core.modules.data.rolling_window import ensure_deque, tail_values
from core.modules.models.pipeline_types import EstimatorOutput, SignalOutput, SignalState, register_signal
from core.modules.signals.exit_rules import resolve_position_exit_decision


@register_signal("zscore_reversion_ma")
class MAReversionSignal:
    """Enter at a fixed z-score threshold after MA confirmation."""

    signal_method = "zscore_reversion_ma"

    def __init__(
        self,
        pair_id: str = "",
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        reversion_ma_lookback_bars: int = 60,
        reversion_ma_short_lookback_bars: int = 15,
        reversion_min_samples: int | None = None,
        **kwargs,
    ):
        self.pair_id = pair_id
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.reversion_ma_lookback_bars = max(5, int(reversion_ma_lookback_bars))
        configured_short = max(1, int(reversion_ma_short_lookback_bars))
        self.reversion_ma_short_lookback_bars = min(configured_short, self.reversion_ma_lookback_bars)
        self.reversion_min_samples = max(5, int(reversion_min_samples or self.reversion_ma_lookback_bars))

    def evaluate(self, state: SignalState, estimator: EstimatorOutput, bar_index: int, para: dict | None = None) -> SignalOutput:
        if not estimator.ready or estimator.spread_std is None:
            return SignalOutput(pair_id=self.pair_id, bar_index=bar_index, ready=False, reason="estimator not ready")
        std = float(estimator.spread_std)
        latest = estimator.latest_spread
        zscore = 0.0 if latest is None or estimator.spread_mean is None or std <= 0 else (float(latest) - float(estimator.spread_mean)) / std
        state.zscore_history = ensure_deque(state.zscore_history, max(self.reversion_ma_lookback_bars, self.reversion_min_samples))
        state.zscore_history.append(zscore)
        should_close, close_reason = resolve_position_exit_decision(para, zscore, self.exit_z)
        if should_close:
            return SignalOutput(pair_id=self.pair_id, bar_index=bar_index, ready=True, action="close", zscore=zscore, exit_z=self.exit_z, reason=close_reason, para={"signal": {"method": self.signal_method, "action": "close", "zscore": zscore, "reason": close_reason}})
        side = "long_x" if zscore >= self.entry_z else "short_x" if zscore <= -self.entry_z else None
        action = "none"
        reason = f"zscore={zscore:.4f}"
        if side is not None:
            checks = self._reversion_check(state, zscore, side)
            if checks["passed"]:
                action = "open"
                reason = f"zscore threshold and MA reversion passed: {zscore:.4f}"
            else:
                reason = checks["reason"]
        return SignalOutput(pair_id=self.pair_id, bar_index=bar_index, ready=True, action=action, side=side if action == "open" else None, zscore=zscore, entry_z_upper=self.entry_z, entry_z_lower=-self.entry_z, exit_z=self.exit_z, signal_strength=abs(zscore), entry_thresholds_ready=True, reason=reason, para={"signal": {"method": self.signal_method, "action": action, "side": side if action == "open" else None, "zscore": zscore, "reason": reason}})

    def _reversion_check(self, state: SignalState, zscore: float, side: str) -> dict:
        if len(state.zscore_history) < self.reversion_min_samples:
            return {"passed": False, "reason": f"MA reversion warmup {len(state.zscore_history)}/{self.reversion_min_samples}"}
        recent = tail_values(state.zscore_history, self.reversion_ma_lookback_bars)
        short_n = min(len(recent), self.reversion_ma_short_lookback_bars)
        short_ma = float(np.mean(recent[-short_n:]))
        long_ma = float(np.mean(recent))
        gap = short_ma - long_ma
        if side == "long_x" and gap >= 0:
            return {"passed": False, "reason": f"MA gap={gap:.4f} >= 0"}
        if side == "short_x" and gap <= 0:
            return {"passed": False, "reason": f"MA gap={gap:.4f} <= 0"}
        return {"passed": True, "reason": "ok"}


ZScoreReversionSignal = MAReversionSignal
