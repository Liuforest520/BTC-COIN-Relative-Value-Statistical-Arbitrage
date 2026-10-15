"""Fixed-threshold simple z-score signal."""
from __future__ import annotations
from core.modules.models.pipeline_types import EstimatorOutput, SignalOutput, SignalState, register_signal
from core.modules.signals.exit_rules import resolve_position_exit_decision

@register_signal("simple_zscore")
@register_signal("zscore")
class SimpleZScoreSignal:
    signal_method = "simple_zscore"

    def __init__(self, pair_id: str = "", entry_z: float = 2.0, exit_z: float = 0.5, **kwargs):
        self.pair_id = pair_id
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)

    def evaluate(self, state: SignalState, estimator: EstimatorOutput, bar_index: int, para: dict | None = None) -> SignalOutput:
        if not estimator.ready or estimator.spread_std is None:
            return SignalOutput(pair_id=self.pair_id, bar_index=bar_index, ready=False, reason="estimator not ready")
        latest = estimator.latest_spread
        zscore = 0.0 if latest is None or estimator.spread_mean is None or estimator.spread_std <= 0 else (float(latest) - float(estimator.spread_mean)) / float(estimator.spread_std)
        should_close, close_reason = resolve_position_exit_decision(para, zscore, self.exit_z)
        if should_close:
            return SignalOutput(pair_id=self.pair_id, bar_index=bar_index, ready=True, action="close", zscore=zscore, exit_z=self.exit_z, reason=close_reason)
        if zscore >= self.entry_z:
            action, side = "open", "long_x"
        elif zscore <= -self.entry_z:
            action, side = "open", "short_x"
        else:
            action, side = "none", None
        return SignalOutput(pair_id=self.pair_id, bar_index=bar_index, ready=True, action=action, side=side, zscore=zscore, entry_z_upper=self.entry_z, entry_z_lower=-self.entry_z, exit_z=self.exit_z, signal_strength=abs(zscore), entry_thresholds_ready=True, reason=f"zscore={zscore:.4f}")

ZScoreSignal = SimpleZScoreSignal
