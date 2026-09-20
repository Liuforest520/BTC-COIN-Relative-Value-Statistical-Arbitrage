"""PairPipeline: estimator -> signal -> sizing for one pair. Lives in strategy because it orchestrates the steps."""
from __future__ import annotations

from core.modules.models.pipeline_types import (
    PairPipelineResult,
    PairRuntimeState,
    RawPairTarget,
)
from core.modules.data.rolling_window import tail_values
from core.modules.strategy.config import EstimatorConfig, ExecutionConfig, PairDefinition, SignalConfig, SizingConfig


class PairPipeline:
    """Estimator -> Signal -> Sizing pipeline for one pair."""

    def __init__(
        self,
        pair_def: PairDefinition,
        estimator_ctor: type,
        estimator_cfg: EstimatorConfig,
        signal_ctor: type | None,
        signal_cfg: SignalConfig | None,
        sizing_ctor: type | None,
        sizing_cfg: SizingConfig | None,
        execution_cfg: ExecutionConfig | None,
    ):
        from core.modules.models import OrderType
        self.pair_def = pair_def
        self.state = PairRuntimeState(pair_id=pair_def.pair_id)

        est_kwargs = _dataclass_to_dict(estimator_cfg)
        est_kwargs["pair_id"] = pair_def.pair_id
        if pair_def.model_lookback_bars_override:
            est_kwargs["model_lookback_bars"] = min(
                est_kwargs.get("model_lookback_bars", 10080),
                pair_def.model_lookback_bars_override,
            )
        self.estimator = estimator_ctor(**est_kwargs)

        if signal_ctor and signal_cfg:
            sig_kwargs = _dataclass_to_dict(signal_cfg)
            sig_kwargs["pair_id"] = pair_def.pair_id
            self.signal = signal_ctor(**sig_kwargs)
        else:
            self.signal = None

        if sizing_ctor and sizing_cfg:
            siz_kwargs = _dataclass_to_dict(sizing_cfg)
            siz_kwargs["pair_id"] = pair_def.pair_id
            self.sizing = sizing_ctor(**siz_kwargs)
        else:
            self.sizing = None

        self.execution_cfg = execution_cfg or ExecutionConfig()
        ot = self.execution_cfg.order_type
        self._order_type = OrderType.LIMIT if ot == "limit" else OrderType.MARKET
        self._bar_index = 0
        self._last_block_reason = ""
        self._last_result = None
        self._last_zscore = None
        self._last_action = None
        self.collect_diagnostics = True

    def on_bar(self, bundle, compute_sizing: bool = True) -> PairPipelineResult:
        """Process one bar for this pair."""
        x_close = bundle.x_bar.close if bundle.x_bar else 0.0
        y_close = bundle.y_bar.close if bundle.y_bar else 0.0
        sizing_state = self.state.sizing_state
        has_position = (
            sizing_state.position_side is not None
            and (sizing_state.x_quantity > 0 or sizing_state.y_quantity > 0)
        )
        if self.collect_diagnostics:
            runtime_para = {
                "pair": {
                    "pair_id": self.pair_def.pair_id,
                    "ts": bundle.ts,
                    "bar_index": bundle.bar_index,
                },
                "position": {
                    "has_position": has_position,
                    "side": sizing_state.position_side,
                    "entry_count": sizing_state.entry_count,
                },
            }
        else:
            # Estimators only consume has_position from the runtime metadata.
            runtime_para = {"position": {"has_position": has_position}}

        est_out = self.estimator.update(
            self.state.estimator_state, x_close, y_close, bundle.bar_index,
            para=runtime_para,
        )

        result_para = (
            _merge_para(runtime_para, getattr(est_out, "para", {}))
            if self.collect_diagnostics else {}
        )
        if not est_out.ready:
            self.state.is_ready = False
            self.state.last_block_reason = est_out.reason
            result = PairPipelineResult(
                pair_id=self.pair_def.pair_id,
                estimator=est_out, ready=False, block_reason=est_out.reason,
                para=result_para,
            )
            self._last_result = result
            return result

        position_para = {
            "has_position": has_position,
            "side": sizing_state.position_side,
        }
        signal_para = (
            _merge_para(result_para, {"position": position_para})
            if self.collect_diagnostics else {"position": position_para}
        )
        if self.collect_diagnostics:
            result_para = signal_para
        sig_out = self.signal.evaluate(
            self.state.signal_state, est_out, bundle.bar_index,
            para=signal_para,
        ) if self.signal else None

        if sig_out and self.collect_diagnostics:
            result_para = _merge_para(result_para, getattr(sig_out, "para", {}))
        raw_target = RawPairTarget(pair_id=self.pair_def.pair_id, ready=False)
        if compute_sizing and sig_out and sig_out.action == "open" and sig_out.side and self.sizing:
            x_price = bundle.x_bar.close if bundle.x_bar else 0
            y_price = bundle.y_bar.close if bundle.y_bar else 0
            # Per-leg vol from price history (for volatility_neutral sizing)
            long_vol = short_vol = None
            est_st = self.state.estimator_state
            xh, yh = est_st.x_close_history, est_st.y_close_history
            if len(xh) >= 20 and len(yh) >= 20:
                import numpy as np
                window = min(1440, len(xh))
                x_ret = np.diff(np.log(np.array(tail_values(xh, window)).clip(min=1e-12)))
                y_ret = np.diff(np.log(np.array(tail_values(yh, window)).clip(min=1e-12)))
                long_vol = float(np.std(x_ret, ddof=1)) if len(x_ret) > 5 else None
                short_vol = float(np.std(y_ret, ddof=1)) if len(y_ret) > 5 else None
            try:
                raw_target = self.sizing.compute(
                    self.state.sizing_state, sig_out, est_out,
                    x_price, y_price, long_vol=long_vol, short_vol=short_vol,
                    x_history=xh, y_history=yh, para=result_para,
                )
            except TypeError:
                raw_target = self.sizing.compute(
                    self.state.sizing_state, sig_out, est_out, x_price, y_price
                )
            if self.collect_diagnostics:
                result_para = _merge_para(result_para, getattr(raw_target, "para", {}))
        raw_target.para = result_para

        self.state.is_ready = True
        self.state.last_bar_index = bundle.bar_index
        result = PairPipelineResult(
            pair_id=self.pair_def.pair_id,
            estimator=est_out, signal=sig_out, raw_target=raw_target,
            ready=True, block_reason="", para=result_para,
        )
        self._last_result = result
        return result

def _dataclass_to_dict(dc) -> dict:
    import dataclasses
    result = {}
    for f in dataclasses.fields(dc):
        v = getattr(dc, f.name)
        if v is not None:
            result[f.name] = v
    return result


def _merge_para(base: dict | None, extra: dict | None) -> dict:
    result = dict(base or {})
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_para(result[key], value)
        else:
            result[key] = value
    return result
