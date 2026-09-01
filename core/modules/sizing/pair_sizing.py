"""
Pair-level position sizing strategies.
"""
from __future__ import annotations

from math import isfinite

import numpy as np

from core.modules.models.pipeline_types import (
    EstimatorOutput,
    RawPairTarget,
    SignalOutput,
    SizingState,
    register_sizing,
)
from core.modules.data.rolling_window import tail_values


@register_sizing("fixed_notional")
class FixedNotionalSizing:
    def __init__(self, notional: float = 10000.0, min_notional: float = 0.0, **kwargs):
        self.notional = float(notional)
        self.min_notional = float(min_notional)

    def compute(self, state: SizingState, signal: SignalOutput,
                estimator: EstimatorOutput, x_price: float, y_price: float,
                long_vol: float | None = None, short_vol: float | None = None,
                x_history=None,
                y_history=None,
                para: dict | None = None) -> RawPairTarget:
        if signal.action != "open" or not signal.side:
            return RawPairTarget(pair_id=signal.pair_id, ready=False, reason="no signal")

        x_weight = 0.5
        y_weight = 0.5
        target_hedge_ratio = _long_short_target_ratio(signal.side, x_weight, y_weight)

        return RawPairTarget(
            pair_id=signal.pair_id, ready=True, side=signal.side,
            x_weight=x_weight, y_weight=y_weight,
            x_price=x_price, y_price=y_price,
            hedge_ratio=target_hedge_ratio,
            reason="fixed_notional",
            para=_sizing_para(signal, {
                "method": "fixed_notional",
                "x_weight": x_weight,
                "y_weight": y_weight,
            }),
        )


@register_sizing("beta_neutral")
class BetaNeutralSizing:
    """Allocate gross exposure split by beta.

    Supports open and add actions. Add sizing can be scaled by multiplier,
    reduced by exposure cap, or stepped by z-score deviation.
    """

    def __init__(self, notional: float = 10000.0, equity: float = 100000.0,
                 gross_exposure_ratio: float = 0.2, max_gross_exposure_ratio: float = 1.0,
                 min_notional: float = 0.0,
                 # Add-on params
                 max_add_times: int = 0,
                 add_interval_bars: int = 1,
                 add_z_step: float = 0.0,
                 add_gross_exposure_ratio: float | None = None,
                 add_size_multiplier: float | None = None,
                 require_add_reversion_vs_last_entry: bool = False,
                 add_reversion_buffer: float = 0.0,
                 hedge_beta_lookback_bars: int = 1440,
                 hedge_beta_min_samples: int = 30,
                 # Volatility neutral params
                 half_life_add_interval_multiplier: float = 1.0,
                 add_interval_mode: str = "fixed",
                 **kwargs):
        self.notional = float(notional)
        self.equity = float(equity)
        self.gross_exposure_ratio = float(gross_exposure_ratio)
        self.max_gross_exposure_ratio = float(max_gross_exposure_ratio)
        self.min_notional = float(min_notional)
        self.max_add_times = int(max_add_times)
        self.add_interval_bars = int(add_interval_bars)
        self.add_z_step = float(add_z_step)
        self.add_gross_exposure_ratio = (
            float(add_gross_exposure_ratio) if add_gross_exposure_ratio else self.gross_exposure_ratio
        )
        self.add_size_multiplier = float(add_size_multiplier) if add_size_multiplier else 1.0
        self.require_add_reversion_vs_last_entry = bool(require_add_reversion_vs_last_entry)
        self.add_reversion_buffer = float(add_reversion_buffer)
        self.hedge_beta_lookback_bars = int(hedge_beta_lookback_bars)
        self.hedge_beta_min_samples = int(hedge_beta_min_samples)
        self.half_life_add_interval_multiplier = float(half_life_add_interval_multiplier)
        self.add_interval_mode = add_interval_mode

    def compute(self, state: SizingState, signal: SignalOutput,
                estimator: EstimatorOutput, x_price: float, y_price: float,
                long_vol: float | None = None, short_vol: float | None = None,
                x_history=None,
                y_history=None,
                para: dict | None = None) -> RawPairTarget:
        if not signal.side:
            return RawPairTarget(pair_id=signal.pair_id, ready=False, reason="no signal")

        if signal.action == "open":
            return self._open_sizing(state, signal, estimator, x_price, y_price, x_history, y_history)
        if signal.action == "add":
            return self._add_sizing(state, signal, estimator, x_price, y_price, x_history, y_history)
        return RawPairTarget(pair_id=signal.pair_id, ready=False, reason="action not supported")

    def _open_sizing(self, state, signal, estimator, x_price, y_price, x_history=None, y_history=None):
        returns_beta, beta_samples = self._returns_beta(x_history, y_history)
        beta_source = "returns_on_signal" if returns_beta is not None else "estimator"
        hedge_beta = returns_beta if returns_beta is not None else (estimator.hedge_beta or 1.0)
        beta = abs(float(hedge_beta))
        if not isfinite(beta) or beta <= 1e-12:
            beta = 1.0
            hedge_beta = 1.0
            beta_source = "fallback"
        x_weight = beta / (1 + beta)
        y_weight = 1.0 / (1 + beta)
        target_hedge_ratio = _long_short_target_ratio(signal.side, x_weight, y_weight)

        return RawPairTarget(
            pair_id=signal.pair_id, ready=True, side=signal.side,
            x_weight=x_weight, y_weight=y_weight,
            x_price=x_price, y_price=y_price,
            hedge_ratio=target_hedge_ratio,
            signal_strength=signal.signal_strength,
            reason="beta_neutral",
            para=_sizing_para(signal, {
                "method": "beta_neutral",
                "beta": beta,
                "raw_beta": hedge_beta,
                "x_weight": x_weight,
                "y_weight": y_weight,
                "target_hedge_ratio": target_hedge_ratio,
                "beta_source": beta_source,
                "beta_samples": beta_samples,
                "beta_lookback_bars": self.hedge_beta_lookback_bars,
            }),
        )

    def _add_sizing(self, state, signal, estimator, x_price, y_price, x_history=None, y_history=None):
        """Add to existing position with scaling."""
        if self.max_add_times > 0 and state.entry_count >= self.max_add_times:
            return RawPairTarget(pair_id=signal.pair_id, ready=False, reason="max add reached")

        # Scale add size by multiplier
        base = self._open_sizing(state, signal, estimator, x_price, y_price, x_history, y_history)
        scale = self.add_size_multiplier

        # If add_z_step is set, scale by z-score depth
        if self.add_z_step > 0 and signal.zscore is not None:
            base_z = 2.0  # nominal base z
            depth = abs(signal.zscore) / base_z
            scale *= depth

        return RawPairTarget(
            pair_id=signal.pair_id, ready=True, side=signal.side,
            x_weight=base.x_weight,
            y_weight=base.y_weight,
            x_price=base.x_price,
            y_price=base.y_price,
            hedge_ratio=base.hedge_ratio,
            signal_strength=signal.signal_strength,
            reason=f"add #{state.entry_count + 1} scale={scale:.2f}",
            para=_sizing_para(signal, {
                **(base.para.get("sizing", {}) if isinstance(base.para, dict) else {}),
                "method": "beta_neutral",
                "action": "add",
                "scale": scale,
            }),
        )

    def _returns_beta(self, x_history, y_history):
        if x_history is None or y_history is None:
            return None, 0
        target_returns = max(1, int(self.hedge_beta_lookback_bars))
        price_window = min(target_returns + 1, len(x_history), len(y_history))
        samples_available = max(0, price_window - 1)
        if samples_available < self.hedge_beta_min_samples:
            return None, samples_available

        x_arr = np.array(tail_values(x_history, price_window), dtype=float)
        y_arr = np.array(tail_values(y_history, price_window), dtype=float)
        x_ret = np.diff(np.log(np.clip(x_arr, 1e-12, None)))
        y_ret = np.diff(np.log(np.clip(y_arr, 1e-12, None)))
        mask = np.isfinite(x_ret) & np.isfinite(y_ret)
        x_ret = x_ret[mask]
        y_ret = y_ret[mask]
        samples = len(x_ret)
        if samples < self.hedge_beta_min_samples:
            return None, samples

        var_x = float(np.var(x_ret, ddof=1))
        if not isfinite(var_x) or var_x <= 1e-12:
            return None, samples
        x_centered = x_ret - float(np.mean(x_ret))
        y_centered = y_ret - float(np.mean(y_ret))
        cov_xy = float(np.dot(x_centered, y_centered) / (samples - 1))
        if not isfinite(cov_xy):
            return None, samples
        return cov_xy / var_x, samples


@register_sizing("volatility_neutral")
class VolatilityNeutralSizing:
    """Allocate gross exposure inversely proportional to volatility."""

    def __init__(self, notional: float = 10000.0, equity: float = 100000.0,
                 gross_exposure_ratio: float = 0.2, max_gross_exposure_ratio: float = 1.0,
                 min_notional: float = 0.0, **kwargs):
        self.notional = float(notional)
        self.equity = float(equity)
        self.gross_exposure_ratio = float(gross_exposure_ratio)
        self.max_gross_exposure_ratio = float(max_gross_exposure_ratio)
        self.min_notional = float(min_notional)

    def compute(self, state: SizingState, signal: SignalOutput,
                estimator: EstimatorOutput, x_price: float, y_price: float,
                long_vol: float | None = None, short_vol: float | None = None,
                x_history=None,
                y_history=None,
                para: dict | None = None) -> RawPairTarget:
        if signal.action != "open" or not signal.side:
            return RawPairTarget(pair_id=signal.pair_id, ready=False, reason="no signal")

        vol_source = "input"
        vol_samples = 0
        if (long_vol is None or not isfinite(long_vol) or long_vol <= 0 or
                short_vol is None or not isfinite(short_vol) or short_vol <= 0):
            hist_long_vol, hist_short_vol, vol_samples = _returns_vols(x_history, y_history)
            if hist_long_vol is not None and hist_short_vol is not None:
                long_vol = hist_long_vol
                short_vol = hist_short_vol
                vol_source = "history_on_signal"

        if (long_vol is None or not isfinite(long_vol) or long_vol <= 0 or
                short_vol is None or not isfinite(short_vol) or short_vol <= 0):
            x_weight = 0.5
            y_weight = 0.5
            vol_source = "fallback"
        else:
            x_risk = 1.0 / long_vol
            y_risk = 1.0 / short_vol
            base = x_risk + y_risk
            x_weight = x_risk / base
            y_weight = y_risk / base
        target_hedge_ratio = _long_short_target_ratio(signal.side, x_weight, y_weight)

        return RawPairTarget(
            pair_id=signal.pair_id, ready=True, side=signal.side,
            x_weight=x_weight, y_weight=y_weight,
            x_price=x_price, y_price=y_price,
            hedge_ratio=target_hedge_ratio,
            long_vol=long_vol, short_vol=short_vol,
            signal_strength=signal.signal_strength,
            reason="volatility_neutral",
            para=_sizing_para(
                signal,
                {
                    "method": "volatility_neutral",
                    "long_vol": long_vol,
                    "short_vol": short_vol,
                    "x_weight": x_weight,
                    "y_weight": y_weight,
                    "target_hedge_ratio": target_hedge_ratio,
                    "vol_source": vol_source,
                    "vol_samples": vol_samples,
                },
            ),
        )


def _returns_vols(x_history, y_history, lookback_bars: int = 1440, min_samples: int = 5):
    if x_history is None or y_history is None:
        return None, None, 0
    window = min(lookback_bars, len(x_history), len(y_history))
    if window < min_samples + 1:
        return None, None, max(0, window - 1)

    x_arr = np.array(tail_values(x_history, window), dtype=float)
    y_arr = np.array(tail_values(y_history, window), dtype=float)
    x_ret = np.diff(np.log(np.clip(x_arr, 1e-12, None)))
    y_ret = np.diff(np.log(np.clip(y_arr, 1e-12, None)))
    mask = np.isfinite(x_ret) & np.isfinite(y_ret)
    x_ret = x_ret[mask]
    y_ret = y_ret[mask]
    samples = len(x_ret)
    if samples < min_samples:
        return None, None, samples

    x_vol = float(np.std(x_ret, ddof=1))
    y_vol = float(np.std(y_ret, ddof=1))
    if not isfinite(x_vol) or x_vol <= 0 or not isfinite(y_vol) or y_vol <= 0:
        return None, None, samples
    return x_vol, y_vol, samples


def _long_short_target_ratio(side: str | None, x_notional: float, y_notional: float) -> float:
    if side == "long_x" and y_notional > 0:
        return x_notional / y_notional
    if side == "short_x" and x_notional > 0:
        return y_notional / x_notional
    return 1.0


def _sizing_para(signal: SignalOutput, details: dict) -> dict:
    para = dict(getattr(signal, "para", {}) or {})
    sizing = dict(para.get("sizing", {}) or {})
    sizing.update(details)
    para["sizing"] = sizing
    return para
