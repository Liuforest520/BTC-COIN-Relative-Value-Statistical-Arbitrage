"""
Unified pipeline type layer for multi-pair statistical arbitrage.

All dataclasses live here so every pipeline stage
(estimator / signal / sizing / portfolio / execution)
speaks a shared vocabulary.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


# ---- Bar types ----

@dataclass
class BarSnapshot:
    ts: int  # ms epoch
    open: float
    high: float
    close: float
    low: float
    volume: float
    exchange: str = ""
    symbol: str = ""


@dataclass
class PairBarBundle:
    pair_id: str
    ts: int
    bar_index: int
    x_bar: BarSnapshot | None = None
    y_bar: BarSnapshot | None = None


# ---- Estimator ----

@dataclass
class EstimatorState:
    x_close_history: deque[float] = field(default_factory=deque)
    y_close_history: deque[float] = field(default_factory=deque)
    spread_history: deque[float] = field(default_factory=deque)
    zscore_history: deque[float] = field(default_factory=deque)

    alpha: float | None = None
    beta: float | None = None
    spread_beta: float | None = None          # signal regression beta
    hedge_beta: float | None = None            # hedge regression beta

    spread_mean: float | None = None
    spread_std: float | None = None
    spread_var: float | None = None
    spread_sample_count: int = 0
    spread_rolling_sum: float = 0.0
    spread_rolling_sumsq: float = 0.0

    residual_phi: float | None = None
    residual_half_life_bars: float | None = None

    cointegration_pass: bool = True
    cointegration_pvalue: float | None = None
    cointegration_stat: float | None = None
    cointegration_critical_value: float | None = None
    cointegration_window_bars: int = 0
    cointegration_sample_count: int = 0
    cointegration_last_check_index: int | None = None
    cointegration_reason: str = ""
    cointegration_block_open: bool = False

    last_model_update_index: int | None = None
    last_hedge_model_update_index: int | None = None
    next_model_update_index: int | None = None
    next_hedge_model_update_index: int | None = None
    last_model_update_skip_index: int | None = None
    last_hedge_model_update_skip_index: int | None = None
    # Kalman/RLS internal state
    kalman_P: Any = None
    kalman_theta: Any = None
    kalman_Q: Any = None
    kalman_R: float | None = None
    kalman_initialized_bar_index: int | None = None
    rls_P: Any = None
    rls_theta: Any = None


@dataclass
class EstimatorOutput:
    pair_id: str
    ts: int | None = None
    bar_index: int = 0
    ready: bool = False

    alpha: float | None = None
    beta: float | None = None
    spread_beta: float | None = None
    hedge_beta: float | None = None
    spread_mean: float | None = None
    spread_std: float | None = None
    spread_var: float | None = None
    spread_sample_count: int = 0

    latest_spread: float | None = None  # for z-score computation

    residual_phi: float | None = None
    residual_half_life_bars: float | None = None

    cointegration_pass: bool = True
    cointegration_pvalue: float | None = None
    cointegration_stat: float | None = None
    cointegration_critical_value: float | None = None
    cointegration_window_bars: int = 0
    cointegration_sample_count: int = 0
    cointegration_last_check_index: int | None = None
    cointegration_reason: str = ""
    cointegration_block_open: bool = False

    last_model_update_index: int | None = None
    last_hedge_model_update_index: int | None = None
    next_model_update_index: int | None = None
    next_hedge_model_update_index: int | None = None
    last_model_update_skip_index: int | None = None
    last_hedge_model_update_skip_index: int | None = None
    # Kalman/RLS internal state
    kalman_P: Any = None
    kalman_theta: Any = None
    rls_P: Any = None
    rls_theta: Any = None

    reason: str = ""
    para: dict = field(default_factory=dict)


# ---- Signal ----

@dataclass
class SignalState:
    entry_z_upper: float = 2.0
    entry_z_lower: float = -2.0
    exit_z: float = 0.5
    entry_thresholds_ready: bool = False
    entry_rule_reason: str = ""
    last_entry_rule_update_index: int | None = None

    # z-score history for threshold calibration
    zscore_history: deque[float] = field(default_factory=deque)

    # two-stage
    armed_side: str | None = None
    armed_bar_index: int | None = None
    armed_zscore: float | None = None
    armed_trigger_z: float | None = None
    armed_entry_z: float | None = None

    last_open_reject_bar_index: int | None = None
    reversion_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalOutput:
    pair_id: str
    ts: int | None = None
    bar_index: int = 0
    ready: bool = False

    action: str = "none"            # "open" / "add" / "close" / "none"
    side: str | None = None         # "long_x" / "short_x"
    zscore: float | None = None
    entry_z_upper: float | None = None
    entry_z_lower: float | None = None
    exit_z: float | None = None
    signal_strength: float = 0.0
    entry_thresholds_ready: bool = False

    armed_side: str | None = None
    armed_bar_index: int | None = None
    armed_zscore: float | None = None
    armed_trigger_z: float | None = None
    armed_entry_z: float | None = None

    reason: str = ""
    para: dict = field(default_factory=dict)


# ---- Sizing ----

@dataclass
class SizingState:
    position_side: str | None = None
    position_id: str | None = None
    x_quantity: float = 0.0
    y_quantity: float = 0.0
    entry_count: int = 0
    last_entry_bar_index: int | None = None

    current_gross_notional: float = 0.0
    pair_cap_gross: float = 0.0
    entry_gross_cap: float = 0.0
    remaining_pair_gross: float = 0.0
    pair_full: bool = False
    add_cooldown_remaining_bars: int = 0
    min_hold_remaining_bars: int = 0
    last_schedule_reason: str = ""

    pending_state: dict[str, Any] = field(default_factory=dict)
    pending_fills: dict[str, Any] = field(default_factory=dict)

    size_multiplier: float = 1.0
    target_hedge_ratio: float | None = None


@dataclass
class PositionProtectionState:
    """Frozen model and accounting basis for one pair position."""

    active: bool = False
    pending_exit: bool = False
    side: str | None = None
    position_id: str | None = None
    entry_bar_index: int | None = None
    entry_ts: int | None = None
    entry_x_price: float | None = None
    entry_y_price: float | None = None
    entry_x_quantity: float = 0.0
    entry_y_quantity: float = 0.0
    entry_gross_notional: float = 0.0
    entry_fee: float = 0.0
    entry_slippage: float = 0.0
    funding_cost: float = 0.0
    alpha: float | None = None
    beta: float | None = None
    spread_mean: float | None = None
    spread_std: float | None = None
    exit_z: float | None = None
    stop_loss_x_price: float | None = None
    stop_loss_direction: str | None = None
    stop_loss_net_pnl: float | None = None
    stop_loss_reason: str | None = None
    take_profit_return: float | None = None
    pair_loss_stop_return: float | None = None
    pair_loss_stop_freeze_bars: int = 0
    pair_loss_stop_freeze_until_bar: int | None = None
    protection_rule: str | None = None
    freeze_rule: str | None = None
    freeze_bars: int = 0
    freeze_until_bar: int | None = None
    max_holding_bars: int | None = None
    max_holding_deadline_bar: int | None = None
    close_bar_index: int | None = None
    reopen_lock_pending: bool = False
    last_trigger: str | None = None
    last_exit_reason: str | None = None
    last_exit_class: str | None = None


# ---- Portfolio ----

@dataclass
class PortfolioState:
    ready: bool = False
    warmup_bars_seen: int = 0
    warmup_bars_required: int = 0

    pair_return_history: dict[str, list[float]] = field(default_factory=dict)
    covariance_matrix: Any = None

    # Diagnostics for currently occupied Pair capital.  These fields are no
    # longer a concurrency limit; Pair target capital and exchange margin are
    # the authoritative constraints.
    open_pair_count: int = 0
    open_pair_ids: list[str] = field(default_factory=list)

    portfolio_gross_exposure: float = 0.0
    portfolio_net_exposure: float = 0.0
    symbol_exposure_map: dict[str, float] = field(default_factory=dict)
    selected_pair_ids: list[str] = field(default_factory=list)
    last_allocation_reason: str = ""
    equity: float = 0.0
    available_balance: float = 0.0


# ---- Aggregate pair state ----

@dataclass
class PairRuntimeState:
    pair_id: str
    lifecycle_state: str = "unknown"       # "warmup" / "ready" / "blocked" / "closed"
    is_ready: bool = False
    last_ts: int | None = None
    last_bar_index: int = 0

    estimator_state: EstimatorState = field(default_factory=EstimatorState)
    signal_state: SignalState = field(default_factory=SignalState)
    sizing_state: SizingState = field(default_factory=SizingState)
    protection_state: PositionProtectionState = field(default_factory=PositionProtectionState)
    portfolio_state: PortfolioState = field(default_factory=PortfolioState)

    last_block_reason: str = ""


# ---- Target & allocation types ----

@dataclass
class RawPairTarget:
    pair_id: str
    ts: int | None = None
    bar_index: int = 0
    ready: bool = False
    side: str | None = None
    x_weight: float = 0.0
    y_weight: float = 0.0
    x_price: float = 0.0
    y_price: float = 0.0
    x_notional: float = 0.0
    y_notional: float = 0.0
    x_quantity: float = 0.0
    y_quantity: float = 0.0
    hedge_ratio: float = 1.0
    long_vol: float | None = None
    short_vol: float | None = None
    gross_notional: float = 0.0
    target_capital: float | None = None
    signal_strength: float = 0.0
    reason: str = ""
    para: dict = field(default_factory=dict)


@dataclass
class AllocatedPairTarget:
    pair_id: str
    selected: bool = False
    side: str | None = None           # "long_x" | "short_x" — passed from signal
    hedge_ratio: float | None = None
    scale_factor: float = 1.0
    final_x_notional: float = 0.0
    final_y_notional: float = 0.0
    final_x_quantity: float = 0.0
    final_y_quantity: float = 0.0
    portfolio_weight: float = 0.0
    allocation_reason: str = ""
    para: dict = field(default_factory=dict)


@dataclass
class PortfolioAllocation:
    ts: int | None = None
    bar_index: int = 0
    selected_pair_ids: list[str] = field(default_factory=list)
    pair_targets: dict[str, AllocatedPairTarget] = field(default_factory=dict)
    symbol_exposure_map: dict[str, float] = field(default_factory=dict)
    portfolio_gross_exposure: float = 0.0
    portfolio_net_exposure: float = 0.0
    constraint_report: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    para: dict = field(default_factory=dict)


# ---- Pipeline result (one pair) ----

@dataclass
class PairPipelineResult:
    pair_id: str
    estimator: EstimatorOutput = field(default_factory=lambda: EstimatorOutput(pair_id=""))
    signal: SignalOutput = field(default_factory=lambda: SignalOutput(pair_id=""))
    raw_target: RawPairTarget = field(default_factory=lambda: RawPairTarget(pair_id=""))
    allocated_target: AllocatedPairTarget = field(default_factory=lambda: AllocatedPairTarget(pair_id=""))
    orders: list[Any] = field(default_factory=list)
    ready: bool = False
    block_reason: str = ""
    para: dict = field(default_factory=dict)


# ---- Estimator registry ----

ESTIMATOR_REGISTRY: dict[str, type] = {}
SIGNAL_REGISTRY: dict[str, type] = {}
SIZING_REGISTRY: dict[str, type] = {}
PORTFOLIO_REGISTRY: dict[str, type] = {}


def register_estimator(name: str):
    def dec(cls):
        ESTIMATOR_REGISTRY[name] = cls
        return cls
    return dec


def register_signal(name: str):
    def dec(cls):
        SIGNAL_REGISTRY[name] = cls
        return cls
    return dec


def register_sizing(name: str):
    def dec(cls):
        SIZING_REGISTRY[name] = cls
        return cls
    return dec


def register_portfolio(name: str):
    def dec(cls):
        PORTFOLIO_REGISTRY[name] = cls
        return cls
    return dec
