"""
Configuration dataclasses for the pipeline stages.
Each stage has its own config so a single setup can pick estimator=A, signal=B, etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite


@dataclass
class PairDefinition:
    pair_id: str
    x_exchange: str = "binance"
    x_symbol: str = ""
    y_exchange: str = "binance"
    y_symbol: str = ""
    enabled: bool = True
    tags: list[str] = field(default_factory=list)
    model_lookback_bars_override: int | None = None
    notes: str = ""
    # Target gross capital for the complete Pair (both legs together).
    # ``None`` keeps legacy sizing behavior for old configurations.
    target_capital: float | None = None

    def __post_init__(self):
        if self.target_capital is None:
            return
        value = float(self.target_capital)
        if not isfinite(value) or value < 0.0:
            raise ValueError("pair.target_capital must be a finite non-negative number")
        self.target_capital = value


@dataclass
class EstimatorConfig:
    method: str = "rolling_ols"               # Select a registered estimator.
    regression_method: str = "log_price"      # "log_price" | "price"
    model_lookback_bars: int = 10080       # fit window and initial warmup length
    model_update_interval_bars: int = 240
    # Source-minute model window.  1m preserves all legacy configs.
    model_timeframe: str = "1m"
    position_update_policy: str = "freeze"     # "freeze" | "update"
    # Residual ADF, run after every successful scheduled fit.
    residual_adf_max_pvalue: float = 0.40
    residual_adf_maxlag: int = 1
    residual_adf_regression: str = "c"
    # DOLS / EWLS / Huber
    dols_lead_lag_order: int = 1
    ewls_half_life_bars: float = 1440.0
    huber_delta: float = 1.345
    huber_iterations: int = 5
    # Winsorized OLS
    winsor_lower_quantile: float = 0.005
    winsor_upper_quantile: float = 0.995
    winsor_quantile_method: str = "linear"
    # Filters
    mean_reversion_filter_enabled: bool = True
    beta_stability_enabled: bool = False
    beta_stability_lookback_bars: int = 1440
    beta_stability_max_cv: float = 0.10
    beta_stability_min_samples: int = 240
    # RLS
    rls_forgetting_factor: float = 0.995
    # Kalman
    kalman_delta: float = 1e-5
    kalman_v0: float = 0.1
    kalman_q_alpha: float | None = None
    kalman_q_beta: float | None = None
    kalman_r_multiplier: float = 1.0
    spread_std_floor: float | None = None
    zscore_cap: float | None = None


@dataclass
class SignalConfig:
    # Entry family -- pick exactly one:
    #   "simple_zscore"               hit +/-entry_z and enter immediately
    #   "zscore_reversion_two_stage"  arm at trigger, enter on the pullback
    #   "zscore_reversion_ma"         require the z-score MA to turn first
    # Legacy names: "zscore" (== simple_zscore) and "zscore_reversion"
    # (combined; depends on both switches below).
    method: str = "simple_zscore"
    entry_z: float = 2.0
    # >0: symmetric exit band; 0: cross zero; <0: reverse by abs(exit_z).
    exit_z: float = 0.5
    entry_rule_method: str = "fixed_z"        # "fixed_z" | "percentile" | "adaptive"
    entry_rule_lookback_bars: int = 10080
    entry_rule_update_interval_bars: int = 60
    entry_rule_upper_percentile: float = 97.5
    entry_rule_lower_percentile: float = 2.5
    entry_rule_min_samples: int | None = None
    entry_rule_min_abs_entry_z: float | None = None
    entry_rule_max_abs_entry_z: float | None = None
    # Two-stage
    two_stage_enabled: bool = False
    two_stage_trigger_z: float | None = None
    two_stage_entry_z: float | None = None
    two_stage_entry_window_z: float | None = None
    two_stage_levels: list | None = None
    # Reversion filter
    reversion_filter_enabled: bool = True
    reversion_ma_lookback_bars: int = 60
    reversion_ma_short_lookback_bars: int | None = None
    reversion_min_samples: int | None = None
    # Two-stage expiry
    two_stage_max_wait_bars: int = 0
    # Optional entry filters.
    pair_quality_filter_enabled: bool = False
    pair_quality_min_samples: int = 30
    pair_quality_min_corr: float = 0.0
    cost_filter_enabled: bool = False
    cost_fee_rate: float = 0.0005
    cost_slippage_bps: float = 1.0
    cost_buffer_multiplier: float = 1.0


@dataclass
class SizingConfig:
    method: str = "beta_neutral"
    notional: float = 10000.0
    equity: float = 100000.0
    gross_exposure_ratio: float = 0.2
    max_gross_exposure_ratio: float = 1.0
    min_notional: float = 0.0
    # Add-on
    max_add_times: int = 0
    add_interval_bars: int = 1
    add_interval_mode: str = "fixed"
    half_life_add_interval_multiplier: float = 1.0
    min_add_interval_bars: int | None = None
    max_add_interval_bars: int | None = None
    require_half_life_for_add: bool = False
    add_z_step: float = 0.0
    add_gross_exposure_ratio: float | None = None
    add_size_multiplier: float | None = None
    require_add_reversion_vs_last_entry: bool = False
    add_reversion_buffer: float = 0.0
    # Beta-neutral hedge beta computed only when sizing is triggered.
    hedge_beta_lookback_bars: int = 1440
    hedge_beta_min_samples: int = 30


@dataclass
class PortfolioConfig:
    method: str = "pair_target_capital"
    max_entries_per_pair: int = 1
    add_cooldown_bars: int = 1
    min_hold_bars: int = 0
    allocation_mode: str = "equal_cap"   # "equal_cap" | "score_weighted"
    min_signal_score: float = 0.0
    minimum_entry_capital_ratio: float = 0.5

    def __post_init__(self):
        ratio = float(self.minimum_entry_capital_ratio)
        if not isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
            raise ValueError("portfolio.minimum_entry_capital_ratio must be in (0, 1]")
        self.minimum_entry_capital_ratio = ratio


@dataclass
class ProfitablePositionReplacementConfig:
    enabled: bool = False
    min_theoretical_zero_return_x_move: float = 0.20
    max_adf_pvalue: float = 0.4

    def __post_init__(self):
        move = float(self.min_theoretical_zero_return_x_move)
        pvalue = float(self.max_adf_pvalue)
        if not isfinite(move) or move < 0:
            raise ValueError("rebalance.profitable_position_replacement.min_theoretical_zero_return_x_move must be non-negative")
        if not isfinite(pvalue) or pvalue < 0 or pvalue > 1:
            raise ValueError("rebalance.profitable_position_replacement.max_adf_pvalue must be in [0, 1]")
        self.min_theoretical_zero_return_x_move = move
        self.max_adf_pvalue = pvalue


@dataclass
class RebalanceConfig:
    enabled: bool = False
    minimum_entry_capital_ratio: float = 0.5
    closed_pair_freeze_model_lookback_multiplier: float = 0.5
    eviction_min_holding_bars: int = 0
    profitable_position_replacement: ProfitablePositionReplacementConfig = field(
        default_factory=ProfitablePositionReplacementConfig
    )

    def __post_init__(self):
        ratio = float(self.minimum_entry_capital_ratio)
        multiplier = float(self.closed_pair_freeze_model_lookback_multiplier)
        if not isfinite(ratio) or ratio <= 0 or ratio > 1:
            raise ValueError("rebalance.minimum_entry_capital_ratio must be in (0, 1]")
        if not isfinite(multiplier) or multiplier < 0:
            raise ValueError("rebalance.closed_pair_freeze_model_lookback_multiplier must be non-negative")
        self.minimum_entry_capital_ratio = ratio
        self.closed_pair_freeze_model_lookback_multiplier = multiplier
        self.eviction_min_holding_bars = int(self.eviction_min_holding_bars)
        if self.eviction_min_holding_bars < 0:
            raise ValueError("rebalance.eviction_min_holding_bars must be non-negative")


@dataclass
class ExecutionConfig:
    pending_timeout_bars: int = 0
    close_reject_cooldown_bars: int = 0
    order_type: str = "market"
    cancel_stale_orders: bool = False
    retry_on_reject: bool = False


@dataclass
class ProtectionConfig:
    """Optional pair-level protective exits."""

    enabled: bool = False
    stop_loss_enabled: bool = True
    stop_loss_freeze_bars: int = 0
    pair_loss_stop_enabled: bool = False
    pair_loss_stop_return: float = 0.05
    pair_loss_stop_freeze_bars: int = 0
    # Optional model-relative freeze. When set, effective freeze bars are
    # ceil(model lookback bars * multiplier) for each Pair.
    pair_loss_stop_freeze_model_lookback_multiplier: float | None = None
    take_profit_enabled: bool = True
    # Take profit is measured as the pair net return on the entry gross
    # notional (both legs), after entry fees, funding paid and the estimated
    # close cost.  It overrides the z-score exit whenever it triggers.
    take_profit_return: float = 0.03
    take_profit_freeze_bars: int = 0
    max_holding_time_enabled: bool = False
    max_holding_time_model_lookback_multiplier: float = 2.0
    max_holding_time_freeze_bars: int = 0
    # Only stop-loss (and max-holding) exits wait for the next model refit; a
    # take-profit exit may reopen immediately.
    wait_for_model_update_after_non_z_exit: bool | None = None

    def __post_init__(self):
        value = float(self.take_profit_return)
        if not isfinite(value) or value < 0.0:
            raise ValueError("protection.take_profit_return must be a finite non-negative number")
        self.take_profit_return = value
        loss_value = float(self.pair_loss_stop_return)
        if not isfinite(loss_value) or loss_value <= 0.0:
            raise ValueError("protection.pair_loss_stop_return must be a finite positive number")
        self.pair_loss_stop_return = loss_value
        freeze_bars = int(self.pair_loss_stop_freeze_bars)
        if freeze_bars < 0:
            raise ValueError("protection.pair_loss_stop_freeze_bars must be non-negative")
        self.pair_loss_stop_freeze_bars = freeze_bars
        if self.pair_loss_stop_freeze_model_lookback_multiplier is not None:
            freeze_multiplier = float(self.pair_loss_stop_freeze_model_lookback_multiplier)
            if not isfinite(freeze_multiplier) or freeze_multiplier <= 0.0:
                raise ValueError("protection.pair_loss_stop_freeze_model_lookback_multiplier must be finite and positive")
            self.pair_loss_stop_freeze_model_lookback_multiplier = freeze_multiplier
        for name in ("stop_loss_freeze_bars", "take_profit_freeze_bars", "max_holding_time_freeze_bars"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"protection.{name} must be non-negative")
            setattr(self, name, value)
        multiplier = float(self.max_holding_time_model_lookback_multiplier)
        if not isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("protection.max_holding_time_model_lookback_multiplier must be a finite positive number")
        self.max_holding_time_model_lookback_multiplier = multiplier
        if self.wait_for_model_update_after_non_z_exit is not None:
            self.wait_for_model_update_after_non_z_exit = bool(self.wait_for_model_update_after_non_z_exit)


@dataclass
class PipelineStageConfigs:
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    protection: ProtectionConfig = field(default_factory=ProtectionConfig)
    rebalance: RebalanceConfig = field(default_factory=RebalanceConfig)
