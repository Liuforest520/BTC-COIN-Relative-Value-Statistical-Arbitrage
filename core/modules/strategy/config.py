"""
Configuration dataclasses for the pipeline stages.
Each stage has its own config so a single setup can pick estimator=A, signal=B, etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass
class EstimatorConfig:
    method: str = "rolling_ols"               # Select a registered estimator.
    regression_method: str = "log_price"      # "log_price" | "log_return" | "price"
    return_interval_bars: int = 1              # Return fit, residual, ADF, and live Z-score horizon.
    model_lookback_bars: int = 10080       # fit window and initial warmup length
    model_update_interval_bars: int = 240
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
    # Age-weighted WLS / Winsorized OLS / residual-weighted WLS
    age_wls_profile: str = "medium"
    age_wls_weights: list[float] | None = None
    age_bucket_end_bars: list[int] | None = None
    # Legacy proportional boundary input; new configs should use absolute bars.
    age_bucket_end_fractions: list[float] | None = None
    age_bucket_weights: list[float] | None = None
    winsor_lower_quantile: float = 0.005
    winsor_upper_quantile: float = 0.995
    winsor_quantile_method: str = "linear"
    residual_wls_profile: str = "medium"
    short_model_lookback_bars: int | None = None
    shock_score_boundaries: list[float] | None = None
    shock_weights: list[float] | None = None
    # Legacy alias for short_model_lookback_bars.
    residual_wls_scale_lookback_bars: int | None = None
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
    method: str = "zscore"          # "zscore" | "zscore_reversion"
    entry_z: float = 2.0
    exit_z: float = 0.5
    # standard | sum_zscore | cumulative_return
    position_exit_zscore_method: str = "standard"
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
    method: str = "equal_weight"
    max_open_pairs: int = 10
    max_entries_per_pair: int = 1
    add_cooldown_bars: int = 1
    min_hold_bars: int = 0
    allocation_mode: str = "equal_cap"   # "equal_cap" | "score_weighted"
    min_signal_score: float = 0.0
    max_gross_exposure_ratio: float = 1.0
    max_net_exposure_ratio: float = 0.2
    max_symbol_exposure_ratio: float = 0.5
    target_net_beta: float = 0.0
    risk_aversion: float = 1.0
    covariance_method: str = "sample"
    covariance_lookback_bars: int = 1440
    covariance_update_interval_bars: int = 60
    turnover_penalty: float = 0.0
    min_pair_weight: float = 0.01


@dataclass
class ExecutionConfig:
    pending_timeout_bars: int = 0
    close_reject_cooldown_bars: int = 0
    order_type: str = "market"
    cancel_stale_orders: bool = False
    retry_on_reject: bool = False


@dataclass
class PipelineStageConfigs:
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
