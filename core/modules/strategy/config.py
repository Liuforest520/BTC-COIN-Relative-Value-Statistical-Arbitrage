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
    market: str = "legacy"
    parameter_grid: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)

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
    """Fixed-z MA-reversion signal configuration."""

    method: str = "zscore_reversion_ma"
    entry_z: float = 2.0
    exit_z: float = 0.5
    reversion_ma_lookback_bars: int = 60
    reversion_ma_short_lookback_bars: int = 15
    reversion_min_samples: int | None = None


@dataclass
class SizingConfig:
    """Beta-neutral sizing with fixed pair target capital."""

    method: str = "beta_neutral"
    hedge_beta_lookback_bars: int = 1440
    hedge_beta_min_samples: int = 30


@dataclass
class PortfolioConfig:
    method: str = "pair_target_capital"

    def __post_init__(self):
        method = str(self.method or "").strip().lower()
        if method != "pair_target_capital":
            raise ValueError("portfolio.method must be 'pair_target_capital'")
        self.method = method


@dataclass
class ProfitablePositionReplacementConfig:
    """Rules for replacing profitable, sufficiently converged positions."""

    enabled: bool = True
    min_net_return: float = 0.0
    min_convergence_ratio: float = 0.80

    def __post_init__(self):
        net_return = float(self.min_net_return)
        convergence = float(self.min_convergence_ratio)
        if not isfinite(net_return) or net_return < 0.0:
            raise ValueError(
                "rebalance.profitable_position_replacement.min_net_return must be finite and non-negative"
            )
        if not isfinite(convergence) or convergence < 0.0 or convergence > 1.0:
            raise ValueError(
                "rebalance.profitable_position_replacement.min_convergence_ratio must be in [0, 1]"
            )
        self.min_net_return = net_return
        self.min_convergence_ratio = convergence

@dataclass
class RebalanceCandidateQualityConfig:
    """Hard gates for a new Pair that would require evicting old positions."""

    enabled: bool = True
    min_theoretical_zero_return_x_move: float = 0.20
    max_adf_pvalue: float = 0.4
    min_expected_net_return: float = 0.0

    def __post_init__(self):
        move = float(self.min_theoretical_zero_return_x_move)
        pvalue = float(self.max_adf_pvalue)
        expected_return = float(self.min_expected_net_return)
        if not isfinite(move) or move < 0:
            raise ValueError(
                "rebalance.candidate_quality.min_theoretical_zero_return_x_move "
                "must be non-negative"
            )
        if not isfinite(pvalue) or pvalue < 0 or pvalue > 1:
            raise ValueError(
                "rebalance.candidate_quality.max_adf_pvalue must be in [0, 1]"
            )
        if not isfinite(expected_return):
            raise ValueError(
                "rebalance.candidate_quality.min_expected_net_return must be finite"
            )
        self.min_theoretical_zero_return_x_move = move
        self.max_adf_pvalue = pvalue
        self.min_expected_net_return = expected_return


@dataclass
class RebalanceConfig:
    enabled: bool = False
    candidate_quality: RebalanceCandidateQualityConfig = field(
        default_factory=RebalanceCandidateQualityConfig
    )
    profitable_position_replacement: ProfitablePositionReplacementConfig = field(
        default_factory=ProfitablePositionReplacementConfig
    )

@dataclass
class ExecutionConfig:
    order_type: str = "market"


@dataclass
class ProtectionConfig:
    """Pair loss stop protection only."""

    enabled: bool = False
    pair_loss_stop_enabled: bool = False
    pair_loss_stop_return: float = 0.05
    pair_loss_stop_freeze_bars: int = 0
    pair_loss_stop_freeze_model_lookback_multiplier: float | None = None
    wait_for_model_update_after_non_z_exit: bool | None = None

    def __post_init__(self):
        loss_value = float(self.pair_loss_stop_return)
        if not isfinite(loss_value) or loss_value <= 0.0:
            raise ValueError("protection.pair_loss_stop_return must be finite and positive")
        freeze_bars = int(self.pair_loss_stop_freeze_bars)
        if freeze_bars < 0:
            raise ValueError("protection.pair_loss_stop_freeze_bars must be non-negative")
        self.pair_loss_stop_freeze_bars = freeze_bars
        if self.pair_loss_stop_freeze_model_lookback_multiplier is not None:
            multiplier = float(self.pair_loss_stop_freeze_model_lookback_multiplier)
            if not isfinite(multiplier) or multiplier <= 0.0:
                raise ValueError("protection.pair_loss_stop_freeze_model_lookback_multiplier must be finite and positive")
            self.pair_loss_stop_freeze_model_lookback_multiplier = multiplier
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
