from core.modules.strategy.multi_pair_strategy import MultiPairStrategy
from core.modules.strategy.config import (
    EstimatorConfig,
    ExecutionConfig,
    PairDefinition,
    PortfolioConfig,
    SignalConfig,
    SizingConfig,
)
from core.modules.models.pipeline_types import (
    ESTIMATOR_REGISTRY,
    SIGNAL_REGISTRY,
    SIZING_REGISTRY,
    PORTFOLIO_REGISTRY,
)


def build_strategy(setup_config, symbols):
    """Build the current multi-pair pipeline strategy."""
    if not setup_config.pairs:
        raise ValueError("multi-pair strategy requires at least one pair in setup_config.pairs")
    return _build_multi_pair_strategy(setup_config, symbols)

def _resolve_registry(name: str, registry: dict, default_label: str):
    """Look up a registered class by name."""
    if name in registry:
        return registry[name]
    raise ValueError(
        f"unknown {default_label} '{name}', available: {list(registry.keys())}"
    )


def _build_multi_pair_strategy(setup_config, symbols):
    """Build a MultiPairStrategy from setup config with pairs + pipeline specs."""
    # Parse pair definitions
    pair_defs = []
    for pdict in setup_config.pairs:
        x_sym = pdict.get("x_symbol", pdict.get("long_symbol", ""))
        y_sym = pdict.get("y_symbol", pdict.get("short_symbol", ""))
        if x_sym not in symbols or y_sym not in symbols:
            raise ValueError(f"pair {pdict.get('pair_id','?')}: symbols not in data config")

        # Pull exchange from symbol config if available
        x_info = symbols.get(x_sym, {})
        y_info = symbols.get(y_sym, {})
        pair_defs.append(PairDefinition(
            pair_id=pdict.get("pair_id", f"{x_sym}_{y_sym}"),
            x_exchange=pdict.get("x_exchange", x_info.get("exchange", "binance")),
            x_symbol=x_sym,
            y_exchange=pdict.get("y_exchange", y_info.get("exchange", "binance")),
            y_symbol=y_sym,
            enabled=pdict.get("enabled", True),
            tags=list(pdict.get("tags", [])),
            model_lookback_bars_override=pdict.get("model_lookback_bars_override"),
            notes=pdict.get("notes", ""),
        ))

    pipeline = setup_config.pipeline or {}
    est_cfg = pipeline.get("estimator", {})
    sig_cfg = pipeline.get("signal", {})
    siz_cfg = pipeline.get("sizing", {})
    pf_cfg = pipeline.get("portfolio", {})
    exec_cfg = pipeline.get("execution", {})

    # Filter to only known fields to avoid TypeError from extra legacy keys
    def _filter_kwargs(dc_class, raw_dict, label="config"):
        import dataclasses
        valid = {f.name for f in dataclasses.fields(dc_class)}
        filtered = {k: v for k, v in raw_dict.items() if k in valid}
        dropped = set(raw_dict) - set(filtered)
        if dropped:
            from core.modules.logger import logger
            logger.warning(
                "{}: dropped unknown keys {} (expected: {})",
                label, sorted(dropped), sorted(valid),
            )
        return filtered

    estimator_cfg = EstimatorConfig(**_filter_kwargs(EstimatorConfig, est_cfg, "estimator")) if est_cfg else EstimatorConfig()
    signal_cfg = SignalConfig(**_filter_kwargs(SignalConfig, sig_cfg, "signal")) if sig_cfg else SignalConfig()
    sizing_cfg = SizingConfig(**_filter_kwargs(SizingConfig, siz_cfg, "sizing")) if siz_cfg else SizingConfig()
    portfolio_cfg = PortfolioConfig(**_filter_kwargs(PortfolioConfig, pf_cfg, "portfolio")) if pf_cfg else PortfolioConfig()
    execution_cfg = ExecutionConfig(**_filter_kwargs(ExecutionConfig, exec_cfg, "execution")) if exec_cfg else ExecutionConfig()

    # Import estimators / signals / sizing / portfolio to trigger registration
    from core.modules.estimator import (  # noqa: F401
        AgeWeightedWLSEstimator, DOLSEstimator, EWLSEstimator, HuberEstimator,
        KalmanEstimator, PeriodicOLSEstimator, ResidualWeightedWLSEstimator,
        RLSEstimator, RollingOLSEstimator, TLSEstimator, WinsorizedOLSEstimator,
    )
    from core.modules.signals import ZScoreReversionSignal, ZScoreSignal  # noqa: F401
    from core.modules.sizing import (  # noqa: F401
        BetaNeutralSizing, FixedNotionalSizing, VolatilityNeutralSizing,
    )
    from core.modules.portfolio import (  # noqa: F401
        ConstrainedQPAllocator, EqualWeightAllocator, MaxSharpeAllocator,
        MinVarianceAllocator, RiskParityAllocator,
    )

    estimator_ctor = _resolve_registry(estimator_cfg.method, ESTIMATOR_REGISTRY, "estimator")
    signal_ctor = _resolve_registry(signal_cfg.method, SIGNAL_REGISTRY, "signal") if sig_cfg else None
    sizing_ctor = _resolve_registry(sizing_cfg.method, SIZING_REGISTRY, "sizing") if siz_cfg else None
    portfolio_ctor = _resolve_registry(portfolio_cfg.method, PORTFOLIO_REGISTRY, "portfolio") if pf_cfg else None

    return MultiPairStrategy(
        pairs=pair_defs,
        estimator_ctor=estimator_ctor,
        estimator_cfg=estimator_cfg,
        signal_ctor=signal_ctor,
        signal_cfg=signal_cfg,
        sizing_ctor=sizing_ctor,
        sizing_cfg=sizing_cfg,
        portfolio_ctor=portfolio_ctor,
        portfolio_cfg=portfolio_cfg,
        execution_cfg=execution_cfg,
    )
