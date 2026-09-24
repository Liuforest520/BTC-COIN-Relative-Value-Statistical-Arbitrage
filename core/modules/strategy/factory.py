from core.modules.strategy.multi_pair_strategy import MultiPairStrategy
from core.modules.strategy.config import (
    EstimatorConfig,
    ExecutionConfig,
    PairDefinition,
    PortfolioConfig,
    ProtectionConfig,
    RebalanceConfig,
    RebalanceCandidateQualityConfig,
    ProfitablePositionReplacementConfig,
    SignalConfig,
    SizingConfig,
)
from core.modules.models.pipeline_types import (
    ESTIMATOR_REGISTRY,
    SIGNAL_REGISTRY,
    SIZING_REGISTRY,
    PORTFOLIO_REGISTRY,
)


def build_strategy(setup_config, symbols, fee_rate: float = 0.0005, slippage_bps: float = 1.0):
    """Build the current multi-pair pipeline strategy."""
    if not setup_config.pairs:
        raise ValueError("multi-pair strategy requires at least one pair in setup_config.pairs")
    return _build_multi_pair_strategy(setup_config, symbols, fee_rate, slippage_bps)

def _resolve_registry(name: str, registry: dict, default_label: str):
    """Look up a registered class by name."""
    if name in registry:
        return registry[name]
    raise ValueError(
        f"unknown {default_label} '{name}', available: {list(registry.keys())}"
    )


def _warn_signal_method_conflicts(sig_cfg: dict) -> None:
    """Warn once per run when a config still carries the other variant's switch.

    The dedicated signal classes force their own flags, so a leftover
    ``two_stage_enabled`` / ``reversion_filter_enabled`` from an older config is
    ignored.  Only *explicitly present* keys are reported: the dataclass
    defaults must not produce a warning.
    """
    method = str((sig_cfg or {}).get("method", "")).strip().lower()
    if method == "zscore_reversion_two_stage":
        if sig_cfg.get("two_stage_enabled") is False or sig_cfg.get("reversion_filter_enabled") is True:
            from core.modules.logger import logger

            logger.warning(
                "signal.method {} ignores two_stage_enabled/reversion_filter_enabled "
                "from the config (two_stage is always on, MA filter always off)",
                method,
            )
    elif method == "zscore_reversion_ma":
        if sig_cfg.get("two_stage_enabled") is True or sig_cfg.get("reversion_filter_enabled") is False:
            from core.modules.logger import logger

            logger.warning(
                "signal.method {} ignores two_stage_enabled/reversion_filter_enabled "
                "from the config (MA filter is always on, two_stage always off)",
                method,
            )


def _build_multi_pair_strategy(setup_config, symbols, fee_rate=0.0005, slippage_bps=1.0):
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
            target_capital=(float(pdict["target_capital"]) if pdict.get("target_capital") is not None else None),
        ))

    pipeline = setup_config.pipeline or {}
    est_cfg = pipeline.get("estimator", {})
    sig_cfg = pipeline.get("signal", {})
    siz_cfg = pipeline.get("sizing", {})
    pf_cfg = pipeline.get("portfolio", {})
    exec_cfg = pipeline.get("execution", {})
    protection_cfg = pipeline.get("protection", {})
    setup_rebalance_cfg = dict(getattr(setup_config, "rebalance", {}) or {})
    pipeline_rebalance_cfg = dict(pipeline.get("rebalance", {}) or {})
    for source_name, source_cfg in (
        ("setup.rebalance", setup_rebalance_cfg),
        ("pipeline.rebalance", pipeline_rebalance_cfg),
    ):
        if "eviction_min_holding_bars" in source_cfg:
            raise ValueError(
                f"{source_name}.eviction_min_holding_bars is no longer supported; "
                "use rebalance.eviction_min_holding_model_lookback_multiplier "
                "with a model-lookback multiplier"
            )
    rebalance_cfg = setup_rebalance_cfg or pipeline_rebalance_cfg

    # Pair-target-capital is intentionally strict. Falling back to sizing
    # notional silently creates a Pair with a different capital allocation.
    portfolio_method = str(pf_cfg.get("method", ""))
    if portfolio_method == "pair_target_capital":
        missing = [
            p.pair_id for p in pair_defs
            if p.enabled and (p.target_capital is None or p.target_capital <= 0)
        ]
        if missing:
            raise ValueError(
                "pair_target_capital requires a positive pairs[].target_capital "
                f"for every enabled Pair; missing/invalid: {missing}"
            )

    # Filter to only known fields to avoid TypeError from extra legacy keys
    def _filter_kwargs(dc_class, raw_dict, label="config"):
        import dataclasses
        raw_dict = dict(raw_dict or {})
        # Legacy estimator configs used ``warmup_bars``. In the current
        # pipeline warm-up is derived from ``model_lookback_bars``; preserve
        # the old meaning without forwarding a dead constructor argument or
        # emitting a noisy dropped-key warning.
        if dc_class is EstimatorConfig and "warmup_bars" in raw_dict:
            legacy_warmup = raw_dict.pop("warmup_bars")
            if "model_lookback_bars" not in raw_dict and legacy_warmup is not None:
                raw_dict["model_lookback_bars"] = legacy_warmup
            elif (
                legacy_warmup is not None
                and "model_lookback_bars" in raw_dict
                and int(legacy_warmup) != int(raw_dict["model_lookback_bars"])
            ):
                from core.modules.logger import logger
                logger.warning(
                    "{}: legacy warmup_bars={} ignored because model_lookback_bars={} is set",
                    label, legacy_warmup, raw_dict["model_lookback_bars"],
                )
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
    _warn_signal_method_conflicts(sig_cfg)
    signal_cfg = SignalConfig(**_filter_kwargs(SignalConfig, sig_cfg, "signal")) if sig_cfg else SignalConfig()
    sizing_cfg = SizingConfig(**_filter_kwargs(SizingConfig, siz_cfg, "sizing")) if siz_cfg else SizingConfig()
    portfolio_cfg = PortfolioConfig(**_filter_kwargs(PortfolioConfig, pf_cfg, "portfolio")) if pf_cfg else PortfolioConfig()
    execution_cfg = ExecutionConfig(**_filter_kwargs(ExecutionConfig, exec_cfg, "execution")) if exec_cfg else ExecutionConfig()
    protection_cfg_obj = ProtectionConfig(**_filter_kwargs(ProtectionConfig, protection_cfg, "protection")) if protection_cfg else ProtectionConfig()
    replacement_raw = rebalance_cfg.get("profitable_position_replacement", {}) or {}
    candidate_quality_raw = rebalance_cfg.get("candidate_quality", {}) or {}
    rebalance_cfg = dict(rebalance_cfg)
    rebalance_cfg.pop("profitable_position_replacement", None)
    rebalance_cfg.pop("candidate_quality", None)
    replacement_raw = _filter_kwargs(
        ProfitablePositionReplacementConfig, replacement_raw,
        "rebalance.profitable_position_replacement",
    )
    candidate_quality_raw = _filter_kwargs(
        RebalanceCandidateQualityConfig, candidate_quality_raw,
        "rebalance.candidate_quality",
    )
    if "minimum_entry_capital_ratio" in rebalance_cfg and "minimum_entry_capital_ratio" in pf_cfg:
        if float(rebalance_cfg["minimum_entry_capital_ratio"]) != float(pf_cfg["minimum_entry_capital_ratio"]):
            raise ValueError("portfolio.minimum_entry_capital_ratio and rebalance.minimum_entry_capital_ratio must match")
    elif "minimum_entry_capital_ratio" in pf_cfg:
        rebalance_cfg["minimum_entry_capital_ratio"] = pf_cfg["minimum_entry_capital_ratio"]
    rebalance_cfg_obj = RebalanceConfig(
        candidate_quality=RebalanceCandidateQualityConfig(**candidate_quality_raw),
        profitable_position_replacement=ProfitablePositionReplacementConfig(**replacement_raw),
        **_filter_kwargs(RebalanceConfig, rebalance_cfg, "rebalance"),
    )
    if (
        protection_cfg_obj.enabled
        or protection_cfg_obj.max_holding_time_enabled
        or rebalance_cfg_obj.enabled
    ) and (
        estimator_cfg.position_update_policy != "freeze"
        or estimator_cfg.regression_method not in {"price", "log_price"}
    ):
        raise ValueError(
            "pipeline protection/rebalance is supported only for freeze Price/Log-Price models"
        )
    if rebalance_cfg_obj.profitable_position_replacement.enabled:
        from core.modules.logger import logger
        logger.warning(
            "rebalance.profitable_position_replacement.enabled is ignored: "
            "the current rebalance policy replaces losing positions only"
        )

    # Import estimators / signals / sizing / portfolio to trigger registration
    from core.modules.estimator import (  # noqa: F401
        DOLSEstimator, EWLSEstimator, HuberEstimator,
        KalmanEstimator, PeriodicOLSEstimator,
        RLSEstimator, RollingOLSEstimator, TLSEstimator, WinsorizedOLSEstimator,
    )
    from core.modules.signals import (  # noqa: F401
        MAReversionSignal,
        SimpleZScoreSignal,
        TwoStageReversionSignal,
        ZScoreReversionSignal,
        ZScoreSignal,
    )
    from core.modules.sizing import (  # noqa: F401
        BetaNeutralSizing, FixedNotionalSizing, VolatilityNeutralSizing,
    )
    from core.modules.portfolio import (  # noqa: F401
        ConstrainedQPAllocator, EquitySlotAllocator, EqualWeightAllocator, MaxSharpeAllocator, PairTargetCapitalAllocator,
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
        protection_cfg=protection_cfg_obj,
        rebalance_cfg=rebalance_cfg_obj,
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
    )
