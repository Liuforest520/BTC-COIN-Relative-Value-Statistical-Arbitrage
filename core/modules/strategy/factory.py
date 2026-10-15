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


def _filter_kwargs(dc_class, raw_dict, label="config"):
    import dataclasses
    raw_dict = dict(raw_dict or {})
    if dc_class is EstimatorConfig and "warmup_bars" in raw_dict:
        legacy_warmup = raw_dict.pop("warmup_bars")
        if "model_lookback_bars" not in raw_dict and legacy_warmup is not None:
            raw_dict["model_lookback_bars"] = legacy_warmup
        elif legacy_warmup is not None and "model_lookback_bars" in raw_dict and int(legacy_warmup) != int(raw_dict["model_lookback_bars"]):
            from core.modules.logger import logger
            logger.warning("{}: legacy warmup_bars={} ignored because model_lookback_bars={} is set", label, legacy_warmup, raw_dict["model_lookback_bars"])
    valid = {f.name for f in dataclasses.fields(dc_class)}
    filtered = {k: v for k, v in raw_dict.items() if k in valid}
    dropped = set(raw_dict) - set(filtered)
    if dropped:
        from core.modules.logger import logger
        logger.warning("{}: dropped unknown keys {} (expected: {})", label, sorted(dropped), sorted(valid))
    return filtered


def _build_market_components(pipeline, setup_rebalance, market):
    """Build typed stage components for one market block."""
    pipeline = pipeline or {}
    est_cfg = pipeline.get("estimator", {}) or {}
    sig_cfg = pipeline.get("signal", {}) or {}
    siz_cfg = pipeline.get("sizing", {}) or {}
    pf_cfg = pipeline.get("portfolio", {}) or {}
    exec_cfg = pipeline.get("execution", {}) or {}
    protection_cfg = pipeline.get("protection", {}) or {}
    setup_rebalance_cfg = dict(setup_rebalance or {})
    pipeline_rebalance_cfg = dict(pipeline.get("rebalance", {}) or {})
    for source_name, source_cfg in ((f"{market}.setup.rebalance", setup_rebalance_cfg), (f"{market}.pipeline.rebalance", pipeline_rebalance_cfg)):
        legacy_keys = {
            "closed_pair_freeze_model_lookback_multiplier",
            "eviction_min_holding_model_lookback_multiplier",
            "eviction_min_holding_bars",
            "minimum_entry_capital_ratio",
        }
        found = sorted(legacy_keys.intersection(source_cfg))
        if found:
            raise ValueError(f"{source_name} contains removed rebalance key(s) {found}; rebalance replacement no longer freezes or enforces a minimum holding period")
        replacement = source_cfg.get("profitable_position_replacement", {}) or {}
        if isinstance(replacement, dict):
            found = sorted({"min_theoretical_zero_return_x_move", "max_adf_pvalue"}.intersection(replacement))
            if found:
                raise ValueError(f"{source_name}.profitable_position_replacement contains removed key(s) {found}; use min_net_return and min_convergence_ratio")
    rebalance_cfg = setup_rebalance_cfg or pipeline_rebalance_cfg
    if "minimum_entry_capital_ratio" in pf_cfg:
        raise ValueError(
            f"{market}.portfolio contains removed key 'minimum_entry_capital_ratio'; "
            "a Pair must be fully funded to its target_capital"
        )
    estimator_cfg = EstimatorConfig(**_filter_kwargs(EstimatorConfig, est_cfg, f"{market}.estimator"))
    signal_cfg = SignalConfig(**_filter_kwargs(SignalConfig, sig_cfg, f"{market}.signal")) if sig_cfg else SignalConfig()
    sizing_cfg = SizingConfig(**_filter_kwargs(SizingConfig, siz_cfg, f"{market}.sizing")) if siz_cfg else SizingConfig()
    portfolio_cfg = PortfolioConfig(**_filter_kwargs(PortfolioConfig, pf_cfg, f"{market}.portfolio")) if pf_cfg else PortfolioConfig()
    execution_cfg = ExecutionConfig(**_filter_kwargs(ExecutionConfig, exec_cfg, f"{market}.execution")) if exec_cfg else ExecutionConfig()
    protection_cfg_obj = ProtectionConfig(**_filter_kwargs(ProtectionConfig, protection_cfg, f"{market}.protection")) if protection_cfg else ProtectionConfig()
    replacement_raw = rebalance_cfg.get("profitable_position_replacement", {}) or {}
    candidate_quality_raw = rebalance_cfg.get("candidate_quality", {}) or {}
    clean_rebalance = dict(rebalance_cfg)
    clean_rebalance.pop("profitable_position_replacement", None)
    clean_rebalance.pop("candidate_quality", None)
    replacement_raw = _filter_kwargs(ProfitablePositionReplacementConfig, replacement_raw, f"{market}.rebalance.profitable_position_replacement")
    candidate_quality_raw = _filter_kwargs(RebalanceCandidateQualityConfig, candidate_quality_raw, f"{market}.rebalance.candidate_quality")
    rebalance_cfg_obj = RebalanceConfig(
        candidate_quality=RebalanceCandidateQualityConfig(**candidate_quality_raw),
        profitable_position_replacement=ProfitablePositionReplacementConfig(**replacement_raw),
        **_filter_kwargs(RebalanceConfig, clean_rebalance, f"{market}.rebalance"),
    )
    if (protection_cfg_obj.enabled or rebalance_cfg_obj.enabled) and (estimator_cfg.position_update_policy != "freeze" or estimator_cfg.regression_method not in {"price", "log_price"}):
        raise ValueError(f"{market}: pipeline protection/rebalance is supported only for freeze Price/Log-Price models")
    estimator_ctor = _resolve_registry(estimator_cfg.method, ESTIMATOR_REGISTRY, "estimator")
    signal_ctor = _resolve_registry(signal_cfg.method, SIGNAL_REGISTRY, "signal") if sig_cfg else None
    sizing_ctor = _resolve_registry(sizing_cfg.method, SIZING_REGISTRY, "sizing") if siz_cfg else None
    portfolio_ctor = _resolve_registry(portfolio_cfg.method, PORTFOLIO_REGISTRY, "portfolio") if pf_cfg else None
    return {
        "estimator_ctor": estimator_ctor, "estimator_cfg": estimator_cfg,
        "signal_ctor": signal_ctor, "signal_cfg": signal_cfg,
        "sizing_ctor": sizing_ctor, "sizing_cfg": sizing_cfg,
        "portfolio_ctor": portfolio_ctor, "portfolio_cfg": portfolio_cfg,
        "execution_cfg": execution_cfg, "protection_cfg": protection_cfg_obj,
        "rebalance_cfg": rebalance_cfg_obj,
    }


def _override_components(components, overrides):
    """Create pair-local typed configs without mutating the market defaults."""
    if not overrides:
        return components
    import dataclasses
    result = dict(components)
    for stage, key in (("estimator", "estimator_cfg"), ("signal", "signal_cfg"), ("sizing", "sizing_cfg"), ("portfolio", "portfolio_cfg"), ("execution", "execution_cfg"), ("protection", "protection_cfg")):
        patch = dict(overrides.get(stage) or {})
        if not patch or result.get(key) is None:
            continue
        values = dataclasses.asdict(result[key])
        values.update(patch)
        valid = {field.name for field in dataclasses.fields(type(result[key]))}
        result[key] = type(result[key])(**{name: value for name, value in values.items() if name in valid})
    return result


def _build_multi_pair_strategy(setup_config, symbols, fee_rate=0.0005, slippage_bps=1.0):
    """Build one shared-account strategy with optional market-specific pipelines."""
    # Import implementations once to trigger registry registration.
    from core.modules.estimator import (DOLSEstimator, EWLSEstimator, HuberEstimator, KalmanEstimator, PeriodicOLSEstimator, RLSEstimator, RollingOLSEstimator, TLSEstimator, WinsorizedOLSEstimator)  # noqa: F401
    from core.modules.signals import SimpleZScoreSignal, MAReversionSignal, ZScoreReversionSignal  # noqa: F401
    from core.modules.sizing import (BetaNeutralSizing, FixedNotionalSizing, VolatilityNeutralSizing)  # noqa: F401
    from core.modules.portfolio import (ConstrainedQPAllocator, EquitySlotAllocator, EqualWeightAllocator, MaxSharpeAllocator, PairTargetCapitalAllocator, MinVarianceAllocator, RiskParityAllocator)  # noqa: F401

    market_views = getattr(setup_config, "market_pipelines", None) or {
        getattr(setup_config, "market", "legacy"): {
            "pipeline": setup_config.pipeline,
            "rebalance": getattr(setup_config, "rebalance", None) or {},
        }
    }
    components_by_market = {
        market: _build_market_components(view.get("pipeline", {}), view.get("rebalance", {}), market)
        for market, view in market_views.items()
    }
    default_components = next(iter(components_by_market.values()))
    pair_defs = []
    for pdict in setup_config.pairs:
        x_sym = pdict.get("x_symbol", pdict.get("long_symbol", ""))
        y_sym = pdict.get("y_symbol", pdict.get("short_symbol", ""))
        enabled = pdict.get("enabled", True)
        # A disabled Pair never receives bars, so its symbols may legitimately be
        # absent from data.symbols; only enabled Pairs must resolve.
        if enabled and (x_sym not in symbols or y_sym not in symbols):
            raise ValueError(f"pair {pdict.get('pair_id','?')}: symbols not in data config")
        x_info, y_info = symbols.get(x_sym) or {}, symbols.get(y_sym) or {}
        pair_defs.append(PairDefinition(
            pair_id=pdict.get("pair_id", f"{x_sym}_{y_sym}"),
            market=pdict.get("market", getattr(setup_config, "market", "legacy")),
            x_exchange=pdict.get("x_exchange", x_info.get("exchange", "binance")),
            x_symbol=x_sym, y_exchange=pdict.get("y_exchange", y_info.get("exchange", "binance")), y_symbol=y_sym,
            enabled=pdict.get("enabled", True), tags=list(pdict.get("tags", [])),
            model_lookback_bars_override=pdict.get("model_lookback_bars_override"), notes=pdict.get("notes", ""),
            target_capital=(float(pdict["target_capital"]) if pdict.get("target_capital") is not None else None),
            parameter_grid=dict(pdict.get("parameter_grid") or {}),
            parameters=dict(pdict.get("parameters") or {}),
        ))
    for market, components in components_by_market.items():
        pf_cfg = components["portfolio_cfg"]
        if ("method" in (market_views.get(market, {}).get("pipeline", {}).get("portfolio", {}) or {})
                and pf_cfg.method == "pair_target_capital"):
            missing = [p.pair_id for p in pair_defs if p.enabled and p.market == market and (p.target_capital is None or p.target_capital <= 0)]
            if missing:
                raise ValueError(f"{market}: pair_target_capital requires positive target_capital for enabled Pair(s): {missing}")

    missing_profiles = [
        p.pair_id for p in pair_defs
        if p.enabled and p.market != "legacy" and p.market not in components_by_market
    ]
    if missing_profiles:
        raise ValueError(
            "enabled Pair(s) reference a market without a pipeline profile: "
            f"{missing_profiles}"
        )
    pair_components = {}
    for p, raw_pair in zip(pair_defs, setup_config.pairs):
        base_components = components_by_market.get(p.market, default_components)
        overrides = dict(raw_pair.get("parameters") or {})
        pair_components[p.pair_id] = _override_components(base_components, overrides)
    return MultiPairStrategy(
        pairs=pair_defs,
        estimator_ctor=default_components["estimator_ctor"], estimator_cfg=default_components["estimator_cfg"],
        signal_ctor=default_components["signal_ctor"], signal_cfg=default_components["signal_cfg"],
        sizing_ctor=default_components["sizing_ctor"], sizing_cfg=default_components["sizing_cfg"],
        portfolio_ctor=default_components["portfolio_ctor"], portfolio_cfg=default_components["portfolio_cfg"],
        execution_cfg=default_components["execution_cfg"], protection_cfg=default_components["protection_cfg"],
        rebalance_cfg=default_components["rebalance_cfg"], pair_components=pair_components,
        fee_rate=fee_rate, slippage_bps=slippage_bps,
    )
