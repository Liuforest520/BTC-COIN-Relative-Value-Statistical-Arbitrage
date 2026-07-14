from core.modules.signals.cointegration import CointegrationZScoreSignal


def build_signal(setup_config, symbols):
    pair = setup_config.pair
    x_symbol = pair.get("x_symbol", pair.get("long_symbol"))
    y_symbol = pair.get("y_symbol", pair.get("short_symbol"))
    if x_symbol not in symbols or y_symbol not in symbols:
        raise ValueError("pair must contain valid x_symbol and y_symbol")

    signal_config = setup_config.signal
    signal_type = signal_config["type"]
    spread_model = signal_config.get("spread_model", {})
    hedge_model = signal_config.get("hedge_model", {})
    entry_rule = signal_config.get("entry_rule", {})
    entry_filters = signal_config.get("entry_filters", {})
    beta_stability = entry_filters.get("beta_stability", {})
    common_kwargs = {
        "x_exchange": symbols[x_symbol]["exchange"],
        "x_symbol": x_symbol,
        "y_exchange": symbols[y_symbol]["exchange"],
        "y_symbol": y_symbol,
        "warmup_bars": int(signal_config["warmup_bars"]),
        "zscore_lookback_bars": int(signal_config.get("zscore_lookback_bars", signal_config["warmup_bars"])),
        "entry_z": float(signal_config.get("entry_z", 2.0)),
        "exit_z": float(signal_config.get("exit_z", 0.5)),
        "entry_rule_method": entry_rule.get("method", "fixed_z"),
        "entry_rule_lookback_bars": int(entry_rule.get("lookback_bars", 10080)),
        "entry_rule_update_interval_bars": int(entry_rule.get("update_interval_bars", 60)),
        "entry_rule_upper_percentile": float(entry_rule.get("upper_percentile", 97.5)),
        "entry_rule_lower_percentile": float(entry_rule.get("lower_percentile", 2.5)),
        "entry_rule_min_samples": int(entry_rule.get("min_samples", entry_rule.get("lookback_bars", 10080))),
        "entry_rule_min_abs_entry_z": entry_rule.get("min_abs_entry_z"),
        "entry_rule_max_abs_entry_z": entry_rule.get("max_abs_entry_z"),
    }
    model_kwargs = {
        "model_lookback_bars": int(spread_model.get("lookback_bars", signal_config.get("model_lookback_bars", signal_config["warmup_bars"]))),
        "model_update_interval_bars": int(spread_model.get("update_interval_bars", signal_config.get("model_update_interval_bars", 1))),
        "regression_method": spread_model.get("regression_method", signal_config.get("regression_method", "log_price")),
        "deming_delta": float(spread_model.get("deming_delta", signal_config.get("deming_delta", 1.0))),
        "hedge_model_lookback_bars": int(hedge_model.get("lookback_bars", spread_model.get("lookback_bars", signal_config.get("model_lookback_bars", signal_config["warmup_bars"])))),
        "hedge_model_update_interval_bars": int(hedge_model.get("update_interval_bars", 1)),
        "hedge_regression_method": hedge_model.get("regression_method", spread_model.get("regression_method", signal_config.get("regression_method", "log_price"))),
        "hedge_deming_delta": float(hedge_model.get("deming_delta", spread_model.get("deming_delta", signal_config.get("deming_delta", 1.0)))),
        "beta_stability_enabled": _as_bool(beta_stability.get("enabled", False)),
        "beta_stability_lookback_bars": int(beta_stability.get("lookback_bars", 1440)),
        "beta_stability_max_cv": float(beta_stability.get("max_cv", 0.10)),
        "beta_stability_min_samples": int(beta_stability.get("min_samples", 240)),
    }

    if signal_type == "cointegration_zscore":
        return CointegrationZScoreSignal(**common_kwargs, **model_kwargs)

    raise ValueError(f"unsupported signal type: {signal_type}; only cointegration_zscore is supported")


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)
