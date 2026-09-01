from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = (
    PROJECT_ROOT
    / "documents"
    / "01_Beta模型研究"
    / "02_评估数据"
    / "三维度数据"
)
OUTPUT_ROOT = SOURCE_ROOT / "统一指标"
SWEEP_RESULTS = PROJECT_ROOT / "results" / "sweep_batch" / "sweep_results.csv"

MODEL_FOLDERS = {
    "Rolling OLS": ("rolling_ols", "Rolling OLS"),
    "TLS": ("tls", "TLS"),
    "DOLS": ("dols", "DOLS"),
    "EWLS": ("ewls", "EWLS"),
    "Huber": ("huber", "Huber"),
}

PRIMARY_EVENT_Z = 2.5
PRIMARY_EXIT_Z = 0.5
PRIMARY_HORIZON_MIN = 300

IDENTITY_COLUMNS = [
    "model",
    "model_display",
    "config_key",
    "spec",
    "lookback_label",
    "lookback_bars",
    "update_interval_label",
    "update_interval_bars",
    "sample_scope",
    "dols_lead_lag_order",
    "ewls_half_life_bars",
    "huber_delta",
    "huber_iterations",
    "sweep_eligible",
]


def _column(frame: pd.DataFrame, *names: str, default=np.nan) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(default, index=frame.index)


def _number_column(frame: pd.DataFrame, *names: str) -> pd.Series:
    return pd.to_numeric(_column(frame, *names), errors="coerce")


def _format_parameter(value) -> str:
    if pd.isna(value):
        return "-"
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else f"{numeric:g}"


def _config_key(row: pd.Series) -> str:
    parts = [
        str(row["model"]),
        f"L{int(row['lookback_bars'])}",
        f"U{int(row['update_interval_bars'])}",
    ]
    if row["model"] == "dols":
        parts.append(f"K{_format_parameter(row['dols_lead_lag_order'])}")
    elif row["model"] == "ewls":
        parts.append(f"HL{_format_parameter(row['ewls_half_life_bars'])}")
    elif row["model"] == "huber":
        parts.extend([
            f"D{_format_parameter(row['huber_delta'])}",
            f"I{_format_parameter(row['huber_iterations'])}",
        ])
    return "|".join(parts)


def _identity(frame: pd.DataFrame, folder: str) -> pd.DataFrame:
    model, display = MODEL_FOLDERS[folder]
    result = pd.DataFrame(index=frame.index)
    result["model"] = model
    result["model_display"] = display
    result["spec"] = _column(frame, "spec", default="")
    result["lookback_label"] = _column(frame, "lookback_label")
    result["lookback_bars"] = _number_column(frame, "lookback_bars").astype("Int64")
    result["update_interval_label"] = _column(frame, "update_interval_label")
    result["update_interval_bars"] = _number_column(frame, "update_interval_bars").astype("Int64")
    result["sample_scope"] = _column(frame, "sample_scope")
    result["dols_lead_lag_order"] = _number_column(frame, "K").astype("Int64")
    result["ewls_half_life_bars"] = _number_column(frame, "ewls_half_life_bars").astype("Int64")
    result["huber_delta"] = _number_column(frame, "huber_delta")
    result["huber_iterations"] = _number_column(frame, "huber_iterations").astype("Int64")
    result["sweep_eligible"] = (
        (result["update_interval_bars"] == 1440)
        | ((result["update_interval_bars"] == 4320) & (result["lookback_bars"] >= 28800))
    )
    result["config_key"] = result.apply(_config_key, axis=1)
    return result


def _pair_identity(frame: pd.DataFrame, folder: str) -> pd.DataFrame:
    result = _identity(frame, folder)
    result.insert(0, "pair_id", _column(frame, "pair_id"))
    result.insert(1, "peer", _column(frame, "peer"))
    result.insert(2, "target", _column(frame, "target"))
    return result


def normalize_residual_quality(folder: str) -> pd.DataFrame:
    path = SOURCE_ROOT / folder / "03_residual_quality_pair.csv"
    source = pd.read_csv(path)
    result = _pair_identity(source, folder)
    result["n_oos_cycles"] = _number_column(source, "n_oos_cycles", "n_cycles")
    result["median_oos_bars"] = _number_column(source, "median_oos_bars")
    result["formation_adf_pass_rate"] = _number_column(source, "formation_adf_pass_rate")
    result["oos_adf_pass_rate"] = _number_column(source, "oos_adf_pass_rate")
    result["formation_kpss_pass_rate"] = _number_column(source, "formation_kpss_auto_pass_rate")
    result["oos_kpss_pass_rate"] = _number_column(source, "oos_kpss_auto_pass_rate")
    result["formation_joint_pass_rate"] = _number_column(source, "formation_joint_pass_rate")
    result["oos_joint_pass_rate"] = _number_column(
        source, "oos_joint_stationarity_pass_rate", "oos_joint_pass_rate"
    )
    result["oos_residual_mean"] = _number_column(source, "median_oos_resid_mean")
    result["oos_residual_std"] = _number_column(source, "median_oos_resid_std")
    result["oos_abs_skew"] = _number_column(source, "median_oos_abs_skew")
    result["oos_kurtosis"] = _number_column(source, "median_oos_kurtosis")
    result["abs_mean_drift"] = _number_column(source, "median_abs_mean_drift")
    result["std_ratio_oos_over_formation"] = _number_column(source, "median_std_ratio_oos_over_form")
    result["abs_q05_drift"] = _number_column(source, "median_abs_q05_drift")
    result["abs_q95_drift"] = _number_column(source, "median_abs_q95_drift")
    result["oos_ar1_phi"] = _number_column(source, "median_oos_ar1_phi")
    result["oos_half_life_min"] = _number_column(source, "median_oos_half_life_min")
    result["source_file"] = str(path.relative_to(PROJECT_ROOT))
    return result


def normalize_reversion_quality(folder: str, residual: pd.DataFrame) -> pd.DataFrame:
    path = SOURCE_ROOT / folder / "04_reversion_quality_pair.csv"
    source = pd.read_csv(path)

    if folder in {"Rolling OLS", "TLS"}:
        result = _pair_identity(source, folder)
        result["ar1_phi"] = _number_column(source, "ar1_phi_median")
        result["half_life_valid_rate"] = _number_column(source, "half_life_valid_rate")
        result["half_life_median_min"] = _number_column(source, "half_life_median_min")
        result["half_life_q25_min"] = _number_column(source, "half_life_q25_min")
        result["half_life_q75_min"] = _number_column(source, "half_life_q75_min")
        result["half_life_iqr_min"] = _number_column(source, "half_life_iqr_min")
        result["event_metrics_available_at_pair_level"] = False
        for name, value in {
            "event_z_threshold": PRIMARY_EVENT_Z,
            "exit_z": PRIMARY_EXIT_Z,
            "horizon_min": PRIMARY_HORIZON_MIN,
            "n_events": np.nan,
            "reversion_hit_rate": np.nan,
            "timeout_rate": np.nan,
            "median_time_to_reversion_min": np.nan,
            "q25_time_to_reversion_min": np.nan,
            "q75_time_to_reversion_min": np.nan,
            "mean_max_adverse_z_increment": np.nan,
            "median_max_adverse_z_increment": np.nan,
            "mean_max_favorable_z_reduction": np.nan,
            "median_max_favorable_z_reduction": np.nan,
            "positive_event_share": np.nan,
        }.items():
            result[name] = value
    else:
        event_mask = (
            np.isclose(_number_column(source, "event_z_threshold"), PRIMARY_EVENT_Z)
            & np.isclose(_number_column(source, "exit_z"), PRIMARY_EXIT_Z)
            & (_number_column(source, "horizon_min") == PRIMARY_HORIZON_MIN)
        )
        selected = source.loc[event_mask].copy()
        result = _pair_identity(selected, folder)
        result["event_metrics_available_at_pair_level"] = True
        for name in [
            "event_z_threshold",
            "exit_z",
            "horizon_min",
            "n_events",
            "reversion_hit_rate",
            "timeout_rate",
            "median_time_to_reversion_min",
            "q25_time_to_reversion_min",
            "q75_time_to_reversion_min",
            "mean_max_adverse_z_increment",
            "median_max_adverse_z_increment",
            "mean_max_favorable_z_reduction",
            "median_max_favorable_z_reduction",
            "positive_event_share",
        ]:
            result[name] = _number_column(selected, name)

        half_life = residual[
            ["config_key", "pair_id", "sample_scope", "oos_ar1_phi", "oos_half_life_min"]
        ].rename(columns={
            "oos_ar1_phi": "ar1_phi",
            "oos_half_life_min": "half_life_median_min",
        })
        result = result.merge(
            half_life,
            on=["config_key", "pair_id", "sample_scope"],
            how="left",
            validate="one_to_one",
        )
        result["half_life_valid_rate"] = np.nan
        result["half_life_q25_min"] = np.nan
        result["half_life_q75_min"] = np.nan
        result["half_life_iqr_min"] = np.nan

    result["source_file"] = str(path.relative_to(PROJECT_ROOT))
    return result


def normalize_beta_stability(folder: str) -> pd.DataFrame:
    path = SOURCE_ROOT / folder / "05_beta_stability_pair.csv"
    source = pd.read_csv(path)
    result = _pair_identity(source, folder)
    result["n_model_updates"] = _number_column(source, "n_model_updates", "n_valid_updates")
    result["n_beta_changes"] = _number_column(source, "n_beta_changes", "n_consecutive_beta_changes")
    result["beta_median"] = _number_column(source, "beta_median")
    result["median_abs_beta"] = _number_column(source, "median_abs_beta")
    result["near_zero_beta_rate"] = _number_column(source, "near_zero_beta_rate")
    result["extreme_beta_rate"] = _number_column(source, "extreme_beta_rate_abs_gt_5", "extreme_beta_rate")
    result["abs_beta_change_median"] = _number_column(source, "abs_change_median", "median_abs_beta_change")
    result["abs_beta_change_p95"] = _number_column(source, "abs_change_p95")
    result["abs_beta_change_p99"] = _number_column(source, "abs_change_p99")
    result["abs_beta_change_max"] = _number_column(source, "abs_change_max")
    result["relative_beta_change_median"] = _number_column(
        source, "rel_change_median", "median_relative_beta_change"
    )
    result["relative_beta_change_p95"] = _number_column(
        source, "rel_change_p95", "p95_relative_beta_change"
    )
    result["relative_beta_change_p99"] = _number_column(
        source, "rel_change_p99", "p99_relative_beta_change"
    )
    result["relative_beta_change_max"] = _number_column(
        source, "rel_change_max", "max_relative_beta_change"
    )
    result["jump_rate_gt_10pct"] = _number_column(
        source, "jump_rate_rel_gt_10pct", "jump_gt_10pct_rate"
    )
    result["jump_rate_gt_20pct"] = _number_column(
        source, "jump_rate_rel_gt_20pct", "jump_gt_20pct_rate"
    )
    result["sign_flip_rate"] = _number_column(source, "sign_flip_rate")

    update_days = result["update_interval_bars"].astype(float) / 1440.0
    supplied_daily = _number_column(
        source, "median_abs_beta_change_per_day", "supplemental_abs_change_per_day_median"
    )
    supplied_daily_p95 = _number_column(
        source, "p95_abs_beta_change_per_day", "supplemental_abs_change_per_day_p95"
    )
    result["abs_beta_change_per_day_median"] = supplied_daily.fillna(
        result["abs_beta_change_median"] / update_days
    )
    result["abs_beta_change_per_day_p95"] = supplied_daily_p95.fillna(
        result["abs_beta_change_p95"] / update_days
    )
    result["per_day_metric_derived"] = supplied_daily.isna()
    result["source_file"] = str(path.relative_to(PROJECT_ROOT))
    return result


def _median(frame: pd.DataFrame, column: str):
    return pd.to_numeric(frame[column], errors="coerce").median() if column in frame else np.nan


def _quantile(frame: pd.DataFrame, column: str, quantile: float):
    return pd.to_numeric(frame[column], errors="coerce").quantile(quantile) if column in frame else np.nan


def _ols_tls_event_summaries() -> dict[tuple[str, str], dict]:
    summaries = {}
    for folder in ["Rolling OLS", "TLS"]:
        path = SOURCE_ROOT / folder / "11_cross_pair_model_summary.csv"
        source = pd.read_csv(path)
        identity = _identity(source, folder)
        for index, row in identity.iterrows():
            summaries[(row["config_key"], row["sample_scope"])] = {
                "d2_reversion_hit_rate_median": source.loc[index, "primary_event_median_pair_hit_rate"],
                "d2_reversion_hit_rate_q25": source.loc[index, "primary_event_q25_pair_hit_rate"],
                "d2_reversion_time_median_min": source.loc[index, "primary_event_median_reversion_time_min"],
                "d2_max_adverse_z_median": source.loc[index, "primary_event_median_max_adverse_z"],
                "d2_event_summary_source": str(path.relative_to(PROJECT_ROOT)),
            }
    return summaries


def build_config_summary(
    residual: pd.DataFrame,
    reversion: pd.DataFrame,
    beta: pd.DataFrame,
) -> pd.DataFrame:
    event_summaries = _ols_tls_event_summaries()
    rows = []
    for (config_key, scope), residual_group in residual.groupby(
        ["config_key", "sample_scope"], dropna=False, sort=True
    ):
        first = residual_group.iloc[0]
        reversion_group = reversion[
            (reversion["config_key"] == config_key) & (reversion["sample_scope"] == scope)
        ]
        beta_group = beta[(beta["config_key"] == config_key) & (beta["sample_scope"] == scope)]
        row = {column: first[column] for column in IDENTITY_COLUMNS}
        row.update({
            "n_pairs_residual": residual_group["pair_id"].nunique(),
            "n_pairs_reversion": reversion_group["pair_id"].nunique(),
            "n_pairs_beta": beta_group["pair_id"].nunique(),
            "d1_formation_adf_pass_rate_median": _median(residual_group, "formation_adf_pass_rate"),
            "d1_formation_kpss_pass_rate_median": _median(residual_group, "formation_kpss_pass_rate"),
            "d1_oos_adf_pass_rate_median": _median(residual_group, "oos_adf_pass_rate"),
            "d1_oos_kpss_pass_rate_median": _median(residual_group, "oos_kpss_pass_rate"),
            "d1_oos_joint_pass_rate_median": _median(residual_group, "oos_joint_pass_rate"),
            "d1_oos_joint_pass_rate_q25": _quantile(residual_group, "oos_joint_pass_rate", 0.25),
            "d1_abs_mean_drift_median": _median(residual_group, "abs_mean_drift"),
            "d1_std_ratio_oos_over_formation_median": _median(
                residual_group, "std_ratio_oos_over_formation"
            ),
            "d1_abs_q05_drift_median": _median(residual_group, "abs_q05_drift"),
            "d1_abs_q95_drift_median": _median(residual_group, "abs_q95_drift"),
            "d2_half_life_valid_rate_median": _median(reversion_group, "half_life_valid_rate"),
            "d2_half_life_median_min": _median(reversion_group, "half_life_median_min"),
            "d2_half_life_q25_min": _quantile(reversion_group, "half_life_median_min", 0.25),
            "d2_half_life_q75_min": _quantile(reversion_group, "half_life_median_min", 0.75),
            "d2_event_z_threshold": PRIMARY_EVENT_Z,
            "d2_exit_z": PRIMARY_EXIT_Z,
            "d2_horizon_min": PRIMARY_HORIZON_MIN,
            "d2_event_count_total": pd.to_numeric(
                reversion_group.get("n_events", pd.Series(dtype=float)), errors="coerce"
            ).sum(min_count=1),
            "d2_reversion_hit_rate_median": _median(reversion_group, "reversion_hit_rate"),
            "d2_reversion_hit_rate_q25": _quantile(reversion_group, "reversion_hit_rate", 0.25),
            "d2_timeout_rate_median": _median(reversion_group, "timeout_rate"),
            "d2_reversion_time_median_min": _median(reversion_group, "median_time_to_reversion_min"),
            "d2_max_adverse_z_median": _median(reversion_group, "median_max_adverse_z_increment"),
            "d2_max_favorable_z_reduction_median": _median(
                reversion_group, "median_max_favorable_z_reduction"
            ),
            "d2_event_summary_source": "aggregated from unified pair rows",
            "d3_abs_beta_change_median": _median(beta_group, "abs_beta_change_median"),
            "d3_relative_beta_change_median": _median(beta_group, "relative_beta_change_median"),
            "d3_relative_beta_change_p95_median": _median(beta_group, "relative_beta_change_p95"),
            "d3_jump_rate_gt_10pct_median": _median(beta_group, "jump_rate_gt_10pct"),
            "d3_jump_rate_gt_20pct_median": _median(beta_group, "jump_rate_gt_20pct"),
            "d3_sign_flip_rate_median": _median(beta_group, "sign_flip_rate"),
            "d3_near_zero_beta_rate_median": _median(beta_group, "near_zero_beta_rate"),
            "d3_extreme_beta_rate_median": _median(beta_group, "extreme_beta_rate"),
            "d3_abs_beta_change_per_day_median": _median(
                beta_group, "abs_beta_change_per_day_median"
            ),
        })
        row.update(event_summaries.get((config_key, scope), {}))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["sample_scope", "model", "lookback_bars", "update_interval_bars", "config_key"]
    )


def _sweep_identity(source: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=source.index)
    result["model"] = _column(
        source, "setups.multi_pair_beta.pipeline.estimator.method"
    ).astype(str)
    display = {value[0]: value[1] for value in MODEL_FOLDERS.values()}
    result["model_display"] = result["model"].map(display)
    result["lookback_bars"] = _number_column(
        source, "setups.multi_pair_beta.pipeline.estimator.model_lookback_bars"
    ).astype("Int64")
    result["update_interval_bars"] = _number_column(
        source, "setups.multi_pair_beta.pipeline.estimator.model_update_interval_bars"
    ).astype("Int64")
    result["dols_lead_lag_order"] = _number_column(
        source, "setups.multi_pair_beta.pipeline.estimator.dols_lead_lag_order"
    ).astype("Int64")
    result["ewls_half_life_bars"] = _number_column(
        source, "setups.multi_pair_beta.pipeline.estimator.ewls_half_life_bars"
    ).astype("Int64")
    result["huber_delta"] = _number_column(
        source, "setups.multi_pair_beta.pipeline.estimator.huber_delta"
    )
    result["huber_iterations"] = _number_column(
        source, "setups.multi_pair_beta.pipeline.estimator.huber_iterations"
    ).astype("Int64")
    result["config_key"] = result.apply(_config_key, axis=1)
    return result


def build_four_dimension_join(config_summary: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(SWEEP_RESULTS)
    identity = _sweep_identity(source)
    common = config_summary[config_summary["sample_scope"] == "COMMON_60D"].copy()
    metric_columns = [column for column in common.columns if column.startswith(("d1_", "d2_", "d3_"))]
    lookup = common[["config_key", *metric_columns]].drop_duplicates("config_key")

    base = pd.DataFrame({
        "config_id": _column(source, "config_id"),
        "experiment_name": _column(source, "experiment_name"),
        "config_path": _column(source, "config_path"),
        "model": identity["model"],
        "model_display": identity["model_display"],
        "config_key": identity["config_key"],
        "lookback_bars": identity["lookback_bars"],
        "update_interval_bars": identity["update_interval_bars"],
        "dols_lead_lag_order": identity["dols_lead_lag_order"],
        "ewls_half_life_bars": identity["ewls_half_life_bars"],
        "huber_delta": identity["huber_delta"],
        "huber_iterations": identity["huber_iterations"],
        "entry_z": _number_column(source, "setups.multi_pair_beta.pipeline.signal.entry_z"),
        "exit_z": _number_column(source, "setups.multi_pair_beta.pipeline.signal.exit_z"),
    })
    joined = base.merge(lookup, on="config_key", how="left", indicator=True, validate="many_to_one")
    joined["three_dimension_match_status"] = joined["_merge"].map({
        "both": "MATCHED",
        "left_only": "PARAMETER_MISMATCH",
        "right_only": "UNEXPECTED",
    }).astype(str)
    joined["three_dimension_match_reason"] = ""
    ewls_mismatch = (joined["model"] == "ewls") & (joined["_merge"] == "left_only")
    huber_mismatch = (joined["model"] == "huber") & (joined["_merge"] == "left_only")
    joined.loc[ewls_mismatch, "three_dimension_match_reason"] = (
        "三维数据 EWLS half_life=1440；回测 half_life=模型窗口"
    )
    joined.loc[huber_mismatch, "three_dimension_match_reason"] = (
        "三维数据 Huber delta=1.345；回测 delta=1.96"
    )
    other_mismatch = (joined["_merge"] == "left_only") & ~ewls_mismatch & ~huber_mismatch
    joined.loc[other_mismatch, "three_dimension_match_reason"] = "未找到完整参数一致的 COMMON_60D 三维数据"
    joined = joined.drop(columns="_merge")

    trading_mapping = {
        "initial_equity": "d4_initial_equity",
        "final_equity": "d4_final_equity",
        "total_return": "d4_total_return",
        "annualized_return": "d4_annualized_return",
        "annualized_volatility": "d4_annualized_volatility",
        "sharpe": "d4_sharpe",
        "calmar": "d4_calmar",
        "max_drawdown": "d4_max_drawdown",
        "trade_count": "d4_trade_count",
        "win_rate": "d4_win_rate",
        "profit_loss_ratio": "d4_profit_loss_ratio",
        "average_holding_minutes": "d4_average_holding_minutes",
        "daily_turnover": "d4_daily_turnover",
        "total_fee": "d4_total_fee",
        "total_slippage": "d4_total_slippage",
        "funding_fee": "d4_funding_fee",
        "funding_paid": "d4_funding_paid",
        "funding_received": "d4_funding_received",
        "funding_payment_count": "d4_funding_payment_count",
        "orders": "d4_orders",
        "trades": "d4_trades",
        "risk_checks": "d4_risk_checks",
        "error": "d4_error",
    }
    for source_name, target_name in trading_mapping.items():
        joined[target_name] = _column(source, source_name)
    return joined.sort_values("config_id")


def metric_dictionary() -> pd.DataFrame:
    rows = [
        ("d1_oos_adf_pass_rate_median", 1, "样本外ADF通过率的跨Pair中位数", "HIGH"),
        ("d1_oos_kpss_pass_rate_median", 1, "样本外KPSS通过率的跨Pair中位数", "HIGH"),
        ("d1_oos_joint_pass_rate_median", 1, "样本外ADF与KPSS联合通过率的跨Pair中位数", "HIGH"),
        ("d1_oos_joint_pass_rate_q25", 1, "样本外联合通过率的跨Pair 25%分位数", "HIGH"),
        ("d1_abs_mean_drift_median", 1, "残差均值绝对漂移的跨Pair中位数", "LOW"),
        ("d1_std_ratio_oos_over_formation_median", 1, "样本外/形成期残差标准差比率", "TARGET_1"),
        ("d1_abs_q05_drift_median", 1, "残差5%分位数绝对漂移", "LOW"),
        ("d1_abs_q95_drift_median", 1, "残差95%分位数绝对漂移", "LOW"),
        ("d2_half_life_median_min", 2, "Half-life跨Pair中位数（分钟）", "TARGET_RANGE"),
        ("d2_reversion_hit_rate_median", 2, "主事件300分钟内回归成功率中位数", "HIGH"),
        ("d2_reversion_hit_rate_q25", 2, "主事件回归成功率跨Pair 25%分位数", "HIGH"),
        ("d2_reversion_time_median_min", 2, "成功事件中位回归时间（分钟）", "LOW"),
        ("d2_max_adverse_z_median", 2, "回归前最大不利Z增量中位数", "LOW"),
        ("d3_relative_beta_change_median", 3, "Beta相对变化中位数", "LOW"),
        ("d3_relative_beta_change_p95_median", 3, "各Pair Beta相对变化95%分位数的中位数", "LOW"),
        ("d3_jump_rate_gt_20pct_median", 3, "Beta相对跳变超过20%的比例中位数", "LOW"),
        ("d3_sign_flip_rate_median", 3, "Beta符号翻转率中位数", "LOW"),
        ("d3_near_zero_beta_rate_median", 3, "Beta接近零比例中位数", "LOW"),
        ("d3_extreme_beta_rate_median", 3, "极端Beta比例中位数", "LOW"),
        ("d3_abs_beta_change_per_day_median", 3, "按天标准化的Beta绝对变化中位数", "LOW"),
        ("d4_total_return", 4, "扣除手续费、滑点和资金费率后的总收益率", "HIGH"),
        ("d4_annualized_return", 4, "扣除成本后的复合年化收益率（CAGR）", "HIGH"),
        ("d4_annualized_volatility", 4, "权益分钟收益年化波动率", "LOW"),
        ("d4_sharpe", 4, "扣除成本后的Sharpe", "HIGH"),
        ("d4_calmar", 4, "复合年化收益率（CAGR）/最大回撤绝对值", "HIGH"),
        ("d4_max_drawdown", 4, "最大回撤（负数）", "HIGH"),
        ("d4_win_rate", 4, "按position_id汇总的盈利持仓比例", "HIGH"),
        ("d4_profit_loss_ratio", 4, "平均盈利/平均亏损绝对值", "HIGH"),
        ("d4_average_holding_minutes", 4, "平均持仓分钟数", "CONTEXT"),
        ("d4_daily_turnover", 4, "日均成交名义金额/平均权益", "LOW"),
        ("d4_total_fee", 4, "手续费总额", "LOW"),
        ("d4_total_slippage", 4, "滑点总额", "LOW"),
        ("d4_funding_fee", 4, "资金费率净支付结果", "LOW"),
    ]
    return pd.DataFrame(rows, columns=["metric", "dimension", "definition", "preferred_direction"])


def _write_csv(frame: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_ROOT / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    residual_frames = []
    reversion_frames = []
    beta_frames = []
    for folder in MODEL_FOLDERS:
        residual = normalize_residual_quality(folder)
        residual_frames.append(residual)
        reversion_frames.append(normalize_reversion_quality(folder, residual))
        beta_frames.append(normalize_beta_stability(folder))

    residual = pd.concat(residual_frames, ignore_index=True)
    reversion = pd.concat(reversion_frames, ignore_index=True)
    beta = pd.concat(beta_frames, ignore_index=True)
    config_summary = build_config_summary(residual, reversion, beta)
    four_dimension = build_four_dimension_join(config_summary)
    dictionary = metric_dictionary()

    outputs = [
        _write_csv(residual, "01_residual_quality_pair_unified.csv"),
        _write_csv(reversion, "02_reversion_quality_pair_unified.csv"),
        _write_csv(beta, "03_beta_stability_pair_unified.csv"),
        _write_csv(config_summary, "04_three_dimension_config_unified.csv"),
        _write_csv(four_dimension, "05_four_dimension_sweep_joined.csv"),
        _write_csv(dictionary, "06_metric_dictionary.csv"),
    ]

    matched = int((four_dimension["three_dimension_match_status"] == "MATCHED").sum())
    mismatch = len(four_dimension) - matched
    checks = pd.DataFrame([
        ("residual_pair_rows", len(residual), 1932, len(residual) == 1932),
        ("reversion_pair_rows", len(reversion), 1932, len(reversion) == 1932),
        ("beta_pair_rows", len(beta), 1932, len(beta) == 1932),
        ("three_dimension_config_rows", len(config_summary), 140, len(config_summary) == 140),
        ("sweep_rows", len(four_dimension), 168, len(four_dimension) == 168),
        ("exact_parameter_matches", matched, 120, matched == 120),
        ("parameter_mismatches", mismatch, 48, mismatch == 48),
        (
            "duplicate_config_scope_keys",
            int(config_summary.duplicated(["config_key", "sample_scope"]).sum()),
            0,
            not config_summary.duplicated(["config_key", "sample_scope"]).any(),
        ),
    ], columns=["check", "actual", "expected", "passed"])
    outputs.append(_write_csv(checks, "07_checks.csv"))

    print(f"output_dir={OUTPUT_ROOT}")
    for path in outputs:
        print(f"wrote={path.name}")
    print(f"matched={matched}")
    print(f"parameter_mismatch={mismatch}")
    if not checks["passed"].all():
        raise RuntimeError("unified metric checks failed")


if __name__ == "__main__":
    main()
