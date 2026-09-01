from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BETA_DATA_ROOT = PROJECT_ROOT / "documents" / "01_Beta模型研究" / "02_评估数据"
UNIFIED_ROOT = BETA_DATA_ROOT / "三维度数据" / "统一指标"
RETURN_ROOT = BETA_DATA_ROOT / "收益率模型总结"
RETURN_BACKTEST_ROOT = RETURN_ROOT / "收益率回测结果"
OUTPUT_ROOT = BETA_DATA_ROOT / "Beta模型四维评估分组"

THREE_DIMENSION_PATH = UNIFIED_ROOT / "04_three_dimension_config_unified.csv"
FOUR_DIMENSION_PATH = UNIFIED_ROOT / "05_four_dimension_sweep_joined.csv"

OLD_MODELS = ["rolling_ols", "tls", "dols", "ewls", "huber"]
RETURN_MODELS = ["age_weighted", "fast_slow_slow"]
MODEL_ORDER = [*OLD_MODELS, *RETURN_MODELS]
MODEL_DISPLAY = {
    "rolling_ols": "Rolling OLS",
    "tls": "TLS",
    "dols": "DOLS",
    "ewls": "EWLS",
    "huber": "Huber",
    "age_weighted": "Age Weighted",
    "fast_slow_slow": "Fast-Slow-Slow",
}

COMMON_PARAMETERS = [
    "model_lookback_bars",
    "model_update_interval_bars",
    "entry_z",
    "exit_z",
]

DIMENSIONS = {
    "01_残差统计性质": [
        "formation_adf_pass_rate_median",
        "formation_kpss_pass_rate_median",
        "oos_adf_pass_rate_median",
        "oos_kpss_pass_rate_median",
        "oos_joint_pass_rate_median",
        "oos_joint_pass_rate_q25",
        "abs_mean_drift_median",
        "std_ratio_oos_over_formation_median",
        "abs_q05_drift_median",
        "abs_q95_drift_median",
    ],
    "02_均值回复能力": [
        "half_life_median_min",
        "half_life_q25_min",
        "half_life_q75_min",
        "reversion_hit_rate_median",
        "reversion_time_median_min",
        "max_adverse_z_median",
    ],
    "03_Beta稳定性": [
        "abs_beta_change_median",
        "relative_beta_change_median",
        "relative_beta_change_p95_median",
        "jump_rate_gt_10pct_median",
        "jump_rate_gt_20pct_median",
        "sign_flip_rate_median",
        "near_zero_beta_rate_median",
        "extreme_beta_rate_median",
        "abs_beta_change_per_day_median",
    ],
    "04_回测结果": [
        "initial_equity",
        "final_equity",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "calmar",
        "max_drawdown",
        "trade_count",
        "win_rate",
        "profit_loss_ratio",
        "average_holding_minutes",
        "daily_turnover",
        "total_fee",
        "total_slippage",
        "funding_fee",
        "funding_paid",
        "funding_received",
        "funding_payment_count",
        "orders",
        "trades",
        "risk_checks",
    ],
}

RETURN_FIELD_MAPPING = {
    "01_残差统计性质": {
        "oos_adf_pass_rate_median": "median_raw_adf_pass",
        "oos_kpss_pass_rate_median": "median_raw_kpss_pass",
        "oos_joint_pass_rate_median": "median_raw_joint_pass",
        "abs_mean_drift_median": "median_raw_mean_drift",
        "std_ratio_oos_over_formation_median": "median_raw_std_ratio",
    },
    "02_均值回复能力": {
        "half_life_median_min": "median_raw_half_life",
        "reversion_hit_rate_median": "median_pair_repair_rate",
        "reversion_time_median_min": "median_pair_repair_time",
        "max_adverse_z_median": "median_pair_max_adverse_z",
    },
    "03_Beta稳定性": {
        "relative_beta_change_median": "median_relative_beta_change",
        "relative_beta_change_p95_median": "p95_relative_beta_change",
        "jump_rate_gt_10pct_median": "jump_gt_10pct_rate",
        "jump_rate_gt_20pct_median": "jump_gt_20pct_rate",
        "sign_flip_rate_median": "sign_flip_rate",
        "abs_beta_change_per_day_median": "median_abs_beta_change_per_day",
    },
    "04_回测结果": {
        "total_return": "portfolio_total_return_net_ex_funding",
        "annualized_return": "portfolio_cagr_net_ex_funding",
        "sharpe": "portfolio_daily_sharpe_net_ex_funding",
        "calmar": "portfolio_calmar_net_ex_funding",
        "max_drawdown": "portfolio_max_drawdown_net_ex_funding",
        "trade_count": "total_closed_trades",
    },
}


def _label_days(bars: int) -> str:
    days = bars / 1440
    return f"{int(days)}D" if days.is_integer() else f"{days:g}D"


def _label_number(value: float) -> str:
    numeric = float(value)
    text = str(int(numeric)) if numeric.is_integer() else f"{numeric:g}"
    return text.replace(".", "p")


def _group_folder_name(index: int, row: pd.Series) -> str:
    lookback = _label_days(int(row["model_lookback_bars"]))
    update = _label_days(int(row["model_update_interval_bars"]))
    entry = _label_number(float(row["entry_z"]))
    exit_value = _label_number(float(row["exit_z"]))
    return f"G{index:03d}_L{lookback}_U{update}_E{entry}_X{exit_value}"


def _result_file_name(row: pd.Series) -> str:
    lookback = _label_days(int(row["model_lookback_bars"]))
    update = _label_days(int(row["model_update_interval_bars"]))
    entry = _label_number(float(row["entry_z"]))
    exit_value = _label_number(float(row["exit_z"]))
    return f"L{lookback}_U{update}_Entry{entry}_Exit{exit_value}.csv"


def _prepare_three_dimension(source: pd.DataFrame) -> pd.DataFrame:
    frame = source[source["sample_scope"] == "COMMON_60D"].copy()
    return frame.rename(columns={
        "lookback_bars": "model_lookback_bars",
        "update_interval_bars": "model_update_interval_bars",
    })


def _prepare_four_dimension(source: pd.DataFrame) -> pd.DataFrame:
    return source.rename(columns={
        "lookback_bars": "model_lookback_bars",
        "update_interval_bars": "model_update_interval_bars",
    }).copy()


def _load_return_sources() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    summaries = {
        model: pd.read_csv(
            RETURN_ROOT / model / "three_dimensions" / "04_three_dimension_cross_pair_summary.csv"
        )
        for model in RETURN_MODELS
    }
    backtests = {
        model: pd.read_csv(RETURN_BACKTEST_ROOT / f"{model}.csv")
        for model in RETURN_MODELS
    }
    return summaries, backtests


def _build_groups(four_dimension: pd.DataFrame) -> pd.DataFrame:
    groups = (
        four_dimension[COMMON_PARAMETERS]
        .drop_duplicates()
        .sort_values(COMMON_PARAMETERS, kind="stable")
        .reset_index(drop=True)
    )
    groups.insert(0, "group_id", [f"G{index:03d}" for index in range(1, len(groups) + 1)])
    groups["lookback_label"] = groups["model_lookback_bars"].map(
        lambda value: _label_days(int(value))
    )
    groups["update_interval_label"] = groups["model_update_interval_bars"].map(
        lambda value: _label_days(int(value))
    )
    groups["folder_name"] = [
        _group_folder_name(index, row)
        for index, (_, row) in enumerate(groups.iterrows(), start=1)
    ]
    groups["result_file_name"] = [_result_file_name(row) for _, row in groups.iterrows()]
    return groups


def _base_model_row(group_row: pd.Series, model: str, variant_count: int) -> dict:
    return {
        "group_id": group_row["group_id"],
        "model": model,
        "model_display": MODEL_DISPLAY[model],
        "model_lookback_bars": int(group_row["model_lookback_bars"]),
        "model_update_interval_bars": int(group_row["model_update_interval_bars"]),
        "entry_z": float(group_row["entry_z"]),
        "exit_z": float(group_row["exit_z"]),
        "variant_count": variant_count,
        "variant_weight": 1.0 / variant_count,
        "aggregation_method": "equal_weight_arithmetic_mean",
    }


def _mean_or_na(frame: pd.DataFrame, column: str):
    if column not in frame.columns:
        return pd.NA
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.mean() if values.notna().any() else pd.NA


def _old_model_rows(
    dimension_name: str,
    metrics: list[str],
    three_dimension: pd.DataFrame,
    four_dimension: pd.DataFrame,
    group_row: pd.Series,
) -> list[dict]:
    lookback = group_row["model_lookback_bars"]
    update = group_row["model_update_interval_bars"]
    if dimension_name == "04_回测结果":
        variants = four_dimension[
            (four_dimension["model_lookback_bars"] == lookback)
            & (four_dimension["model_update_interval_bars"] == update)
            & (four_dimension["entry_z"] == group_row["entry_z"])
            & (four_dimension["exit_z"] == group_row["exit_z"])
        ]
    else:
        variants = three_dimension[
            (three_dimension["model_lookback_bars"] == lookback)
            & (three_dimension["model_update_interval_bars"] == update)
        ]

    dimension_number = int(dimension_name[:2])
    rows = []
    for model in OLD_MODELS:
        model_variants = variants[variants["model"] == model]
        if model_variants.empty:
            continue
        row = _base_model_row(group_row, model, len(model_variants))
        for metric in metrics:
            row[metric] = _mean_or_na(model_variants, f"d{dimension_number}_{metric}")
        rows.append(row)
    return rows


def _return_three_dimension_variants(
    model: str,
    dimension_name: str,
    summary: pd.DataFrame,
    group_row: pd.Series,
) -> pd.DataFrame:
    lookback_days = int(group_row["model_lookback_bars"]) / 1440
    update_days = int(group_row["model_update_interval_bars"]) / 1440
    source_dimension = "Beta Stability" if dimension_name == "03_Beta稳定性" else "Residual Quality"
    base_mask = (
        (summary["dimension"] == source_dimension)
        & (summary["sample_scope"] == "COMMON_60D")
        & (summary["update_days"] == update_days)
    )

    if model == "age_weighted":
        return summary[base_mask & (summary["lookback_days"] == lookback_days)].copy()
    if model == "fast_slow_slow":
        return summary[
            base_mask
            & (summary["model_role"] == "SLOW")
            & (summary["formation_days"] == lookback_days)
        ].copy()
    raise ValueError(f"unsupported return model: {model}")


def _return_backtest_variants(
    model: str,
    backtest: pd.DataFrame,
    group_row: pd.Series,
) -> pd.DataFrame:
    lookback_days = int(group_row["model_lookback_bars"]) / 1440
    update_days = int(group_row["model_update_interval_bars"]) / 1440
    mask = (
        (backtest["update_days"] == update_days)
        & (backtest["entry_z_threshold"] == group_row["entry_z"])
        & (backtest["repair_z_threshold"] == group_row["exit_z"])
        & (backtest["scope"] == "COMMON_60D")
    )
    lookback_column = "slow_lookback_days" if model == "fast_slow_slow" else "lookback_days"
    return backtest[mask & (backtest[lookback_column] == lookback_days)].copy()


def _return_reversion_variants(
    model: str,
    summary: pd.DataFrame,
    group_row: pd.Series,
) -> pd.DataFrame:
    lookback_days = int(group_row["model_lookback_bars"]) / 1440
    update_days = int(group_row["model_update_interval_bars"]) / 1440
    mask = (
        (summary["dimension"] == "Reversion Quality")
        & (summary["sample_scope"] == "COMMON_60D")
        & (summary["update_days"] == update_days)
        & (summary["entry_z"] == 2.5)
        & (summary["repair_z"] == 0.5)
        & (summary["horizon_min"] == 300)
    )
    lookback_column = "slow_lookback_days" if model == "fast_slow_slow" else "lookback_days"
    return summary[mask & (summary[lookback_column] == lookback_days)].copy()


def _return_model_rows(
    dimension_name: str,
    metrics: list[str],
    summaries: dict[str, pd.DataFrame],
    backtests: dict[str, pd.DataFrame],
    group_row: pd.Series,
) -> list[dict]:
    rows = []
    mapping = RETURN_FIELD_MAPPING[dimension_name]
    for model in RETURN_MODELS:
        if dimension_name == "04_回测结果":
            variants = _return_backtest_variants(model, backtests[model], group_row)
        elif dimension_name == "02_均值回复能力":
            residual_variants = _return_three_dimension_variants(
                model, dimension_name, summaries[model], group_row
            )
            reversion_variants = _return_reversion_variants(
                model, summaries[model], group_row
            )
            if residual_variants.empty and reversion_variants.empty:
                continue
            variant_count = len(reversion_variants) or len(residual_variants)
            row = _base_model_row(group_row, model, variant_count)
            for metric in metrics:
                source_column = mapping.get(metric)
                source = residual_variants if metric.startswith("half_life_") else reversion_variants
                row[metric] = _mean_or_na(source, source_column) if source_column else pd.NA
            rows.append(row)
            continue
        else:
            variants = _return_three_dimension_variants(
                model, dimension_name, summaries[model], group_row
            )
        if variants.empty:
            continue
        row = _base_model_row(group_row, model, len(variants))
        for metric in metrics:
            source_column = mapping.get(metric)
            row[metric] = _mean_or_na(variants, source_column) if source_column else pd.NA
        rows.append(row)
    return rows


def _write_group_readme(group_root: Path, group_row: pd.Series) -> None:
    content = f"""# {group_row['group_id']} 公共参数组

- `model_lookback_bars`: {int(group_row['model_lookback_bars'])}（{_label_days(int(group_row['model_lookback_bars']))}）
- `model_update_interval_bars`: {int(group_row['model_update_interval_bars'])}（{_label_days(int(group_row['model_update_interval_bars']))}）
- `entry_z`: {float(group_row['entry_z']):g}
- `exit_z`: {float(group_row['exit_z']):g}

文件只列出在本公共参数组中有数据的模型。模型未覆盖本组时不填零，也不生成占位行。

同一模型存在多个专属参数版本时，各版本结果按相同权重计算算术平均。`variant_count` 是参与平均的版本数，`variant_weight` 为单个版本权重。

前三个维度统一使用 `COMMON_60D`。收益率模型无法映射到统一字段的指标保留为空。收益率模型第四维结果未包含资金费率，具体口径见根目录说明。
"""
    (group_root / "README.md").write_text(content, encoding="utf-8")


def _write_root_readme() -> None:
    content = """# Beta 模型四维评估分组

## 公共参数

所有结果按照 `model_lookback_bars`、`model_update_interval_bars`、`entry_z` 和 `exit_z` 分组。

当前目录包含原五个价格模型以及 Age Weighted、Fast-Slow-Slow 两个收益率模型。每个维度文件只列出实际覆盖该公共参数组的模型；缺少的模型不按零分处理。

## 专属参数等权整合

同一模型在同一公共参数组中存在多个专属参数版本时，公共指标采用算术平均，每个版本权重均为 `1 / variant_count`。

## 收益率模型的映射

- `age_weighted`：`lookback_days` 映射为公共模型窗口。
- `fast_slow_slow`：`slow_lookback_days` 映射为公共模型窗口；前三维使用 `model_role=SLOW` 的统计结果。
收益率模型无法对应现有统一列的指标写为空值，不进行推断。字段对应关系见 `字段映射.csv`。

## 第四维口径说明

两个收益率模型的第四维是 13 个 Pair 的组合级汇总，但字段明确标记为 `net_ex_funding`，资金费率未纳入。原五模型第四维包含资金费率。当前仅统一字段并保留结果，正式评分时必须把该口径差异作为限制条件。
"""
    (OUTPUT_ROOT / "README.md").write_text(content, encoding="utf-8")


def _write_field_mapping() -> None:
    rows = []
    for dimension_name, metrics in DIMENSIONS.items():
        dimension_number = int(dimension_name[:2])
        return_mapping = RETURN_FIELD_MAPPING[dimension_name]
        for metric in metrics:
            source = return_mapping.get(metric)
            rows.append({
                "dimension": dimension_name,
                "unified_metric": metric,
                "original_five_source": f"d{dimension_number}_{metric}",
                "return_models_source": source if source else "",
                "return_models_status": "MAPPED" if source else "LEFT_BLANK",
            })
    pd.DataFrame(rows).to_csv(
        OUTPUT_ROOT / "字段映射.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    three_dimension = _prepare_three_dimension(pd.read_csv(THREE_DIMENSION_PATH))
    four_dimension = _prepare_four_dimension(pd.read_csv(FOUR_DIMENSION_PATH))
    return_summaries, return_backtests = _load_return_sources()
    groups = _build_groups(four_dimension)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    groups.to_csv(OUTPUT_ROOT / "分组索引.csv", index=False, encoding="utf-8-sig")
    _write_root_readme()
    _write_field_mapping()

    checks = []
    total_rows_by_model = {model: 0 for model in MODEL_ORDER}
    for _, group_row in groups.iterrows():
        group_root = OUTPUT_ROOT / group_row["folder_name"]
        group_root.mkdir(exist_ok=True)
        _write_group_readme(group_root, group_row)

        for dimension_name, metrics in DIMENSIONS.items():
            dimension_root = group_root / dimension_name
            dimension_root.mkdir(exist_ok=True)
            rows = _old_model_rows(
                dimension_name, metrics, three_dimension, four_dimension, group_row
            )
            rows.extend(_return_model_rows(
                dimension_name,
                metrics,
                return_summaries,
                return_backtests,
                group_row,
            ))
            summary = pd.DataFrame(rows)
            summary["_model_order"] = summary["model"].map(MODEL_ORDER.index)
            summary = summary.sort_values("_model_order").drop(columns="_model_order")
            result_path = dimension_root / group_row["result_file_name"]
            summary.to_csv(result_path, index=False, encoding="utf-8-sig")
            for model in summary["model"]:
                total_rows_by_model[model] += 1
            checks.append({
                "group_id": group_row["group_id"],
                "dimension": dimension_name,
                "model_rows": len(summary),
                "available_models": "|".join(summary["model"]),
                "duplicate_model_rows": int(summary["model"].duplicated().sum()),
                "blank_metric_cells": int(summary[metrics].isna().sum().sum()),
            })

    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(OUTPUT_ROOT / "分组检查.csv", index=False, encoding="utf-8-sig")

    expected_files = len(groups) * len(DIMENSIONS)
    actual_files = len(list(OUTPUT_ROOT.glob("G*/**/L*_U*_Entry*_Exit*.csv")))
    if len(groups) != 24:
        raise RuntimeError(f"expected 24 groups, got {len(groups)}")
    if actual_files != expected_files:
        raise RuntimeError(f"expected {expected_files} dimension files, got {actual_files}")
    if checks_frame["duplicate_model_rows"].sum() != 0:
        raise RuntimeError("one or more files contain duplicate model rows")
    if (checks_frame["model_rows"] == 0).any():
        raise RuntimeError("one or more group/dimension files are empty")

    expected_model_rows = {
        **{model: 24 * 4 for model in OLD_MODELS},
        "age_weighted": 24 * 4,
        "fast_slow_slow": 18 * 4,
    }
    if total_rows_by_model != expected_model_rows:
        raise RuntimeError(
            f"model coverage mismatch: actual={total_rows_by_model}, expected={expected_model_rows}"
        )

    print(f"output_root={OUTPUT_ROOT}")
    print(f"groups={len(groups)}")
    print(f"dimension_files={actual_files}")
    for model in MODEL_ORDER:
        print(f"{model}_dimension_rows={total_rows_by_model[model]}")


if __name__ == "__main__":
    main()
