from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BETA_ROOT = PROJECT_ROOT / "documents" / "01_Beta模型研究"
BETA_DATA_ROOT = BETA_ROOT / "02_评估数据"
OLD_GROUP_ROOT = BETA_DATA_ROOT / "Beta模型四维评估分组"
NEW_RESULT_ROOT = BETA_DATA_ROOT / "4三维度评价+2回测" / "四维评估结果"
OUTPUT_ROOT = BETA_DATA_ROOT / "Beta模型全部模型评估结果"
REPORT_PATH = BETA_ROOT / "03_评估报告" / "Beta模型全部模型四维评估报告.md"

MODEL_ORDER = [
    "rolling_ols_price",
    "tls_price",
    "tls_return",
    "dols_price",
    "ewls_price",
    "huber_price",
    "huber_return",
    "age_weighted_price",
    "age_weighted_return",
    "fast_slow_price",
    "residual_weighted_return",
]
MODEL_DISPLAY = {
    "rolling_ols_price": "Rolling OLS Price",
    "tls_price": "TLS Price",
    "dols_price": "DOLS Price",
    "ewls_price": "EWLS Price",
    "huber_price": "Huber Price",
    "age_weighted_price": "Age-Weighted Price",
    "fast_slow_price": "Fast-Slow Price",
    "age_weighted_return": "Age-Weighted Return",
    "residual_weighted_return": "Residual-Weighted Return",
    "tls_return": "TLS Return",
    "huber_return": "Huber Return",
}
OLD_MODEL_MAPPING = {
    "rolling_ols": "rolling_ols_price",
    "tls": "tls_price",
    "dols": "dols_price",
    "ewls": "ewls_price",
    "huber": "huber_price",
    "age_weighted": "age_weighted_return",
    "fast_slow_slow": "residual_weighted_return",
}
NEW_MODEL_MAPPING = {
    "age_weighted": "age_weighted_price",
    "fast_slow": "fast_slow_price",
    "tls_return": "tls_return",
    "huber_return": "huber_return",
}


@dataclass(frozen=True)
class Metric:
    column: str
    label: str
    direction: str
    transform: str = "identity"
    display_format: str = "decimal"


DIMENSIONS = {
    "01_残差统计性质": [
        Metric("joint_pass_rate", "样本外联合平稳通过率", "HIGH", display_format="percentage"),
        Metric("abs_mean_drift", "残差均值绝对漂移", "LOW", display_format="compact"),
        Metric("std_ratio", "标准差比率偏离1程度", "LOW", "distance_from_1", "percentage"),
    ],
    "02_均值回复能力": [
        Metric("half_life_min", "Half-life", "LOW", display_format="minutes"),
        Metric("repair_rate_300m", "300分钟内回归成功率", "HIGH", display_format="percentage"),
        Metric("repair_time_300m", "中位回归时间", "LOW", display_format="minutes"),
        Metric("max_adverse_z_300m", "最大不利Z偏离", "LOW", display_format="z"),
    ],
    "03_Beta稳定性": [
        Metric("relative_beta_change_median", "Beta相对变化中位数", "LOW", display_format="percentage"),
        Metric("relative_beta_change_p95", "Beta相对变化P95", "LOW", display_format="percentage"),
    ],
    "04_回测结果": [
        Metric("total_return", "总收益率", "HIGH", display_format="percentage"),
        Metric("annualized_return", "复合年化收益率（CAGR）", "HIGH", display_format="percentage"),
        Metric("sharpe", "Sharpe", "HIGH"),
        Metric("calmar", "Calmar", "HIGH"),
        Metric("max_drawdown", "最大回撤绝对值", "LOW", "absolute", "percentage"),
    ],
}

OLD_METRIC_MAPPING = {
    "01_残差统计性质": {
        "oos_joint_pass_rate_median": "joint_pass_rate",
        "abs_mean_drift_median": "abs_mean_drift",
        "std_ratio_oos_over_formation_median": "std_ratio",
    },
    "02_均值回复能力": {
        "half_life_median_min": "half_life_min",
        "reversion_hit_rate_median": "repair_rate_300m",
        "reversion_time_median_min": "repair_time_300m",
        "max_adverse_z_median": "max_adverse_z_300m",
    },
    "03_Beta稳定性": {
        "relative_beta_change_median": "relative_beta_change_median",
        "relative_beta_change_p95_median": "relative_beta_change_p95",
    },
    "04_回测结果": {
        "total_return": "total_return",
        "annualized_return": "annualized_return",
        "sharpe": "sharpe",
        "calmar": "calmar",
        "max_drawdown": "max_drawdown",
    },
}

IDENTITY = [
    "group_id",
    "lookback_days",
    "update_days",
    "entry_z",
    "exit_z",
    "model",
    "model_display",
    "source_report",
]


def _group_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["lookback_days"].astype(int).astype(str)
        + "|"
        + frame["update_days"].astype(int).astype(str)
        + "|"
        + frame["entry_z"].astype(float).astype(str)
        + "|"
        + frame["exit_z"].astype(float).astype(str)
    )


def _canonical_groups() -> pd.DataFrame:
    groups = pd.read_csv(OLD_GROUP_ROOT / "分组索引.csv")
    return pd.DataFrame(
        {
            "group_id": groups["group_id"],
            "lookback_days": groups["model_lookback_bars"] / 1440,
            "update_days": groups["model_update_interval_bars"] / 1440,
            "entry_z": groups["entry_z"],
            "exit_z": groups["exit_z"],
            "folder_name": groups["folder_name"],
            "result_file_name": groups["result_file_name"],
        }
    )


def _old_dimension_rows(groups: pd.DataFrame, dimension: str) -> pd.DataFrame:
    frames = []
    mapping = OLD_METRIC_MAPPING[dimension]
    for _, group in groups.iterrows():
        path = OLD_GROUP_ROOT / group["folder_name"] / dimension / group["result_file_name"]
        source = pd.read_csv(path)
        source = source[source["model"].isin(OLD_MODEL_MAPPING)].copy()
        source["model"] = source["model"].map(OLD_MODEL_MAPPING)
        source["model_display"] = source["model"].map(MODEL_DISPLAY)
        source["lookback_days"] = source["model_lookback_bars"] / 1440
        source["update_days"] = source["model_update_interval_bars"] / 1440
        source["source_report"] = "原七模型报告"
        source = source.rename(columns=mapping)
        metric_columns = [metric.column for metric in DIMENSIONS[dimension]]
        frames.append(source[IDENTITY + metric_columns])
    return pd.concat(frames, ignore_index=True)


def _new_dimension_rows(groups: pd.DataFrame, dimension: str) -> pd.DataFrame:
    filename = "回测指标组内汇总.csv" if dimension == "04_回测结果" else "三维指标组内汇总.csv"
    source = pd.read_csv(NEW_RESULT_ROOT / filename)
    source = source[source["model"].isin(NEW_MODEL_MAPPING)].copy()
    source["model"] = source["model"].map(NEW_MODEL_MAPPING)
    source["model_display"] = source["model"].map(MODEL_DISPLAY)
    source["source_report"] = "新增四模型报告"
    source["_group_key"] = _group_key(source)
    group_ids = groups.copy()
    group_ids["_group_key"] = _group_key(group_ids)
    source = source.drop(columns="group_id").merge(
        group_ids[["group_id", "_group_key"]], on="_group_key", how="left", validate="many_to_one"
    )
    metric_columns = [metric.column for metric in DIMENSIONS[dimension]]
    return source[IDENTITY + metric_columns]


def _comparison_values(values: pd.Series, transform: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if transform == "distance_from_1":
        return (numeric - 1.0).abs()
    if transform == "absolute":
        return numeric.abs()
    return numeric


def _standardized_score(values: pd.Series, direction: str) -> pd.Series:
    valid = values.notna()
    scores = pd.Series(np.nan, index=values.index, dtype=float)
    if not valid.any():
        return scores
    utility = values.loc[valid] if direction == "HIGH" else -values.loc[valid]
    standard_deviation = utility.std(ddof=1)
    if valid.sum() == 1 or not np.isfinite(standard_deviation) or standard_deviation <= 1e-12:
        scores.loc[valid] = 50.0
        return scores
    scores.loc[valid] = (50.0 + 10.0 * (utility - utility.mean()) / standard_deviation).clip(0, 100)
    return scores


def _score_dimension(frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
    result = frame.copy()
    score_columns = []
    for metric in DIMENSIONS[dimension]:
        comparison_column = f"comparison_{metric.column}"
        score_column = f"score_{metric.column}"
        result[comparison_column] = _comparison_values(result[metric.column], metric.transform)
        result[score_column] = result.groupby("group_id", group_keys=False)[comparison_column].apply(
            lambda values: _standardized_score(values, metric.direction)
        )
        score_columns.append(score_column)
    result["available_metric_count"] = result[score_columns].notna().sum(axis=1)
    result["dimension_score"] = result[score_columns].mean(axis=1, skipna=True)
    result["dimension"] = dimension
    return result


def _build_detail(groups: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for dimension in DIMENSIONS:
        source = pd.concat(
            [_old_dimension_rows(groups, dimension), _new_dimension_rows(groups, dimension)],
            ignore_index=True,
        )
        frames.append(_score_dimension(source, dimension))
    detail = pd.concat(frames, ignore_index=True)
    detail["_model_order"] = detail["model"].map(MODEL_ORDER.index)
    return detail.sort_values(["dimension", "group_id", "_model_order"]).drop(columns="_model_order")


def _build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    summary = (
        detail.groupby(["dimension", "model", "model_display"], sort=False)
        .agg(
            dimension_score=("dimension_score", "mean"),
            median_group_score=("dimension_score", "median"),
            score_std=("dimension_score", "std"),
            available_group_count=("group_id", "nunique"),
        )
        .reset_index()
    )
    summary["dimension_rank"] = summary.groupby("dimension")["dimension_score"].rank(
        method="min", ascending=False
    ).astype(int)
    summary["_model_order"] = summary["model"].map(MODEL_ORDER.index)
    return summary.sort_values(["dimension", "_model_order"]).drop(columns="_model_order")


def _format_value(value: float, display_format: str) -> str:
    if pd.isna(value):
        return ""
    if display_format == "percentage":
        return f"{value * 100:.2f}%"
    if display_format == "minutes":
        return f"{value:.2f}"
    if display_format == "compact":
        return f"{value:.6g}"
    if display_format == "z":
        return f"{value:.3f}"
    return f"{value:.4f}"


def _dimension_score_table(summary: pd.DataFrame) -> str:
    lines = [
        "| 模型 | 残差统计性质 | 均值回复能力 | Beta稳定性 | 回测结果 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        scores = {}
        for dimension in DIMENSIONS:
            row = summary[(summary["model"] == model) & (summary["dimension"] == dimension)]
            scores[dimension[:2]] = "" if row.empty else f"{row.iloc[0]['dimension_score']:.2f}"
        lines.append(
            f"| {MODEL_DISPLAY[model]} | {scores['01']} | {scores['02']} | {scores['03']} | {scores['04']} |"
        )
    return "\n".join(lines)


def _ranking_tables(summary: pd.DataFrame) -> str:
    sections = []
    for dimension in DIMENSIONS:
        selected = summary[summary["dimension"] == dimension].sort_values("dimension_rank")
        lines = [
            f"### {dimension[3:]}",
            "",
            "| 排名 | 模型 | 得分 | 有效组数 |",
            "|---:|---|---:|---:|",
        ]
        for _, row in selected.iterrows():
            lines.append(
                f"| {int(row['dimension_rank'])} | {row['model_display']} | "
                f"{row['dimension_score']:.2f} | {int(row['available_group_count'])} |"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _distribution_tables(detail: pd.DataFrame) -> str:
    sections = []
    for dimension, metrics in DIMENSIONS.items():
        metric_sections = []
        dimension_rows = detail[detail["dimension"] == dimension]
        for metric in metrics:
            comparison_column = f"comparison_{metric.column}"
            lines = [
                f"#### {metric.label}",
                "",
                "| 模型 | 均值 | 最小值–最大值 | P10–P90 | 有效组数 |",
                "|---|---:|---:|---:|---:|",
            ]
            for model in MODEL_ORDER:
                values = pd.to_numeric(
                    dimension_rows.loc[dimension_rows["model"] == model, comparison_column],
                    errors="coerce",
                ).dropna()
                if values.empty:
                    continue
                lines.append(
                    f"| {MODEL_DISPLAY[model]} | {_format_value(values.mean(), metric.display_format)} | "
                    f"{_format_value(values.min(), metric.display_format)}–"
                    f"{_format_value(values.max(), metric.display_format)} | "
                    f"{_format_value(values.quantile(0.10), metric.display_format)}–"
                    f"{_format_value(values.quantile(0.90), metric.display_format)} | {len(values)} |"
                )
            metric_sections.append("\n".join(lines))
        sections.append(f"### {dimension[3:]}\n\n" + "\n\n".join(metric_sections))
    return "\n\n".join(sections)


def _write_report(summary: pd.DataFrame, detail: pd.DataFrame) -> None:
    report = f"""# Beta模型全部模型四维评估报告

## 一、评估目的

本报告在同一套公共参数和公共指标下比较11个Beta估计模型。四个维度分别展示，不合成为最终综合得分，也不直接指定最终模型。

## 二、参评模型

为避免同一算法在不同回归数据上的结果混淆，模型名称明确标记为Price或Return：

| 模型家族 | Price版本 | Return版本 |
|---|---|---|
| Rolling OLS | Rolling OLS Price | 无 |
| TLS | TLS Price | TLS Return |
| DOLS | DOLS Price | 无 |
| EWLS | EWLS Price | 无 |
| Huber | Huber Price | Huber Return |
| Age-Weighted | Age-Weighted Price | Age-Weighted Return |
| Fast-Slow / Residual-Weighted | Fast-Slow Price | Residual-Weighted Return |

其中，原报告中的Age Weighted重命名为Age-Weighted Return，Fast-Slow-Slow重命名为Residual-Weighted Return；新增报告中的Age Weighted和Fast-Slow分别重命名为Age-Weighted Price和Fast-Slow Price。

## 三、公共参数与分组

| 公共参数 | 用途 | 取值 |
|---|---|---|
| `model_lookback_bars` | 估计Alpha和Beta使用的历史窗口 | 5D、10D、20D、30D、60D |
| `model_update_interval_bars` | Alpha和Beta重新估计周期 | 1D、3D；3D只与20D、30D、60D搭配 |
| `entry_z` | 残差Z-score开仓阈值 | 2.0、2.5、3.0 |
| `exit_z` | 残差回归后的平仓阈值 | 0.5 |

共形成24个公共参数组。只有四个公共参数完全相同的结果才在同一组内比较。同一模型在同一组内存在多个专属参数版本时，先对公共指标等权算术平均。Residual-Weighted Return只覆盖20D、30D、60D窗口，共18组；其余模型覆盖24组。

## 四、评价指标

| 维度 | 指标 | 方向 |
|---|---|---|
| 残差统计性质 | 样本外联合平稳通过率、残差均值绝对漂移、标准差比率偏离1程度 | 通过率越高越好，其余越低越好 |
| 均值回复能力 | Half-life、300分钟内回归成功率、中位回归时间、最大不利Z偏离 | 成功率越高越好，其余越低越好 |
| Beta稳定性 | Beta相对变化中位数、Beta相对变化P95 | 越低越好 |
| 回测结果 | 总收益率、复合年化收益率（CAGR）、Sharpe、Calmar、最大回撤绝对值 | 前四项越高越好，最大回撤绝对值越低越好 |

## 五、统一评分方法

各模型的维度分数统一基于每个公共参数组的原始指标，在11个模型之间重新计算：

```text
z_i = (u_i - mean(u)) / std(u)
指标得分 = max(0, min(100, 50 + 10 * z_i))
```

每项指标先统一为“越高越好”的效用方向。50分表示当前公共参数组内的平均水平。维度内指标等权，随后对模型覆盖的全部公共参数组等权平均。

## 六、结果使用说明

前三个维度使用跨Pair汇总指标。回测维度的源数据同时包含跨Pair中位数和多Pair组合级指标，统计层级并不完全一致，因此回测得分适合判断方向，不应脱离原始收益、风险指标单独使用。

## 七、四维得分

{_dimension_score_table(summary)}

## 八、分维度排序

{_ranking_tables(summary)}

## 九、各指标原始值的跨组分布

以下统计使用模型在公共参数组中的实际指标值。每个公共参数组权重相同，模型专属参数先在组内等权平均；缺失值不按0处理。“最小值–最大值”表示完整范围，“P10–P90”表示中间80%的取值范围。

{_distribution_tables(detail)}

---

报告生成日期：2026-08-26
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def _validate(groups: pd.DataFrame, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    if len(groups) != 24:
        raise RuntimeError(f"expected 24 groups, got {len(groups)}")
    expected_per_dimension = 10 * 24 + 18
    for dimension in DIMENSIONS:
        rows = detail[detail["dimension"] == dimension]
        if len(rows) != expected_per_dimension:
            raise RuntimeError(f"{dimension} expected {expected_per_dimension} rows, got {len(rows)}")
        if rows.duplicated(["group_id", "model"]).any():
            raise RuntimeError(f"{dimension} contains duplicate model/group rows")
        if rows["dimension_score"].isna().any():
            raise RuntimeError(f"{dimension} contains missing scores")
    if len(summary) != len(MODEL_ORDER) * len(DIMENSIONS):
        raise RuntimeError(f"summary expected {len(MODEL_ORDER) * len(DIMENSIONS)} rows, got {len(summary)}")


def main() -> None:
    groups = _canonical_groups()
    detail = _build_detail(groups)
    summary = _build_summary(detail)
    _validate(groups, detail, summary)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    groups.to_csv(OUTPUT_ROOT / "公共参数分组.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUTPUT_ROOT / "全部模型四维评分明细.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_ROOT / "全部模型四维得分.csv", index=False, encoding="utf-8-sig")
    _write_report(summary, detail)

    print(f"groups={len(groups)}")
    print(f"models={len(MODEL_ORDER)}")
    print(f"detail_rows={len(detail)}")
    print(f"summary_rows={len(summary)}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
