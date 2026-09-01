from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = (
    PROJECT_ROOT
    / "documents"
    / "01_Beta模型研究"
    / "02_评估数据"
    / "4三维度评价+2回测"
)
THREE_DIMENSION_ROOT = SOURCE_ROOT / "三维度评价结果"
BACKTEST_ROOT = SOURCE_ROOT / "回测"
TLS_HUBER_BACKTEST = (
    PROJECT_ROOT
    / "results"
    / "sweep_tls_huber_return_intervals"
    / "sweep_results.csv"
)
OUTPUT_ROOT = SOURCE_ROOT / "四维评估结果"
REPORT_PATH = SOURCE_ROOT / "Beta模型四维评估报告.md"

MODEL_ORDER = ["age_weighted", "fast_slow", "tls_return", "huber_return"]
MODEL_DISPLAY = {
    "age_weighted": "Age Weighted",
    "fast_slow": "Fast-Slow",
    "tls_return": "TLS Return",
    "huber_return": "Huber Return",
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
        Metric(
            "std_ratio",
            "标准差比率偏离1程度",
            "LOW",
            transform="distance_from_1",
            display_format="percentage",
        ),
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
        Metric("max_drawdown", "最大回撤绝对值", "LOW", transform="absolute", display_format="percentage"),
    ],
}


def _number(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _canonical_groups() -> pd.DataFrame:
    combinations = [
        *[(lookback, 1) for lookback in (5, 10, 20, 30, 60)],
        *[(lookback, 3) for lookback in (20, 30, 60)],
    ]
    rows = []
    for lookback_days, update_days in combinations:
        for entry_z in (2.0, 2.5, 3.0):
            rows.append(
                {
                    "lookback_days": lookback_days,
                    "update_days": update_days,
                    "entry_z": entry_z,
                    "exit_z": 0.5,
                }
            )
    groups = pd.DataFrame(rows)
    groups.insert(0, "group_id", [f"G{index:03d}" for index in range(1, len(groups) + 1)])
    return groups


def _load_three_dimension_sources() -> dict[str, pd.DataFrame]:
    return {
        "age_weighted": pd.read_csv(THREE_DIMENSION_ROOT / "Age_weight_price.csv"),
        "fast_slow": pd.read_csv(THREE_DIMENSION_ROOT / "Fast_slow_price.csv"),
        "tls_return": pd.read_csv(THREE_DIMENSION_ROOT / "TLS_return.csv"),
        "huber_return": pd.read_csv(THREE_DIMENSION_ROOT / "Huber_return.csv"),
    }


def _three_dimension_variants(model: str, source: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    lookback_column = "formation_days" if model in {"age_weighted", "fast_slow"} else "lookback_days"
    entry_column = "event_z_threshold" if model in {"age_weighted", "fast_slow"} else "entry_z"
    exit_column = "exit_z" if model in {"age_weighted", "fast_slow"} else "repair_z"
    mask = (
        (source["sample_scope"] == "COMMON_60D")
        & (_number(source, lookback_column) == group["lookback_days"])
        & (_number(source, "update_days") == group["update_days"])
        & (_number(source, entry_column) == group["entry_z"])
        & (_number(source, exit_column) == group["exit_z"])
        & (_number(source, "horizon_min") == 300)
    )
    return source.loc[mask].copy()


def _mean(frame: pd.DataFrame, column: str) -> float:
    values = _number(frame, column).replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if not values.empty else np.nan


def _build_three_dimension_rows(groups: pd.DataFrame) -> pd.DataFrame:
    sources = _load_three_dimension_sources()
    rows = []
    mapping = {
        "joint_pass_rate": "d1_median_joint_pass_rate",
        "abs_mean_drift": "d1_median_abs_mean_drift",
        "std_ratio": "d1_median_std_ratio_oos_over_form",
        "half_life_min": "d1_median_half_life_min",
        "repair_rate_300m": "d2_median_pair_repair_rate",
        "repair_time_300m": "d2_median_pair_repair_time_min",
        "max_adverse_z_300m": "d2_median_pair_max_adverse_z",
        "relative_beta_change_median": "d3_median_relative_beta_change",
        "relative_beta_change_p95": "d3_p95_relative_beta_change",
    }
    for _, group in groups.iterrows():
        for model in MODEL_ORDER:
            variants = _three_dimension_variants(model, sources[model], group)
            if variants.empty:
                continue
            row = {
                **group.to_dict(),
                "model": model,
                "model_display": MODEL_DISPLAY[model],
                "three_dimension_variant_count": len(variants),
            }
            for target, source_column in mapping.items():
                row[target] = _mean(variants, source_column)
            rows.append(row)
    return pd.DataFrame(rows)


def _load_backtest_sources() -> dict[str, pd.DataFrame]:
    sweep = pd.read_csv(TLS_HUBER_BACKTEST)
    method_column = "setups.multi_pair_beta.pipeline.estimator.method"
    return {
        "age_weighted": pd.read_csv(BACKTEST_ROOT / "Age_weight_price回测.csv"),
        "fast_slow": pd.read_csv(BACKTEST_ROOT / "Fast_slow_price回测.csv"),
        "tls_return": sweep.loc[sweep[method_column] == "tls"].copy(),
        "huber_return": sweep.loc[sweep[method_column] == "huber"].copy(),
    }


def _backtest_variants(model: str, source: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    if model in {"age_weighted", "fast_slow"}:
        lookback_column = "formation_days" if model == "age_weighted" else "long_days"
        mask = (
            (source["sample_scope"] == "COMMON_60D")
            & (_number(source, lookback_column) == group["lookback_days"])
            & (_number(source, "update_days") == group["update_days"])
            & (_number(source, "entry_z_threshold") == group["entry_z"])
            & (_number(source, "exit_z") == group["exit_z"])
        )
        return source.loc[mask].copy()

    lookback_column = "setups.multi_pair_beta.pipeline.estimator.model_lookback_bars"
    update_column = "setups.multi_pair_beta.pipeline.estimator.model_update_interval_bars"
    entry_column = "setups.multi_pair_beta.pipeline.signal.entry_z"
    exit_column = "setups.multi_pair_beta.pipeline.signal.exit_z"
    mask = (
        (_number(source, lookback_column) == group["lookback_days"] * 1440)
        & (_number(source, update_column) == group["update_days"] * 1440)
        & (_number(source, entry_column) == group["entry_z"])
        & (_number(source, exit_column) == group["exit_z"])
    )
    return source.loc[mask].copy()


def _build_backtest_rows(groups: pd.DataFrame) -> pd.DataFrame:
    sources = _load_backtest_sources()
    return_mapping = {
        "total_return": "cross_pair_median_net_compound_return",
        "annualized_return": "cross_pair_median_net_annualized_return",
        "sharpe": "cross_pair_median_net_sharpe",
        "calmar": "cross_pair_median_net_calmar",
        "max_drawdown": "cross_pair_median_net_max_drawdown",
    }
    sweep_mapping = {
        "total_return": "total_return",
        "annualized_return": "annualized_return",
        "sharpe": "sharpe",
        "calmar": "calmar",
        "max_drawdown": "max_drawdown",
    }
    rows = []
    for _, group in groups.iterrows():
        for model in MODEL_ORDER:
            variants = _backtest_variants(model, sources[model], group)
            if variants.empty:
                continue
            mapping = return_mapping if model in {"age_weighted", "fast_slow"} else sweep_mapping
            row = {
                **group.to_dict(),
                "model": model,
                "model_display": MODEL_DISPLAY[model],
                "backtest_variant_count": len(variants),
                "backtest_statistical_level": (
                    "cross_pair_median" if model in {"age_weighted", "fast_slow"} else "portfolio"
                ),
            }
            for target, source_column in mapping.items():
                row[target] = _mean(variants, source_column)
            rows.append(row)
    return pd.DataFrame(rows)


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


def _build_scored_detail(three_dimension: pd.DataFrame, backtest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension in DIMENSIONS:
        source = backtest if dimension == "04_回测结果" else three_dimension
        identity = [
            "group_id",
            "lookback_days",
            "update_days",
            "entry_z",
            "exit_z",
            "model",
            "model_display",
        ]
        variant_columns = [column for column in source.columns if column.endswith("variant_count")]
        metric_columns = [metric.column for metric in DIMENSIONS[dimension]]
        rows.append(_score_dimension(source[identity + variant_columns + metric_columns], dimension))
    detail = pd.concat(rows, ignore_index=True)
    detail["_model_order"] = detail["model"].map(MODEL_ORDER.index)
    return detail.sort_values(["dimension", "group_id", "_model_order"]).drop(columns="_model_order")


def _build_dimension_summary(detail: pd.DataFrame) -> pd.DataFrame:
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
    return summary.sort_values(["dimension", "dimension_rank", "_model_order"]).drop(columns="_model_order")


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
    columns = {dimension: dimension[:2] for dimension in DIMENSIONS}
    lines = [
        "| 模型 | 残差统计性质 | 均值回复能力 | Beta稳定性 | 回测结果 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        values = {}
        for dimension, key in columns.items():
            selected = summary[(summary["model"] == model) & (summary["dimension"] == dimension)]
            values[key] = "" if selected.empty else f"{selected.iloc[0]['dimension_score']:.2f}"
        lines.append(
            f"| {MODEL_DISPLAY[model]} | {values['01']} | {values['02']} | {values['03']} | {values['04']} |"
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
    report = f"""# 四种Beta模型四维评估报告

## 一、评估目的

本报告比较 Age Weighted、Fast-Slow、TLS Return 和 Huber Return 在残差统计性质、均值回复能力、Beta稳定性和回测结果四个维度上的表现。四个维度分别展示，不合成为最终综合得分，也不直接指定最终模型。

## 二、公共参数与分组

所有结果按照以下四个公共参数分组：

| 参数 | 取值 |
|---|---|
| 模型窗口 | 5D、10D、20D、30D、60D |
| 更新周期 | 1D；3D只与20D、30D、60D搭配 |
| 开仓阈值 | 2.0、2.5、3.0 |
| 平仓阈值 | 0.5 |

共形成24个公共参数组。只有四个公共参数完全相同的结果才在同一组内比较。

同一模型在同一公共参数组下存在多个专属参数版本时，先对公共指标进行等权算术平均：Age Weighted整合不同年龄权重版本；Fast-Slow整合不同权重、短周期和偏离阈值版本；TLS Return与Huber Return整合1、60、300分钟收益率版本。

| 模型 | 三维评价每组版本数 | 回测每组版本数 | 等权整合内容 |
|---|---:|---:|---|
| Age Weighted | 3 | 3 | 年龄权重档位 |
| Fast-Slow | 1 | 9 | 回测中的权重档位、短周期和偏离阈值 |
| TLS Return | 3 | 3 | 1、60、300分钟收益率 |
| Huber Return | 3 | 3 | 1、60、300分钟收益率 |

## 三、评价指标

| 维度 | 指标 | 方向 |
|---|---|---|
| 残差统计性质 | 样本外联合平稳通过率、残差均值绝对漂移、标准差比率偏离1程度 | 通过率越高越好，其余越低越好 |
| 均值回复能力 | Half-life、300分钟内回归成功率、中位回归时间、最大不利Z偏离 | 成功率越高越好，其余越低越好 |
| Beta稳定性 | Beta相对变化中位数、Beta相对变化P95 | 越低越好 |
| 回测结果 | 总收益率、复合年化收益率（CAGR）、Sharpe、Calmar、最大回撤绝对值 | 前四项越高越好，最大回撤绝对值越低越好 |

三维评价统一使用`COMMON_60D`样本，并在均值回复维度固定使用300分钟观察期限。

## 四、评分方法

每个指标先统一为“越高越好”的效用方向，然后在同一个公共参数组内计算标准分数：

```text
z_i = (u_i - mean(u)) / std(u)
指标得分 = max(0, min(100, 50 + 10 * z_i))
```

50分表示该公共参数组内的平均水平。每个维度内部的指标等权，再将模型在24个公共参数组中的维度得分等权平均。

## 五、重要口径说明

前三个维度的四个模型均为跨Pair汇总指标，可以在同组内直接比较。

第四维中，Age Weighted和Fast-Slow源文件提供的是13个Pair指标的跨Pair中位数；TLS Return和Huber Return的Sweep文件提供的是多Pair组合级指标。报告按字段含义统一展示并计算了相对分数，但统计层级并不完全相同，因此第四维分数只能作为方向性参考，不应单独作为最终经济表现排名。源文件也没有共同携带可核验的统一回测起止时间。

## 六、四维得分

{_dimension_score_table(summary)}

### 结果解读

- 残差统计性质方面，Huber Return与TLS Return的联合平稳通过率、均值漂移和标准差比率整体优于另外两个模型。
- 均值回复能力方面，TLS Return与Huber Return的综合分数较高；其回归成功率和回归速度较好，但最大不利Z偏离也更大，因此需要结合原始指标而不是只看维度分数。
- Beta稳定性方面，Huber Return与TLS Return的Beta相对变化中位数和P95明显更低。
- 回测结果方面，Fast-Slow与Age Weighted的源文件指标较高，TLS Return与Huber Return受到1分钟收益率版本高换手和交易成本的显著拖累。该维度同时受到前述统计层级差异影响。

## 七、分维度排序

{_ranking_tables(summary)}

## 八、各指标原始值的跨组分布

以下统计使用每个模型在24个公共参数组中的实际指标值。每个公共参数组权重相同；模型专属参数先在组内等权平均。缺失值不按0处理。

“最小值–最大值”表示全部有效公共参数组的完整范围；“P10–P90”表示中间80%的取值范围。

{_distribution_tables(detail)}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def _validate(
    groups: pd.DataFrame,
    three_dimension: pd.DataFrame,
    backtest: pd.DataFrame,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    if len(groups) != 24:
        raise RuntimeError(f"expected 24 groups, got {len(groups)}")
    expected_model_group_rows = 24 * len(MODEL_ORDER)
    if len(three_dimension) != expected_model_group_rows:
        raise RuntimeError(
            f"three-dimension coverage mismatch: expected {expected_model_group_rows}, got {len(three_dimension)}"
        )
    if len(backtest) != expected_model_group_rows:
        raise RuntimeError(
            f"backtest coverage mismatch: expected {expected_model_group_rows}, got {len(backtest)}"
        )
    expected_detail_rows = expected_model_group_rows * len(DIMENSIONS)
    if len(detail) != expected_detail_rows:
        raise RuntimeError(f"detail row mismatch: expected {expected_detail_rows}, got {len(detail)}")
    if detail["dimension_score"].isna().any() or not detail["dimension_score"].between(0, 100).all():
        raise RuntimeError("dimension scores contain missing or out-of-range values")
    if len(summary) != len(MODEL_ORDER) * len(DIMENSIONS):
        raise RuntimeError("dimension summary coverage mismatch")


def main() -> None:
    groups = _canonical_groups()
    three_dimension = _build_three_dimension_rows(groups)
    backtest = _build_backtest_rows(groups)
    detail = _build_scored_detail(three_dimension, backtest)
    summary = _build_dimension_summary(detail)
    _validate(groups, three_dimension, backtest, detail, summary)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    groups.to_csv(OUTPUT_ROOT / "公共参数分组.csv", index=False, encoding="utf-8-sig")
    three_dimension.to_csv(OUTPUT_ROOT / "三维指标组内汇总.csv", index=False, encoding="utf-8-sig")
    backtest.to_csv(OUTPUT_ROOT / "回测指标组内汇总.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUTPUT_ROOT / "四维评分明细.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_ROOT / "模型四维得分.csv", index=False, encoding="utf-8-sig")
    checks = pd.DataFrame(
        [
            {"check": "公共参数组数", "actual": len(groups), "expected": 24, "passed": len(groups) == 24},
            {
                "check": "三维模型组记录数",
                "actual": len(three_dimension),
                "expected": 96,
                "passed": len(three_dimension) == 96,
            },
            {
                "check": "回测模型组记录数",
                "actual": len(backtest),
                "expected": 96,
                "passed": len(backtest) == 96,
            },
            {
                "check": "评分明细记录数",
                "actual": len(detail),
                "expected": 384,
                "passed": len(detail) == 384,
            },
            {
                "check": "评分缺失数",
                "actual": int(detail["dimension_score"].isna().sum()),
                "expected": 0,
                "passed": not detail["dimension_score"].isna().any(),
            },
        ]
    )
    checks.to_csv(OUTPUT_ROOT / "数据覆盖检查.csv", index=False, encoding="utf-8-sig")
    _write_report(summary, detail)

    print(f"groups={len(groups)}")
    print(f"three_dimension_rows={len(three_dimension)}")
    print(f"backtest_rows={len(backtest)}")
    print(f"detail_rows={len(detail)}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
