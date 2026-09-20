from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BETA_ROOT = PROJECT_ROOT / "documents" / "01_Beta模型研究"
BETA_DATA_ROOT = BETA_ROOT / "02_评估数据"
GROUP_ROOT = BETA_DATA_ROOT / "Beta模型四维评估分组"
OUTPUT_ROOT = BETA_DATA_ROOT / "Beta模型评估结果"
REPORT_PATH = BETA_ROOT / "03_评估报告" / "Beta模型综合评估报告.md"

MODEL_ORDER = [
    "rolling_ols",
    "tls",
    "dols",
    "ewls",
    "huber",
]


@dataclass(frozen=True)
class Metric:
    column: str
    label: str
    direction: str
    transform: str = "identity"
    display_format: str = "decimal"


DIMENSIONS = {
    "01_残差统计性质": {
        "label": "残差统计性质",
        "metrics": [
            Metric(
                "oos_joint_pass_rate_median",
                "样本外联合平稳通过率",
                "HIGH",
                display_format="percentage",
            ),
            Metric(
                "abs_mean_drift_median",
                "残差均值绝对漂移",
                "LOW",
                display_format="compact",
            ),
            Metric(
                "std_ratio_oos_over_formation_median",
                "标准差比率偏离1程度",
                "LOW",
                "distance_from_1",
                "percentage",
            ),
        ],
    },
    "02_均值回复能力": {
        "label": "均值回复能力",
        "metrics": [
            Metric(
                "half_life_median_min",
                "Half-life",
                "LOW",
                display_format="minutes",
            ),
            Metric(
                "reversion_hit_rate_median",
                "300分钟内回归成功率",
                "HIGH",
                display_format="percentage",
            ),
            Metric(
                "reversion_time_median_min",
                "中位回归时间",
                "LOW",
                display_format="minutes",
            ),
            Metric("max_adverse_z_median", "最大不利Z偏离", "LOW", display_format="z"),
        ],
    },
    "03_Beta稳定性": {
        "label": "Beta稳定性",
        "metrics": [
            Metric(
                "relative_beta_change_median",
                "Beta相对变化中位数",
                "LOW",
                display_format="percentage",
            ),
            Metric(
                "relative_beta_change_p95_median",
                "Beta相对变化P95",
                "LOW",
                display_format="percentage",
            ),
        ],
    },
    "04_回测结果": {
        "label": "回测结果",
        "metrics": [
            Metric("total_return", "总收益率", "HIGH", display_format="percentage"),
            Metric("annualized_return", "复合年化收益率（CAGR）", "HIGH", display_format="percentage"),
            Metric("sharpe", "Sharpe", "HIGH"),
            Metric("calmar", "Calmar", "HIGH"),
            Metric("max_drawdown", "最大回撤绝对值", "LOW", "absolute", "percentage"),
        ],
    },
}

DIMENSION_WEIGHTS = {
    "01": 2.0 / 15.0,
    "02": 2.0 / 15.0,
    "03": 2.0 / 15.0,
    "04": 0.60,
}


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
    count = int(valid.sum())
    if count == 0:
        return scores
    utility = values.loc[valid] if direction == "HIGH" else -values.loc[valid]
    standard_deviation = utility.std(ddof=1)
    if count == 1 or not np.isfinite(standard_deviation) or standard_deviation <= 1e-12:
        scores.loc[valid] = 50.0
        return scores
    z_scores = (utility - utility.mean()) / standard_deviation
    scores.loc[valid] = (50.0 + 10.0 * z_scores).clip(lower=0.0, upper=100.0)
    return scores


def _load_group_dimension(group_row: pd.Series, dimension: str) -> pd.DataFrame:
    path = (
        GROUP_ROOT
        / group_row["folder_name"]
        / dimension
        / group_row["result_file_name"]
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _score_dimension(frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
    definition = DIMENSIONS[dimension]
    identity = [
        "group_id",
        "model",
        "model_display",
        "model_lookback_bars",
        "model_update_interval_bars",
        "entry_z",
        "exit_z",
        "variant_count",
        "variant_weight",
        "aggregation_method",
    ]
    result = frame[identity].copy()
    score_columns = []
    for metric in definition["metrics"]:
        if metric.column not in frame.columns:
            raise KeyError(f"{dimension} missing metric column: {metric.column}")
        raw_column = f"raw_{metric.column}"
        comparison_column = f"comparison_{metric.column}"
        score_column = f"score_{metric.column}"
        result[raw_column] = pd.to_numeric(frame[metric.column], errors="coerce")
        result[comparison_column] = _comparison_values(frame[metric.column], metric.transform)
        result[score_column] = _standardized_score(
            result[comparison_column], metric.direction
        )
        score_columns.append(score_column)

    prefix = dimension[:2]
    result[f"{prefix}_available_metric_count"] = result[score_columns].notna().sum(axis=1)
    result[f"{prefix}_dimension_score"] = result[score_columns].mean(axis=1, skipna=True)
    return result


def _merge_group_scores(group_row: pd.Series) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    identity = [
        "group_id",
        "model",
        "model_display",
        "model_lookback_bars",
        "model_update_interval_bars",
        "entry_z",
        "exit_z",
    ]
    for dimension in DIMENSIONS:
        scored = _score_dimension(_load_group_dimension(group_row, dimension), dimension)
        if merged is None:
            merged = scored
            continue
        duplicate_metadata = [
            column
            for column in ["variant_count", "variant_weight", "aggregation_method"]
            if column in scored.columns
        ]
        scored = scored.rename(
            columns={column: f"{dimension[:2]}_{column}" for column in duplicate_metadata}
        )
        merged = merged.merge(scored, on=identity, how="outer", validate="one_to_one")

    if merged is None:
        raise RuntimeError(f"no dimension data for {group_row['group_id']}")
    dimension_columns = [f"{name[:2]}_dimension_score" for name in DIMENSIONS]
    merged["available_dimension_count"] = merged[dimension_columns].notna().sum(axis=1)
    weights = pd.Series(
        {column: DIMENSION_WEIGHTS[column[:2]] for column in dimension_columns}
    )
    available_weights = merged[dimension_columns].notna().mul(weights).sum(axis=1)
    merged["group_score"] = (
        merged[dimension_columns].mul(weights).sum(axis=1, min_count=1)
        / available_weights
    )
    merged["group_rank"] = merged["group_score"].rank(
        method="min", ascending=False
    ).astype("Int64")
    merged["is_group_winner"] = merged["group_rank"] == 1
    merged["is_group_top_two"] = merged["group_rank"] <= 2
    return merged


def _build_detail() -> pd.DataFrame:
    groups = pd.read_csv(GROUP_ROOT / "分组索引.csv")
    detail = pd.concat(
        [_merge_group_scores(group_row) for _, group_row in groups.iterrows()],
        ignore_index=True,
    )
    detail["_model_order"] = detail["model"].map(MODEL_ORDER.index)
    detail = detail.sort_values(
        ["group_id", "group_rank", "_model_order"], kind="stable"
    ).drop(columns="_model_order")
    return detail


def _build_ranking(detail: pd.DataFrame) -> pd.DataFrame:
    dimension_columns = [f"{name[:2]}_dimension_score" for name in DIMENSIONS]
    grouped = detail.groupby(["model", "model_display"], sort=False)
    ranking = grouped.agg(
        available_group_count=("group_id", "nunique"),
        overall_score=("group_score", "mean"),
        median_group_score=("group_score", "median"),
        group_score_std=("group_score", "std"),
        worst_group_score=("group_score", "min"),
        group_winner_count=("is_group_winner", "sum"),
        group_top_two_count=("is_group_top_two", "sum"),
    ).reset_index()
    ranking["group_top_two_rate"] = (
        ranking["group_top_two_count"] / ranking["available_group_count"]
    )
    for column in dimension_columns:
        dimension_average = grouped[column].mean().rename(column).reset_index()
        ranking = ranking.merge(
            dimension_average,
            on=["model", "model_display"],
            how="left",
            validate="one_to_one",
        )
    ranking["overall_rank"] = ranking["overall_score"].rank(
        method="min", ascending=False
    ).astype(int)
    ranking["_model_order"] = ranking["model"].map(MODEL_ORDER.index)
    ranking = ranking.sort_values(
        ["overall_rank", "_model_order"], kind="stable"
    ).drop(columns="_model_order")
    return ranking[
        [
            "overall_rank",
            "model",
            "model_display",
            "overall_score",
            *dimension_columns,
            "median_group_score",
            "group_score_std",
            "worst_group_score",
            "group_winner_count",
            "group_top_two_count",
            "group_top_two_rate",
            "available_group_count",
        ]
    ]


def _format_score(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2f}"


def _markdown_dimension_table(ranking: pd.DataFrame) -> str:
    lines = [
        "| 模型 | 残差统计性质 | 均值回复能力 | Beta稳定性 | 回测结果 |",
        "|---|---:|---:|---:|---:|",
    ]
    ordered = ranking.assign(
        _model_order=ranking["model"].map(MODEL_ORDER.index)
    ).sort_values("_model_order")
    for _, row in ordered.iterrows():
        lines.append(
            "| {model} | {d1} | {d2} | {d3} | {d4} |".format(
                model=row["model_display"],
                d1=_format_score(row["01_dimension_score"]),
                d2=_format_score(row["02_dimension_score"]),
                d3=_format_score(row["03_dimension_score"]),
                d4=_format_score(row["04_dimension_score"]),
            )
        )
    return "\n".join(lines)


def _markdown_dimension_rankings(ranking: pd.DataFrame) -> str:
    dimensions = {
        "残差统计性质": "01_dimension_score",
        "均值回复能力": "02_dimension_score",
        "Beta稳定性": "03_dimension_score",
        "回测结果": "04_dimension_score",
    }
    sections = []
    for label, column in dimensions.items():
        ordered = ranking.sort_values(column, ascending=False, kind="stable")
        lines = [
            f"### {label}",
            "",
            "| 排名 | 模型 | 得分 |",
            "|---:|---|---:|",
        ]
        for position, (_, row) in enumerate(ordered.iterrows(), start=1):
            lines.append(
                f"| {position} | {row['model_display']} | {_format_score(row[column])} |"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _format_metric_value(value: float, display_format: str) -> str:
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


def _markdown_metric_distributions(detail: pd.DataFrame) -> str:
    sections = []
    for dimension, definition in DIMENSIONS.items():
        metric_sections = []
        for metric in definition["metrics"]:
            comparison_column = f"comparison_{metric.column}"
            lines = [
                f"#### {metric.label}",
                "",
                "| 模型 | 均值 | 最小值–最大值 | P10–P90 | 有效组数 |",
                "|---|---:|---:|---:|---:|",
            ]
            for model in MODEL_ORDER:
                model_rows = detail.loc[detail["model"] == model]
                values = pd.to_numeric(
                    model_rows[comparison_column], errors="coerce"
                ).dropna()
                if values.empty:
                    continue
                display_name = str(model_rows["model_display"].iloc[0])
                mean_value = _format_metric_value(values.mean(), metric.display_format)
                minimum = _format_metric_value(values.min(), metric.display_format)
                maximum = _format_metric_value(values.max(), metric.display_format)
                p10 = _format_metric_value(values.quantile(0.10), metric.display_format)
                p90 = _format_metric_value(values.quantile(0.90), metric.display_format)
                lines.append(
                    f"| {display_name} | {mean_value} | {minimum}–{maximum} | "
                    f"{p10}–{p90} | {len(values)} |"
                )
            metric_sections.append("\n".join(lines))
        sections.append(
            f"### {definition['label']}\n\n" + "\n\n".join(metric_sections)
        )
    return "\n\n".join(sections)


def _write_report(ranking: pd.DataFrame, detail: pd.DataFrame) -> None:
    report = rf"""# Beta模型四维评估报告

## 一、评估目的

本报告分别展示参评Beta估计模型在残差统计性质、均值回复能力、Beta稳定性和回测结果四个维度上的表现。四个维度不合成为综合得分，本报告不对模型作最终选择。

统一回测区间为：`2025-01-01` 至 `2026-07-31`。

## 二、参评模型

本次评估包含 Rolling OLS、TLS、DOLS、EWLS 和 Huber。

## 三、公共参数与试验分组

为了保证不同 Beta 模型在相同条件下比较，本次评估使用以下四个公共参数进行分组：

| 公共参数 | 用途 | 本次选择 |
|---|---|---|
| `model_lookback_bars` | 决定每次估计 Alpha 和 Beta 使用多少根历史 K 线，也就是模型观察历史关系的窗口长度 | 5D、10D、20D、30D、60D；按 1 分钟 K 线分别为 7200、14400、28800、43200、86400 bars |
| `model_update_interval_bars` | 决定经过多少根 K 线重新估计一次 Alpha 和 Beta | 1D、3D；按 1 分钟 K 线分别为 1440、4320 bars |
| `entry_z` | 决定残差 Z-score 偏离到什么程度时允许开仓；绝对值达到该阈值时产生入场条件 | 2.0、2.5、3.0 |
| `exit_z` | 决定残差从极端区域回归到什么程度时平仓 | 固定为 0.5 |

### 参数作用说明

`model_lookback_bars` 控制 Beta 估计使用的历史长度。窗口越短，模型越侧重近期关系；窗口越长，估计包含的历史信息越多。该参数只规定数据范围，不规定窗口内每个观测的权重，具体权重仍由各 Beta 模型决定。

`model_update_interval_bars` 控制 Beta 的重新估计频率。1D 表示每经过 1 天数据到达一次模型更新点，3D 表示每经过 3 天数据到达一次更新点。它不改变每次估计使用的窗口长度。

`entry_z` 控制开仓所需的偏离程度。2.0、2.5 和 3.0 分别代表残差距离模型窗口内残差均值至少 2、2.5 和 3 个残差标准差。阈值越高，只有更极端的偏离才会触发开仓。

`exit_z` 控制均值回复后的平仓位置。本次统一固定为 0.5，使所有模型都在残差回到距离均值 0.5 个标准差以内时使用相同的平仓口径。

### 参数组合

1D 更新周期可以与全部五种模型窗口组合：

```text
5D + 1D
10D + 1D
20D + 1D
30D + 1D
60D + 1D
```

3D 更新周期只与较长的三种模型窗口组合：

```text
20D + 3D
30D + 3D
60D + 3D
```

每一种窗口与更新周期组合再分别搭配三个 `entry_z`，`exit_z` 始终为 0.5。因此公共参数试验组数量为：

```text
1D 更新：5 个窗口 * 3 个 entry_z = 15 组
3D 更新：3 个窗口 * 3 个 entry_z = 9 组
合计：15 + 9 = 24 组
```

只有四个公共参数完全相同的结果才进入同一个试验组，在组内比较不同 Beta 模型。同一模型如果还有自身专属参数，则先对这些专属参数版本的公共指标进行等权平均，再作为该模型在当前公共参数组中的结果。

## 四、评价体系

每个维度单独计算，同一维度内部的指标等权。

本次评分共使用14个指标：

| 维度 | 指标 | 计算含义 | 评分方向 |
|---|---|---|---|
| 残差统计性质 | 样本外联合平稳通过率 | 对每个Pair统计样本外残差同时通过ADF和KPSS检验的比例，再汇总为模型指标 | 越高越好 |
| 残差统计性质 | 残差均值绝对漂移 | 样本外残差均值与形成期残差均值之差的绝对值 | 越低越好 |
| 残差统计性质 | 标准差比率偏离1程度 | `abs(样本外残差标准差 / 形成期残差标准差 - 1)` | 越低越好 |
| 均值回复能力 | Half-life | 根据残差AR(1)系数计算偏离衰减一半所需的分钟数，计算式为 `-ln(2) / ln(phi)` | 越低越好 |
| 均值回复能力 | 300分钟内回归成功率 | 入场事件在300分钟内回到退出阈值的比例 | 越高越好 |
| 均值回复能力 | 中位回归时间 | 成功回归事件从入场到回到退出阈值所需分钟数的中位数 | 越低越好 |
| 均值回复能力 | 最大不利Z偏离 | 入场后、回归前，Z-score继续向不利方向偏离的最大幅度 | 越低越好 |
| Beta稳定性 | Beta相对变化中位数 | 相邻两次Beta相对变化绝对值的中位数 | 越低越好 |
| Beta稳定性 | Beta相对变化P95 | 相邻两次Beta相对变化绝对值的95%分位数 | 越低越好 |
| 回测结果 | 总收益率 | `期末权益 / 期初权益 - 1` | 越高越好 |
| 回测结果 | 复合年化收益率（CAGR） | 根据期初、期末权益和实际回测期限计算复合年化收益率 | 越高越好 |
| 回测结果 | Sharpe | 年化平均收益除以年化收益波动率 | 越高越好 |
| 回测结果 | Calmar | 复合年化收益率（CAGR）除以最大回撤绝对值 | 越高越好 |
| 回测结果 | 最大回撤绝对值 | 回测权益曲线最大回撤的绝对值 | 越低越好 |

同一模型存在多组专属参数时，先在模型内部对同一指标等权平均。随后在每个公共试验组内，将指标统一转换为“越高越好”的效用值：原本越高越好的指标取原值，原本越低越好的指标取相反数。再计算组内标准分数：

```text
z_i = (u_i - mean(u)) / std(u)
指标得分 S_i = max(0, min(100, 50 + 10 * z_i))
```

其中，`u_i`是模型经过方向转换后的指标值，`mean(u)`是同组模型的指标平均值，`std(u)`是同组模型的指标标准差。

指标等于组内平均值时得50分，高于组内平均值一个标准差时得60分，低于组内平均值一个标准差时得40分。若同组所有模型的某项指标完全相同，则该指标统一得50分。

同一维度内的指标分数等权平均形成维度分数，再将模型在其全部可用试验组中的维度分数等权平均。整个过程评价模型在各维度上的整体表现，不挑选单一参数版本。

## 五、四维得分

50分代表组内平均水平，高于50分表示整体高于组内平均水平，低于50分表示整体低于组内平均水平。

{_markdown_dimension_table(ranking)}

## 六、分维度排序

{_markdown_dimension_rankings(ranking)}

## 七、各指标原始值的跨组分布

以下统计使用每个模型在各公共参数组中的实际指标值，不使用T-score。每个公共参数组权重相同；同一模型在同一公共参数组中存在多个专属参数版本时，先在组内等权平均，再进行跨组统计。

“最小值–最大值”表示全部有效公共参数组的完整取值范围；“P10–P90”表示去除两端各10%极端结果后的中间80%取值范围。缺失组不按0处理。

{_markdown_metric_distributions(detail)}

---

评估日期：{date.today().isoformat()}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def _validate(detail: pd.DataFrame, ranking: pd.DataFrame) -> None:
    if set(ranking["model"]) != set(MODEL_ORDER):
        raise RuntimeError("final ranking does not contain all expected models")
    if detail.duplicated(["group_id", "model"]).any():
        raise RuntimeError("duplicate model rows found within a public group")
    if not detail["available_dimension_count"].eq(4).all():
        raise RuntimeError("one or more model/group rows do not cover all four dimensions")
    expected_metric_counts = {name[:2]: len(value["metrics"]) for name, value in DIMENSIONS.items()}
    for prefix, expected in expected_metric_counts.items():
        column = f"{prefix}_available_metric_count"
        if not detail[column].eq(expected).all():
            missing = detail.loc[~detail[column].eq(expected), ["group_id", "model", column]]
            raise RuntimeError(f"missing scored metrics:\n{missing.to_string(index=False)}")
    if not detail["group_score"].between(0, 100).all():
        raise RuntimeError("group scores must be between 0 and 100")
    if not ranking["overall_score"].between(0, 100).all():
        raise RuntimeError("overall scores must be between 0 and 100")
    score_columns = [column for column in detail.columns if column.startswith("score_")]
    group_metric_means = detail.groupby("group_id")[score_columns].mean()
    if not np.allclose(group_metric_means, 50.0, atol=1e-10):
        raise RuntimeError("one or more group-level metric T-scores do not average to 50")
    dimension_columns = [f"{name[:2]}_dimension_score" for name in DIMENSIONS]
    expected_group_scores = sum(
        detail[column] * DIMENSION_WEIGHTS[column[:2]]
        for column in dimension_columns
    )
    if not np.allclose(detail["group_score"], expected_group_scores, atol=1e-12):
        raise RuntimeError("group scores do not reconcile to the configured dimension weights")
    recomputed = detail.groupby("model")["group_score"].mean()
    reported = ranking.set_index("model")["overall_score"]
    if not np.allclose(recomputed.loc[reported.index], reported, atol=1e-12):
        raise RuntimeError("overall scores do not reconcile to group-score means")


def main() -> None:
    detail = _build_detail()
    ranking = _build_ranking(detail)
    _validate(detail, ranking)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    detail.to_csv(
        OUTPUT_ROOT / "模型四维评分明细.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ranking.to_csv(
        OUTPUT_ROOT / "模型综合排名.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_report(ranking, detail)

    print(f"detail_rows={len(detail)}")
    print(f"models={len(ranking)}")
    print(f"report={REPORT_PATH}")
    print(ranking[["overall_rank", "model_display", "overall_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
