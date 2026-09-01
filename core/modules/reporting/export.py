from dataclasses import asdict, is_dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
import json
import shutil

import polars as pl

from core.modules.reporting.equity_summary import export_equity_summary_html
from core.modules.logger import logger
from core.modules.reporting.pair_summary import (
    build_pair_summary,
    compact_pair_summary_markdown,
    pair_defs_from_config,
    write_compact_pair_summary_csv,
)


KEY_METRIC_ROWS = [
    ("业绩", "复合年化收益（CAGR）", "annualized_return"),
    ("业绩", "年化波动", "annualized_volatility"),
    ("业绩", "夏普", "sharpe"),
    ("业绩", "卡玛", "calmar"),
    ("业绩", "最大回撤", "max_drawdown"),
    ("交易", "胜率", "win_rate"),
    ("交易", "盈亏比", "profit_loss_ratio"),
    ("交易", "平均持仓周期", "average_holding_minutes"),
    ("交易", "日均换手率", "daily_turnover"),
    ("成本", "手续费", "total_fee"),
    ("成本", "滑点", "total_slippage"),
    ("成本", "资金费率", "funding_fee"),
]


def export_backtest_report(
    result,
    config_path,
    output_root="results/backtests",
    run_name=None,
    include_trade_review=True,
    review_max_points=2000,
    review_max_pairs=20,
):
    config_path = Path(config_path)
    run_name = run_name or _run_name(result)
    output_dir = Path(output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config_output = output_dir / "config.yaml"
    if config_path.exists():
        shutil.copyfile(config_path, config_output)

    _write_json(output_dir / "metrics.json", result.metrics)
    _write_csv(output_dir / "metrics.csv", _dict_rows(result.metrics, "metric", "value"))
    _write_csv(output_dir / "key_metrics.csv", key_metric_rows(result.metrics))
    _write_csv(output_dir / "equity_curve.csv", result.equity_curve)
    _write_csv(output_dir / "position_curve.csv", result.position_curve)
    _write_csv(output_dir / "signal_curve.csv", result.signal_curve)
    _write_csv(output_dir / "orders.csv", _object_rows(result.orders))
    _write_csv(output_dir / "trades.csv", _object_rows(result.trades))
    _write_csv(output_dir / "funding_payments.csv", _object_rows(result.funding_payments))
    _write_csv(output_dir / "risk_history.csv", _object_rows(result.risk_history))
    _write_json(output_dir / "final_position_valuation.json", result.final_position_valuation)
    if getattr(result, 'pair_curve', None):
        _write_csv(output_dir / "pair_curve.csv", result.pair_curve)
    pair_metrics = _per_pair_metrics(result, config_output)
    if pair_metrics:
        write_compact_pair_summary_csv(output_dir / "pair_metrics.csv", pair_metrics)
        (output_dir / "pair_metrics_summary.md").write_text(
            compact_pair_summary_markdown(pair_metrics),
            encoding="utf-8",
        )

    return_attribution = return_attribution_rows(result)
    risk_attribution = risk_attribution_rows(result)
    _write_csv(output_dir / "return_attribution.csv", return_attribution)
    _write_csv(output_dir / "risk_attribution.csv", risk_attribution)

    export_equity_summary_html(result, output_dir / "portfolio_summary.html")

    _write_markdown_summary(output_dir / "summary.md", result, config_output, return_attribution, risk_attribution)
    if include_trade_review:
        try:
            _export_trade_review(output_dir, review_max_points, review_max_pairs)
        except Exception as exc:
            logger.exception("trade review export failed for {}: {}", output_dir, exc)
            (output_dir / "trade_review_error.txt").write_text(
                f"Trade review generation failed: {type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
    return output_dir


def _export_trade_review(output_dir, max_points, max_pairs):
    # Lazy import avoids coupling the base CSV exporter to the HTML module at
    # import time while keeping every full report on one consistent path.
    from core.modules.reporting.trade_review import export_trade_review_html

    return export_trade_review_html(
        output_dir,
        max_points=max_points,
        max_pairs=max_pairs,
    )


def key_metric_rows(metrics):
    rows = []
    for category, name, key in KEY_METRIC_ROWS:
        value = metrics.get(key)
        if value is None:
            continue
        rows.append(
            {
                "category": category,
                "metric": name,
                "key": key,
                "value": value,
            }
        )
    return rows


def return_attribution_rows(result):
    metrics = result.metrics
    final_equity = metrics.get("final_equity") or 0.0
    initial_equity = metrics.get("initial_equity") or 0.0
    gross_pnl = final_equity - initial_equity
    fee = metrics.get("total_fee") or 0.0
    slippage = metrics.get("total_slippage") or 0.0
    funding = metrics.get("funding_fee") or 0.0
    trading_before_cost = gross_pnl + fee + slippage + funding
    return [
        {"item": "交易收益(成本前)", "value": trading_before_cost},
        {"item": "手续费", "value": -fee},
        {"item": "滑点", "value": -slippage},
        {"item": "资金费率", "value": -funding},
        {"item": "净收益", "value": gross_pnl},
    ]


def risk_attribution_rows(result):
    metrics = result.metrics
    return [
        {"item": "最大回撤", "value": abs(metrics.get("max_drawdown") or 0.0)},
        {"item": "年化波动", "value": metrics.get("annualized_volatility") or 0.0},
    ]


def _write_markdown_summary(path, result, config_path, return_attribution, risk_attribution):
    lines = [
        "# 单次回测结果报告",
        "",
        f"- 配置文件: `{config_path.name}`",
        f"- 最终权益: `{result.metrics.get('final_equity')}`",
        f"- 总收益: `{result.metrics.get('total_return')}`",
        f"- Sharpe: `{result.metrics.get('sharpe')}`",
        f"- 最大回撤: `{result.metrics.get('max_drawdown')}`",
        "",
        "## 关键指标汇总表",
        "",
        "| 类别 | 指标 | 值 |",
        "|---|---|---:|",
    ]
    for row in key_metric_rows(result.metrics):
        lines.append(f"| {row['category']} | {row['metric']} | {row['value']} |")

    lines.extend(
        [
            "",
            "## 图表文件",
            "",
            "- `portfolio_summary.html`: 总资金曲线交互网页",
            "- `trade_review.html`: 组合交易复盘与单 Pair 分析入口",
            "",
            "## 明细文件",
            "",
            "- `orders.csv`: 每一次订单",
            "- `trades.csv`: 每一次成交",
            "- `position_curve.csv`: 每一分钟持仓情况",
            "- `signal_curve.csv`: 每一分钟信号、z-score、alpha/beta",
            "- `funding_payments.csv`: 资金费率扣费记录",
            "- `risk_history.csv`: 风控检查记录",
            "- `pair_metrics.csv`: 每个 pair 的交易笔数、胜率、收益率、总盈亏、回撤和持仓时间",
            "- `pair_metrics_summary.md`: 每个 pair 的中文核心指标汇总表",
            "- `return_attribution.csv`: 收益归因",
            "- `risk_attribution.csv`: 风险归因",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _per_pair_metrics(result, config_path: Path) -> list[dict]:
    """Compute per-pair PnL/win-rate summary from filled trades."""
    return build_pair_summary(
        trades=getattr(result, "trades", None),
        position_curve=getattr(result, "position_curve", None),
        funding_payments=getattr(result, "funding_payments", None),
        pair_defs=pair_defs_from_config(Path(config_path)),
        initial_equity=(getattr(result, "metrics", {}) or {}).get("initial_equity"),
    )


def _run_name(result):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_equity = result.metrics.get("final_equity")
    if final_equity is None:
        return f"backtest_{timestamp}"
    return f"backtest_{timestamp}_equity_{final_equity:.2f}"


def _write_csv(path, rows):
    rows = list(rows or [])
    # Polars can union keys from sparse row dictionaries directly.  Avoid
    # expanding every minute-level compact signal row into a wide Python dict,
    # which otherwise multiplies report-export memory usage.
    frame = pl.from_dicts(rows, infer_schema_length=None) if rows else pl.DataFrame()
    if frame.is_empty():
        path.write_text("", encoding="utf-8")
        return
    frame.write_csv(path)


def _write_json(path, data):
    path.write_text(json.dumps(_json_safe(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _dict_rows(data, key_name, value_name):
    return [{key_name: key, value_name: value} for key, value in data.items()]


def _object_rows(items):
    rows = []
    for item in items:
        if is_dataclass(item):
            rows.append(_json_safe(asdict(item)))
        elif isinstance(item, dict):
            rows.append(_json_safe(item))
        else:
            rows.append(_json_safe(vars(item)))
    return rows


def _json_safe(value):
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if isfinite(value):
            return value
        if value > 0:
            return "Infinity"
        if value < 0:
            return "-Infinity"
        return None
    return value
