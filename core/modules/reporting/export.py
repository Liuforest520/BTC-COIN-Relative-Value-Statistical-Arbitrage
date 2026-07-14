from dataclasses import asdict, is_dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
import json
import shutil

import polars as pl

from core.modules.reporting.plots import (
    plot_attribution,
    plot_backtest_summary,
    plot_drawdown,
    plot_neutrality,
    plot_signal,
)
from core.modules.reporting.utils import normalize_rows


KEY_METRIC_ROWS = [
    ("业绩", "年化收益", "annualized_return"),
    ("业绩", "年化波动", "annualized_volatility"),
    ("业绩", "夏普", "sharpe"),
    ("业绩", "卡玛", "calmar"),
    ("业绩", "最大回撤", "max_drawdown"),
    ("交易", "胜率", "win_rate"),
    ("交易", "盈亏比", "profit_loss_ratio"),
    ("交易", "平均持仓周期", "average_holding_minutes"),
    ("交易", "日均换手率", "daily_turnover"),
    ("市场中性", "组合对 BTC 的 Beta", "portfolio_beta_btc"),
    ("市场中性", "组合对 COIN 的 Beta", "portfolio_beta_coin"),
    ("市场中性", "组合对 SPY 的 Beta", "portfolio_beta_spy"),
    ("市场中性", "组合对 QQQ 的 Beta", "portfolio_beta_qqq"),
    ("市场中性", "组合对 BTC 的相关性", "portfolio_corr_btc"),
    ("市场中性", "组合对 COIN 的相关性", "portfolio_corr_coin"),
    ("市场中性", "组合对 SPY 的相关性", "portfolio_corr_spy"),
    ("市场中性", "组合对 QQQ 的相关性", "portfolio_corr_qqq"),
    ("市场中性", "开仓对冲比例通过率", "hedge_ratio_pass_rate"),
    ("市场中性", "平均对冲比例偏离度", "average_hedge_ratio_deviation"),
    ("成本", "手续费", "total_fee"),
    ("成本", "滑点", "total_slippage"),
    ("成本", "资金费率", "funding_fee"),
]


def export_backtest_report(result, config_path, output_root="results/backtests", run_name=None):
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

    return_attribution = return_attribution_rows(result)
    risk_attribution = risk_attribution_rows(result)
    _write_csv(output_dir / "return_attribution.csv", return_attribution)
    _write_csv(output_dir / "risk_attribution.csv", risk_attribution)

    plot_backtest_summary(result, output_dir / "equity_position_summary.png")
    plot_drawdown(result, output_dir / "drawdown_curve.png")
    plot_signal(result, output_dir / "signal_visualization.png")
    plot_neutrality(result, output_dir / "market_neutrality.png")
    plot_attribution(return_attribution, risk_attribution, output_dir / "attribution.png")

    _write_markdown_summary(output_dir / "summary.md", result, config_output, return_attribution, risk_attribution)
    return output_dir


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
        {"item": "平均对冲比例偏离", "value": metrics.get("average_hedge_ratio_deviation") or 0.0},
        {"item": "最终净敞口比例", "value": abs(metrics.get("final_net_exposure_ratio") or 0.0)},
        {"item": "最终总敞口比例", "value": metrics.get("final_gross_exposure_ratio") or 0.0},
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
            "- `equity_position_summary.png`: 净值曲线、回撤区域、持仓比例",
            "- `drawdown_curve.png`: 回撤曲线",
            "- `signal_visualization.png`: 信号可视化",
            "- `market_neutrality.png`: 市场中性验证图",
            "- `attribution.png`: 收益归因、风险归因",
            "",
            "## 明细文件",
            "",
            "- `orders.csv`: 每一次订单",
            "- `trades.csv`: 每一次成交",
            "- `position_curve.csv`: 每一分钟持仓情况",
            "- `signal_curve.csv`: 每一分钟信号、z-score、alpha/beta",
            "- `funding_payments.csv`: 资金费率扣费记录",
            "- `risk_history.csv`: 风控检查记录",
            "- `return_attribution.csv`: 收益归因",
            "- `risk_attribution.csv`: 风险归因",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_name(result):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_equity = result.metrics.get("final_equity")
    if final_equity is None:
        return f"backtest_{timestamp}"
    return f"backtest_{timestamp}_equity_{final_equity:.2f}"


def _write_csv(path, rows):
    rows = normalize_rows(rows)
    frame = pl.DataFrame(rows, infer_schema_length=None)
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
