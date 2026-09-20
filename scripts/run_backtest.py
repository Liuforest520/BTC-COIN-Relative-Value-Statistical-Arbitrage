from pathlib import Path
import sys
from argparse import ArgumentParser
from datetime import datetime

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.backtest import run_backtest
from core.modules.logger import logger, setup_logger
from core.modules.reporting import export_backtest_report


def main():
    parser = ArgumentParser()
    parser.add_argument("config_path", nargs="?", default="config/config.yaml")
    parser.add_argument(
        "--review-max-points",
        type=int,
        default=2000,
        help="兼容旧命令；交易复盘当前保留完整时间序列，不做降采样",
    )
    parser.add_argument("--review-max-pairs", type=int, default=20)
    parser.add_argument("--trade-review", action="store_true", default=True, help="export detailed trade review html")
    parser.add_argument(
        "--no-trade-review",
        action="store_false",
        dest="trade_review",
        help="skip the HTML trade review export (saves several GB and ~half the run time)",
    )
    parser.add_argument(
        "--no-curve-charts",
        action="store_false",
        dest="curve_charts",
        default=True,
        help="skip curve_charts/ (portfolio + per-pair equity images)",
    )
    args = parser.parse_args()

    config_path = Path(args.config_path)
    configured_run_name = _configured_run_name(config_path)
    log_name = configured_run_name or config_path.stem
    log_path = setup_logger(
        Path("logs") / "backtests" / f"{log_name}_{datetime.now():%Y%m%d_%H%M%S}.log",
        rotation=None,
        retention="14 days",
        enqueue=False,
    )
    logger.info("单回测日志: {}", log_path)
    result = run_backtest(config_path)
    equity_points = len(result.equity_curve)
    order_count = len(result.orders)
    trade_count = len(result.trades)
    report_dir = export_backtest_report(
        result,
        config_path,
        run_name=configured_run_name,
        include_trade_review=args.trade_review,
        review_max_points=args.review_max_points,
        review_max_pairs=args.review_max_pairs,
        include_curve_charts=args.curve_charts,
    )
    review_path = report_dir / "trade_review.html" if args.trade_review else None
    curve_dir = report_dir / "curve_charts" if args.curve_charts else None

    logger.info("backtest finished")
    logger.info("report saved: {}", report_dir)
    if curve_dir is not None:
        logger.info("curve charts saved: {}", curve_dir)
    if review_path is not None:
        logger.info("trade review saved: {}", review_path)
    logger.info("equity points: {}", equity_points)
    logger.info("orders: {}", order_count)
    logger.info("trades: {}", trade_count)

    for name, value in result.metrics.items():
        logger.info("metric {}={}", name, value)


def _configured_run_name(config_path: Path) -> str | None:
    """Return an optional stable report directory name from the YAML config."""
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    run_name = (raw.get("report") or {}).get("run_name")
    if run_name is None:
        return None
    run_name = str(run_name).strip()
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError(f"invalid report.run_name: {run_name!r}")
    return run_name


if __name__ == "__main__":
    main()
