"""Run rolling natural-month walk-forward optimization and OOS backtest."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.backtest.walk_forward import run_walk_forward
from core.modules.logger import logger, setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="?", default="config/config.yaml")
    parser.add_argument("--output-root", default="results/walk_forward")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--optimizer", choices=("numba", "sweep"), default="numba")
    parser.add_argument(
        "--optimization-scope", choices=("pair", "portfolio"), default=None,
        help="Use pair or shared portfolio parameter optimization.",
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=None,
        help="Override portfolio Numba candidate batch size without editing YAML.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Walk-Forward training process count; 1 disables process parallelism.",
    )
    parser.add_argument(
        "--parameter-transition-mode",
        choices=("defer_until_flat", "close_at_boundary", "switch_immediately"),
        default=None,
        help="Override how OOS positions are handled when validation parameters change.",
    )
    parser.add_argument(
        "--trade-review",
        action="store_true",
        dest="trade_review",
        default=True,
        help="生成标准 Trade Review HTML（默认开启）",
    )
    parser.add_argument(
        "--no-trade-review",
        action="store_false",
        dest="trade_review",
        help="关闭 Trade Review HTML 导出，减少耗时和磁盘占用",
    )
    parser.add_argument(
        "--no-curve-charts",
        action="store_true",
        help="不生成组合及单 Pair 曲线 PNG",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="日志文件路径；默认写入 logs/walk_forward/<run_name>_<timestamp>.log",
    )
    args = parser.parse_args()
    run_name = args.run_name or Path(args.config_path).stem
    log_file = Path(args.log_file) if args.log_file else (
        Path("logs") / "walk_forward" /
        f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    full_log_path = setup_logger(log_file, rotation=None, retention=None)
    logger.info(
        "walk-forward command started config={} run_name={} optimizer={} candidate_batch_size={} workers={} parameter_transition_mode={} log_file={}",
        args.config_path,
        run_name,
        args.optimizer,
        args.candidate_batch_size,
        args.workers,
        args.parameter_transition_mode,
        full_log_path,
    )
    output_dir, result, _schedule = run_walk_forward(
        args.config_path,
        output_root=args.output_root,
        run_name=run_name,
        show_progress=args.show_progress,
        optimizer=args.optimizer,
        optimization_scope=args.optimization_scope,
        candidate_batch_size=args.candidate_batch_size,
        process_workers=args.workers,
        parameter_transition_mode=args.parameter_transition_mode,
        include_trade_review=args.trade_review,
        include_curve_charts=not args.no_curve_charts,
    )
    print(f"walk-forward report: {output_dir}")
    print(f"standard backtest report: {output_dir / 'backtest_report'}")
    print(f"walk-forward log: {full_log_path}")
    print(f"validation final equity: {result.metrics.get('final_equity')}")
    print(f"validation total return: {result.metrics.get('total_return')}")


if __name__ == "__main__":
    main()
