"""启动本地 trade review 报告中心（仅监听 127.0.0.1）。

用法：
  python scripts/serve_trade_review.py
  python scripts/serve_trade_review.py --port 8080
  python scripts/serve_trade_review.py --run-dir results/backtests/<run> --port 8080
"""
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.modules.reporting.trade_review import serve_report_center  # noqa: E402


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="本地 trade review 报告中心（127.0.0.1）")
    parser.add_argument("--run-dir", type=Path, default=None, help="启动后默认打开的回测输出目录")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "backtests",
        help="包含多个回测结果文件夹的目录（默认 results/backtests）",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    results_root = args.results_root
    run_dir = args.run_dir
    if run_dir is not None:
        resolved_run_dir = run_dir if run_dir.is_absolute() else PROJECT_ROOT / run_dir
        resolved_results_root = results_root if results_root.is_absolute() else PROJECT_ROOT / results_root
        if resolved_run_dir.resolve().parent != resolved_results_root.resolve():
            results_root = resolved_run_dir.resolve().parent

    serve_report_center(
        results_root,
        port=args.port,
        open_browser=not args.no_browser,
        initial_run_dir=run_dir,
    )
