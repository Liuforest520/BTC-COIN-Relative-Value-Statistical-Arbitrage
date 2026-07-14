from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.backtest import run_backtest
from core.modules.logger import logger
from core.modules.reporting import export_backtest_report, export_trade_review_html


def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/config.yaml")
    result = run_backtest(config_path)
    report_dir = export_backtest_report(result, config_path)
    review_path = None
    try:
        review_path = export_trade_review_html(report_dir)
    except Exception as exc:
        logger.warning("trade review export failed: {}", exc)

    logger.info("backtest finished")
    logger.info("report saved: {}", report_dir)
    if review_path is not None:
        logger.info("trade review saved: {}", review_path)
    logger.info("equity points: {}", len(result.equity_curve))
    logger.info("orders: {}", len(result.orders))
    logger.info("trades: {}", len(result.trades))
    logger.info("final position valuation: {}", result.final_position_valuation)

    for name, value in result.metrics.items():
        logger.info("metric {}={}", name, value)


if __name__ == "__main__":
    main()
