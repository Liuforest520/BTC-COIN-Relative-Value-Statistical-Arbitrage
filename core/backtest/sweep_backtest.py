from functools import lru_cache
from pathlib import Path

import polars as pl

from core.backtest.backtest import Backtest, BacktestResult
from core.modules.config import Config, load_config
from core.modules.data import load_csv_data
from core.modules.metrics import calculate_metrics
from tqdm import tqdm


class DiscardList(list):
    def __init__(self):
        super().__init__()
        self.count = 0

    def append(self, item):
        self.count += 1

    def extend(self, items):
        self.count += len(items)

    def __len__(self):
        return self.count


class SweepBacktest(Backtest):
    """Lightweight backtest for config sweeps.

    It keeps the same strategy, exchange, risk, cost, and metric logic as the
    detailed Backtest, but discards per-bar diagnostic curves that are only
    needed for full reports.
    """

    def __init__(
        self,
        config: Config,
        strategy=None,
        risk_manager=None,
        exchange_manager=None,
        show_progress: bool = False,
        progress_callback=None,
        progress_interval_bars: int = 100_000,
    ):
        super().__init__(
            config=config,
            strategy=strategy,
            risk_manager=risk_manager,
            exchange_manager=exchange_manager,
            show_progress=show_progress,
        )
        self.position_curve = DiscardList()
        self.signal_curve = DiscardList()
        self.risk_manager.history = DiscardList()
        self.progress_callback = progress_callback
        self.progress_interval_bars = max(1, int(progress_interval_bars))
        self.exchange_manager.compact_equity_curve = True
        self.exchange_manager.equity_curve = {"ts": [], "equity": []}
        self.exchange_manager.include_trade_history_on_bar = False
        for exchange in self.exchange_manager.exchanges.values():
            exchange.copy_positions_on_bar = False
        for pipeline in getattr(self.strategy, "pipelines", {}).values():
            pipeline.collect_diagnostics = False
            pipeline.estimator.collect_diagnostics = False
        for risk in [*self.risk_manager.pre_trade_risks, *self.risk_manager.post_trade_risks]:
            risk.history = DiscardList()

    def run(self) -> BacktestResult:
        market_data = self._load_market_data()
        funding_data = self._load_funding_data()
        self._check_funding_alignment(funding_data, market_data)
        funding_timestamps = sorted(funding_data)
        funding_index = 0
        pending_funding_events = []
        previous_ts = None
        last_bars = None

        bar_iterator = self._iter_market_bars(market_data)
        if self.show_progress:
            bar_iterator = tqdm(
                bar_iterator,
                total=self._market_progress_total(market_data),
                desc="Sweep backtest",
                unit="bar",
                dynamic_ncols=True,
                mininterval=1.0,
            )

        processed_bars = 0
        for bars in bar_iterator:
            processed_bars += 1
            if self.progress_callback and processed_bars % self.progress_interval_bars == 0:
                self.progress_callback(processed_bars)
            last_bars = bars
            ts = self._bars_ts(bars)
            if self.config.funding_enabled:
                new_funding_events, funding_index = self._funding_events_for_bar(
                    funding_data,
                    funding_timestamps,
                    funding_index,
                    previous_ts,
                    ts,
                )
                funding_events, pending_funding_events = self._available_funding_events(
                    pending_funding_events,
                    new_funding_events,
                    bars,
                )
                funding_rates = self._latest_funding_rates(funding_events)
            else:
                funding_events = []
                funding_rates = {}

            exchange_result = self.exchange_manager.on_bar(bars, funding_rates=funding_events)
            new_trades = exchange_result["new_trades"]
            rejected_orders = exchange_result["rejected_orders"]
            self.strategy.on_funding_rates(funding_rates)
            funding_callback = getattr(self.strategy, "on_funding_payments", None)
            if callable(funding_callback):
                funding_callback(exchange_result.get("funding_payments", []))
            previous_ts = ts

            if new_trades:
                self.trades.extend(new_trades)
                self.strategy.on_trades_filled(new_trades)
                self.risk_manager.check_trades(new_trades, exchange_result["positions"], bars)

            if rejected_orders:
                self.strategy.on_orders_rejected(rejected_orders)

            portfolio_snapshot = exchange_result if isinstance(exchange_result, dict) else {}
            self.strategy.set_portfolio_context(
                cash=portfolio_snapshot.get("cash", 0),
                equity=portfolio_snapshot.get("equity", 0),
                available_balance=portfolio_snapshot.get("available_balance", 0),
            )
            orders = self.strategy(bars)
            if not orders:
                continue

            risk_results = self.risk_manager.check_orders(orders, bars)
            if not self.risk_manager.passed(risk_results):
                self.strategy.on_orders_rejected(orders)
                continue

            accepted_orders = self.exchange_manager.place_orders(orders)
            if len(accepted_orders) != len(orders):
                self.strategy.on_orders_rejected(orders)
                self.exchange_manager.cancel_all_orders()
                continue

            self.strategy.on_orders_accepted(accepted_orders)
            self.orders.extend(accepted_orders)

        metrics = calculate_metrics(
            equity_curve=self.exchange_manager.equity_curve,
            orders=self.orders,
            trades=self.trades,
            funding_payments=self.exchange_manager.funding_payments,
            benchmark_returns=self._benchmark_returns(market_data),
            hedge_ratio_tolerance=float(self.config.risk.get("order_hedge_ratio_tolerance", 0.02)),
        )

        return BacktestResult(
            metrics=metrics,
            equity_curve=self.exchange_manager.equity_curve,
            position_curve=[],
            signal_curve=[],
            trades=self.trades,
            orders=self.orders,
            funding_payments=self.exchange_manager.funding_payments,
            final_position_valuation=self._final_position_valuation(last_bars),
            risk_history=self.risk_manager.history,
        )

    def _join_external_benchmark(self, frame, name, info):
        path = info.get("path") if isinstance(info, dict) else None
        if not path:
            return frame

        benchmark = _cached_load_csv_data(str(_absolute_path(path)))
        benchmark = benchmark.select(["ts", "close"]).rename({"close": f"{name.lower()}_close"})
        benchmark = benchmark.with_columns(
            (pl.col(f"{name.lower()}_close") / pl.col(f"{name.lower()}_close").shift(1) - 1).alias(f"{name.lower()}_return")
        )
        benchmark = benchmark.select(["ts", f"{name.lower()}_return"])
        return frame.join(benchmark, on="ts", how="left")

    def _position_snapshot(self, ts, bars, exchange_result):
        return None

    def _record_signal_state(self, ts, force: bool = False):
        return None

    def _record_pair_state(self, ts, force: bool = False):
        return None


def run_sweep_backtest(
    config_path: str | Path = "config/config.yaml",
    show_progress: bool = False,
    progress_callback=None,
    progress_interval_bars: int = 100_000,
) -> BacktestResult:
    config = load_config(config_path)
    backtest = SweepBacktest(
        config,
        show_progress=show_progress,
        progress_callback=progress_callback,
        progress_interval_bars=progress_interval_bars,
    )
    return backtest.run()


@lru_cache(maxsize=32)
def _cached_load_csv_data(path: str):
    return load_csv_data(Path(path))


def _absolute_path(path: str | Path) -> Path:
    return Path(path).resolve()
