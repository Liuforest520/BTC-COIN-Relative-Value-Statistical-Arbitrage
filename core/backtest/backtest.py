from dataclasses import dataclass
from pathlib import Path

import polars as pl

from core.modules.config import Config, load_config
from core.modules.data import build_funding_map, load_csv_data, load_funding_data
from core.modules.exchange import ExchangeManager
from core.modules.logger import logger
from core.modules.metrics import calculate_metrics
from core.modules.risk import build_risk_manager
from core.modules.strategy import BAR_COLUMNS, build_strategy
from tqdm import tqdm


@dataclass
class BacktestResult:
    metrics: dict
    equity_curve: list[dict]
    position_curve: list[dict]
    signal_curve: list[dict]
    trades: list
    orders: list
    funding_payments: list
    final_position_valuation: dict
    risk_history: list


class Backtest:
    def __init__(self, config: Config, strategy=None, risk_manager=None, exchange_manager=None, show_progress: bool = True):
        self.config = config
        self.strategy = strategy or self._build_strategy()
        self.risk_manager = risk_manager or build_risk_manager(self.config.strategy, self.config.risk)
        self.exchange_manager = exchange_manager or self._build_exchange_manager()
        self.show_progress = show_progress
        self.orders = []
        self.trades = []
        self.position_curve = []
        self.signal_curve = []

    def run(self) -> BacktestResult:
        market_data = self._load_market_data()
        funding_data = self._load_funding_data()
        joined_data = self._join_market_data(market_data)
        self._check_funding_alignment(funding_data, joined_data)
        funding_timestamps = sorted(funding_data)
        funding_index = 0
        previous_ts = None
        last_bars = None

        for bars in tqdm(
            self._iter_bars(market_data, joined_data),
            total=joined_data.height,
            desc="Backtest",
            unit="bar",
            dynamic_ncols=True,
            mininterval=1.0,
            disable=not self.show_progress,
        ):
            last_bars = bars
            ts = self._bars_ts(bars)
            if self.config.funding_enabled:
                funding_events, funding_index = self._funding_events_for_bar(
                    funding_data,
                    funding_timestamps,
                    funding_index,
                    previous_ts,
                    ts,
                )
                funding_rates = self._latest_funding_rates(funding_events)
            else:
                funding_events = []
                funding_rates = {}

            exchange_result = self.exchange_manager.on_bar(bars, funding_rates=funding_events)
            self.position_curve.append(self._position_snapshot(ts, bars, exchange_result))
            new_trades = exchange_result["new_trades"]
            rejected_orders = exchange_result["rejected_orders"]
            self.strategy.on_funding_rates(funding_rates)
            previous_ts = ts

            if new_trades:
                self.trades.extend(new_trades)
                self.strategy.on_trades_filled(new_trades)
                self.risk_manager.check_trades(new_trades, exchange_result["positions"], bars)

            if rejected_orders:
                self.strategy.on_orders_rejected(rejected_orders)

            orders = self.strategy(bars)
            self._record_signal_state(ts)
            if not orders:
                continue

            risk_results = self.risk_manager.check_orders(orders, bars)
            if not self.risk_manager.passed(risk_results):
                if self.risk_manager.block_open_orders:
                    self.signal_curve.append({"ts": ts, "action": "open_blocked_by_risk", "reason": "risk block"})
                self.strategy.on_orders_rejected(orders)
                continue

            accepted_orders = self.exchange_manager.place_orders(orders)
            if len(accepted_orders) != len(orders):
                self.strategy.on_orders_rejected(orders)
                self.exchange_manager.cancel_all_orders()
                continue

            self.strategy.on_orders_accepted(accepted_orders)
            self.orders.extend(accepted_orders)

        final_position_valuation = self._final_position_valuation(last_bars)
        metrics = calculate_metrics(
            equity_curve=self.exchange_manager.equity_curve,
            orders=self.orders,
            trades=self.trades,
            funding_payments=self.exchange_manager.funding_payments,
            benchmark_returns=self._benchmark_returns(joined_data),
            hedge_ratio_tolerance=float(self.config.risk.get("order_hedge_ratio_tolerance", 0.02)),
        )
        metrics.update(self._final_position_metrics(final_position_valuation))

        return BacktestResult(
            metrics=metrics,
            equity_curve=self.exchange_manager.equity_curve,
            position_curve=self.position_curve,
            signal_curve=self.signal_curve,
            trades=self.trades,
            orders=self.orders,
            funding_payments=self.exchange_manager.funding_payments,
            final_position_valuation=final_position_valuation,
            risk_history=self.risk_manager.history,
        )

    def _build_strategy(self):
        return build_strategy(self.config.strategy, self.config.symbols)

    def _build_exchange_manager(self):
        exchange_names = []
        for info in self.config.symbols.values():
            if info["exchange"] not in exchange_names:
                exchange_names.append(info["exchange"])
        return ExchangeManager.from_names(
            exchange_names=exchange_names,
            initial_cash=self.config.initial_cash,
            fee_rate=self.config.fee_rate,
            slippage_bps=self.config.slippage_bps,
            max_leverage=self._max_leverage(),
        )

    def _max_leverage(self):
        raw_value = self.config.risk.get("max_leverage", 1.0)
        if raw_value is None:
            return 1.0
        max_leverage = float(raw_value)
        if max_leverage <= 0:
            raise ValueError("risk.max_leverage must be positive")
        return max_leverage

    def _load_market_data(self):
        data = {}
        for symbol, info in self.config.symbols.items():
            path = Path(info["path"])
            frame = load_csv_data(path)
            data[symbol] = {
                "exchange": info["exchange"],
                "frame": frame.select(BAR_COLUMNS),
            }
        return data

    def _record_signal_state(self, ts):
        signal = getattr(self.strategy, "signal", None)
        state = dict(getattr(signal, "last_state", {}) or {})
        if not state:
            state = {"ts": ts}
        if state.get("ts") is None:
            state["ts"] = ts
        self.signal_curve.append(state)

    def _load_funding_data(self):
        frames = []
        for symbol, info in self.config.symbols.items():
            path = info.get("funding_path")
            if not path:
                continue
            frames.append(load_funding_data(Path(path), symbol=symbol))
        return build_funding_map(frames)

    def _check_funding_alignment(self, funding_data, joined_data):
        if not funding_data or not self.config.funding_enabled:
            return
        bar_timestamps = set(joined_data["ts"].to_list())
        funding_timestamps = set(funding_data.keys())
        matched = funding_timestamps & bar_timestamps
        start_ts = joined_data["ts"].min()
        end_ts = joined_data["ts"].max()
        in_range = {ts for ts in funding_timestamps if start_ts <= ts <= end_ts}
        if not matched:
            logger.warning(
                "funding rate timestamps have zero exact overlap with bar timestamps; "
                "{} events are inside the backtest range and will be applied on the next bar",
                len(in_range),
            )
        elif len(matched) < len(funding_timestamps):
            logger.info(
                "funding rate timestamps: {} exact matches / {} in range / {} total",
                len(matched),
                len(in_range),
                len(funding_timestamps),
            )

    def _funding_events_for_bar(self, funding_data, funding_timestamps, funding_index, previous_ts, ts):
        if ts is None:
            return [], funding_index

        events = []
        while funding_index < len(funding_timestamps) and funding_timestamps[funding_index] <= ts:
            funding_ts = funding_timestamps[funding_index]
            if previous_ts is None or funding_ts > previous_ts:
                events.append({"ts": funding_ts, "rates": funding_data[funding_ts]})
            funding_index += 1
        return events, funding_index

    def _latest_funding_rates(self, funding_events):
        rates = {}
        for event in funding_events:
            rates.update(event.get("rates", {}))
        return rates

    def _iter_bars(self, market_data, joined):
        symbols = list(market_data.keys())

        for row in joined.iter_rows(named=True):
            bars = {}
            for symbol in symbols:
                exchange = market_data[symbol]["exchange"]
                bars.setdefault(exchange, {})
                bars[exchange][symbol] = [
                    row["ts"],
                    row[f"{symbol}_open"],
                    row[f"{symbol}_high"],
                    row[f"{symbol}_close"],
                    row[f"{symbol}_low"],
                    row[f"{symbol}_volume"],
                ]
            yield bars

    def _bars_ts(self, bars):
        for symbols in bars.values():
            for bar in symbols.values():
                return bar[0]
        return None

    def _final_position_valuation(self, bars):
        if not bars:
            return {
                "ts": None,
                "cash": 0.0,
                "position_value": 0.0,
                "long_value": 0.0,
                "short_value": 0.0,
                "gross_exposure": 0.0,
                "net_exposure": 0.0,
                "equity": 0.0,
                "positions": {},
            }

        total_cash = 0.0
        total_position_value = 0.0
        total_long_value = 0.0
        total_short_value = 0.0
        positions = {}

        for exchange_name, exchange in self.exchange_manager.exchanges.items():
            exchange_positions = {}
            total_cash += exchange.cash

            for symbol, quantity in exchange.positions.items():
                price = self._close_price(bars, exchange_name, symbol)
                if price is None:
                    continue

                quantity = float(quantity)
                value = quantity * price
                abs_value = abs(value)
                total_position_value += value
                if value > 0:
                    total_long_value += abs_value
                elif value < 0:
                    total_short_value += abs_value

                exchange_positions[symbol] = {
                    "quantity": quantity,
                    "mark_price": price,
                    "value": value,
                    "abs_value": abs_value,
                }

            positions[exchange_name] = exchange_positions

        equity = total_cash + total_position_value
        return {
            "ts": self._bars_ts(bars),
            "cash": total_cash,
            "position_value": total_position_value,
            "long_value": total_long_value,
            "short_value": total_short_value,
            "gross_exposure": total_long_value + total_short_value,
            "net_exposure": total_long_value - total_short_value,
            "equity": equity,
            "positions": positions,
        }

    def _final_position_metrics(self, valuation):
        equity = valuation["equity"]
        gross_exposure = valuation["gross_exposure"]
        net_exposure = valuation["net_exposure"]
        return {
            "final_cash": valuation["cash"],
            "final_position_value": valuation["position_value"],
            "final_long_value": valuation["long_value"],
            "final_short_value": valuation["short_value"],
            "final_gross_exposure": gross_exposure,
            "final_net_exposure": net_exposure,
            "final_net_exposure_ratio": net_exposure / equity if equity else None,
            "final_gross_exposure_ratio": gross_exposure / equity if equity else None,
        }

    def _position_snapshot(self, ts, bars, exchange_result):
        equity = float(exchange_result["equity"])
        snapshot = {
            "ts": ts,
            "equity": equity,
            "cash": float(exchange_result["cash"]),
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
        }
        for symbol in self.config.symbols:
            snapshot[f"{symbol}_position_value"] = 0.0
            snapshot[f"{symbol}_position_ratio"] = 0.0

        for exchange_name, positions in exchange_result["positions"].items():
            for symbol, quantity in positions.items():
                price = self._close_price(bars, exchange_name, symbol)
                if price is None:
                    continue

                value = float(quantity) * price
                snapshot[f"{symbol}_position_value"] = value
                snapshot[f"{symbol}_position_ratio"] = value / equity if equity else None
                snapshot["gross_exposure"] += abs(value)
                snapshot["net_exposure"] += value

        snapshot["gross_exposure_ratio"] = snapshot["gross_exposure"] / equity if equity else None
        snapshot["net_exposure_ratio"] = snapshot["net_exposure"] / equity if equity else None
        return snapshot

    def _close_price(self, bars, exchange_name, symbol):
        bar = bars.get(exchange_name, {}).get(symbol)
        if bar is None:
            return None
        if isinstance(bar, dict):
            return float(bar["close"])
        return float(bar[3])

    def _join_market_data(self, market_data):
        frames = []

        for symbol, item in market_data.items():
            frame = item["frame"].rename(
                {
                    "open": f"{symbol}_open",
                    "high": f"{symbol}_high",
                    "close": f"{symbol}_close",
                    "low": f"{symbol}_low",
                    "volume": f"{symbol}_volume",
                }
            )
            frames.append((symbol, frame))

        _, joined = frames[0]
        for symbol, frame in frames[1:]:
            left_rows = joined.height
            right_rows = frame.height
            joined = joined.join(frame, on="ts", how="inner")
            dropped_rows = min(left_rows, right_rows) - joined.height
            if dropped_rows > 0:
                logger.warning(
                    "market data inner join dropped {} rows while joining {}; left_rows={} right_rows={} joined_rows={}",
                    dropped_rows,
                    symbol,
                    left_rows,
                    right_rows,
                    joined.height,
                )

        return joined.sort("ts")

    def _benchmark_returns(self, joined_data):
        equity = pl.DataFrame(self.exchange_manager.equity_curve)
        if equity.is_empty() or "ts" not in equity.columns or "equity" not in equity.columns:
            return pl.DataFrame()

        pair = self.config.strategy.pair
        x_symbol = pair.get("x_symbol")
        y_symbol = pair.get("y_symbol")

        frame = equity.select(["ts", "equity"]).with_columns(
            pl.col("equity").cast(pl.Float64, strict=False)
        )
        frame = frame.with_columns(
            (pl.col("equity") / pl.col("equity").shift(1) - 1).alias("portfolio_return")
        )

        market_columns = ["ts"]
        rename_map = {}
        if x_symbol and f"{x_symbol}_close" in joined_data.columns:
            market_columns.append(f"{x_symbol}_close")
            rename_map[f"{x_symbol}_close"] = "btc_close"
        if y_symbol and f"{y_symbol}_close" in joined_data.columns:
            market_columns.append(f"{y_symbol}_close")
            rename_map[f"{y_symbol}_close"] = "coin_close"

        market_returns = joined_data.select(market_columns).rename(rename_map)
        return_columns = ["ts"]
        expressions = []
        for name in ["btc", "coin"]:
            close_column = f"{name}_close"
            if close_column in market_returns.columns:
                expressions.append((pl.col(close_column) / pl.col(close_column).shift(1) - 1).alias(f"{name}_return"))
                return_columns.append(f"{name}_return")

        if expressions:
            market_returns = market_returns.with_columns(expressions).select(return_columns)
            frame = frame.join(market_returns, on="ts", how="left")

        for name, info in self.config.benchmarks.items():
            frame = self._join_external_benchmark(frame, name, info)

        return frame.drop_nulls("portfolio_return")

    def _join_external_benchmark(self, frame, name, info):
        path = info.get("path") if isinstance(info, dict) else None
        if not path:
            return frame

        benchmark = load_csv_data(Path(path))
        benchmark = benchmark.select(["ts", "close"]).rename({"close": f"{name.lower()}_close"})
        benchmark = benchmark.with_columns(
            (pl.col(f"{name.lower()}_close") / pl.col(f"{name.lower()}_close").shift(1) - 1).alias(f"{name.lower()}_return")
        )
        benchmark = benchmark.select(["ts", f"{name.lower()}_return"])
        return frame.join(benchmark, on="ts", how="left")


def run_backtest(config_path: str | Path = "config/config.yaml", show_progress: bool = True) -> BacktestResult:
    config = load_config(config_path)
    backtest = Backtest(config, show_progress=show_progress)
    return backtest.run()
