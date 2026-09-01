from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import polars as pl

from core.modules.config import Config, load_config
from core.modules.data import build_funding_map, iter_csv_bars, load_csv_data, load_funding_data
from core.modules.exchange import ExchangeManager
from core.modules.logger import logger
from core.modules.metrics import calculate_metrics
from core.modules.risk import build_risk_manager
from core.modules.strategy import build_strategy
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
    pair_curve: list[dict] | None = None


class Backtest:
    def __init__(self, config: Config, strategy=None, risk_manager=None, exchange_manager=None, show_progress: bool = True):
        self.config = config
        self.strategy = strategy or self._build_strategy()
        # Backtest market iterators already normalize and validate every bar.
        # Keep BaseStrategy validation enabled for direct callers, but avoid
        # repeating the same per-symbol structure scan inside the hot loop.
        self.strategy.validate_bars = False
        self.risk_manager = risk_manager or build_risk_manager(self.config.strategy, self.config.risk)
        self.exchange_manager = exchange_manager or self._build_exchange_manager()
        # Backtest owns the canonical trade list through new_trades. Rebuilding
        # a cumulative exchange trade snapshot on every bar is unused here and
        # becomes quadratic on trade-heavy runs.
        self.exchange_manager.include_trade_history_on_bar = False
        self.show_progress = show_progress
        self.orders = []
        self.trades = []
        self.position_curve = []
        self.signal_curve = []
        self.pair_curve = []
        self.diagnostic_sample_interval_bars = int(
            (self.config.risk or {}).get("diagnostic_sample_interval_bars", 60)
        )

    def run(self) -> BacktestResult:
        market_data = self._load_market_data()
        funding_data = self._load_funding_data()
        self._check_funding_alignment(funding_data, market_data)
        funding_timestamps = sorted(funding_data)
        funding_index = 0
        pending_funding_events = []
        previous_ts = None
        last_bars = None

        for bars in tqdm(
            self._iter_market_bars(market_data),
            total=self._market_progress_total(market_data),
            desc="Backtest",
            unit="bar",
            dynamic_ncols=True,
            mininterval=1.0,
            disable=not self.show_progress,
        ):
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

            portfolio_snapshot = exchange_result if isinstance(exchange_result, dict) else {}
            self.strategy.set_portfolio_context(
                cash=portfolio_snapshot.get("cash", 0),
                equity=portfolio_snapshot.get("equity", 0),
            )
            orders = self.strategy(bars)
            should_record_diagnostics = self._should_record_diagnostics(orders, new_trades, rejected_orders)
            # Z-score is part of the minute-level trade path and must never be
            # sampled.  The heavier diagnostic fields are still recorded only
            # on events and at the configured diagnostic interval.
            self._record_signal_state(ts, detailed=should_record_diagnostics)
            self._record_pair_state(ts, should_record_diagnostics)
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
            position_curve=self.position_curve,
            signal_curve=self.signal_curve,
            trades=self.trades,
            orders=self.orders,
            funding_payments=self.exchange_manager.funding_payments,
            final_position_valuation=self._final_position_valuation(last_bars),
            risk_history=self.risk_manager.history,
            pair_curve=self.pair_curve if self.pair_curve else None,
        )

    def _build_strategy(self):
        return build_strategy(self.config.strategy, self.config.symbols)

    def _build_exchange_manager(self):
        exchange_names = []
        active_symbols = self._active_symbols()
        for symbol, info in self.config.symbols.items():
            if active_symbols and symbol not in active_symbols:
                continue
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
        if self._risk_pass_through_enabled():
            return float("inf")
        raw_value = self.config.risk.get("max_leverage", 1.0)
        if raw_value is None:
            return 1.0
        max_leverage = float(raw_value)
        if max_leverage <= 0:
            raise ValueError("risk.max_leverage must be positive")
        return max_leverage

    def _risk_pass_through_enabled(self):
        risk_config = self.config.risk or {}
        mode = str(risk_config.get("mode", "")).strip().lower()
        if mode in {"pass_through", "passthrough", "disabled", "off", "none"}:
            return True
        if "enabled" in risk_config:
            return not self._as_bool(risk_config.get("enabled"))
        return False

    def _as_bool(self, value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _load_market_data(self):
        data = {}
        active_symbols = self._active_symbols()
        for symbol, info in self.config.symbols.items():
            if active_symbols and symbol not in active_symbols:
                continue
            path = Path(info["path"])
            data[symbol] = {
                "exchange": info["exchange"],
                "path": path,
            }
        return data

    def _active_symbols(self) -> set[str]:
        symbols = set()
        for pair in getattr(self.config.strategy, "pairs", []) or []:
            if not pair.get("enabled", True):
                continue
            x_symbol = pair.get("x_symbol", pair.get("long_symbol"))
            y_symbol = pair.get("y_symbol", pair.get("short_symbol"))
            if x_symbol:
                symbols.add(x_symbol)
            if y_symbol:
                symbols.add(y_symbol)
        return symbols

    def _market_progress_total(self, market_data):
        total = 0
        for item in market_data.values():
            frame = item.get("frame")
            if frame is None:
                return None
            total += frame.height
        return total

    def _should_record_diagnostics(self, orders=None, trades=None, rejected_orders=None) -> bool:
        if orders or trades or rejected_orders:
            return True
        interval = max(1, int(self.diagnostic_sample_interval_bars or 1))
        bar_index = int(getattr(self.strategy, "_global_bar_index", 0) or 0)
        return bar_index <= 1 or bar_index % interval == 0

    def _record_signal_state(self, ts, detailed: bool = False):
        snapshotter = getattr(self.strategy, "snapshot_signal_state", None)
        if callable(snapshotter):
            try:
                state = snapshotter(ts, compact=not detailed)
            except TypeError:
                state = snapshotter(ts)
        else:
            state = dict(getattr(self.strategy, "last_state", {}) or {})
            if not state:
                signal = getattr(self.strategy, "signal", None)
                state = dict(getattr(signal, "last_state", {}) or {})
        if not state:
            state = {"ts": ts}
        if state.get("ts") is None:
            state["ts"] = ts
        self.signal_curve.append(state)

    def _load_funding_data(self):
        frames = []
        active_symbols = self._active_symbols()
        for symbol, info in self.config.symbols.items():
            if active_symbols and symbol not in active_symbols:
                continue
            path = info.get("funding_path")
            if not path:
                continue
            frames.append(load_funding_data(Path(path), symbol=symbol))
        return build_funding_map(frames)

    def _check_funding_alignment(self, funding_data, market_data):
        if not self.config.funding_enabled:
            return
        if not funding_data:
            logger.warning("funding is enabled but no funding data was loaded")
            return
        active_symbols = set(market_data)
        funding_symbols = {
            symbol
            for rates in funding_data.values()
            for symbol in rates
        }
        unexpected_symbols = funding_symbols - active_symbols
        if unexpected_symbols:
            logger.warning(
                "funding data contains symbols outside the active market set: {}",
                sorted(unexpected_symbols),
            )

        event_count = sum(len(rates) for rates in funding_data.values())
        logger.info(
            "funding events loaded: {} rates at {} actual timestamps; "
            "events settle at each symbol's next available bar",
            event_count,
            len(funding_data),
        )

    def _funding_events_for_bar(self, funding_data, funding_timestamps, funding_index, previous_ts, ts):
        if ts is None:
            return [], funding_index

        events = []
        if previous_ts is None:
            while funding_index < len(funding_timestamps) and funding_timestamps[funding_index] < ts:
                funding_index += 1

        while funding_index < len(funding_timestamps) and funding_timestamps[funding_index] <= ts:
            funding_ts = funding_timestamps[funding_index]
            if previous_ts is None or funding_ts > previous_ts:
                events.append({"ts": funding_ts, "rates": funding_data[funding_ts]})
            funding_index += 1
        return events, funding_index

    def _available_funding_events(self, pending_events, new_events, bars):
        pending = list(pending_events)
        for event in new_events:
            event_ts = event.get("ts")
            for symbol, rate in event.get("rates", {}).items():
                pending.append({"ts": event_ts, "symbol": symbol, "rate": rate})

        available_symbols = {
            symbol
            for exchange_bars in (bars or {}).values()
            for symbol in exchange_bars
        }
        payable_by_ts = {}
        remaining = []
        for item in pending:
            if item["symbol"] not in available_symbols:
                remaining.append(item)
                continue
            payable_by_ts.setdefault(item["ts"], {})[item["symbol"]] = item["rate"]

        payable = [
            {"ts": event_ts, "rates": payable_by_ts[event_ts]}
            for event_ts in sorted(payable_by_ts)
        ]
        return payable, remaining

    def _latest_funding_rates(self, funding_events):
        rates = {}
        for event in funding_events:
            rates.update(event.get("rates", {}))
        return rates

    def _iter_market_bars(self, market_data):
        rows_by_symbol = {}
        current_by_symbol = {}
        pair_specs = self._alignment_pair_specs(market_data)

        for symbol, item in market_data.items():
            iterator = self._symbol_bar_iterator(item)
            rows_by_symbol[symbol] = iterator
            current = next(iterator, None)
            if current is not None:
                current_by_symbol[symbol] = current

        while current_by_symbol:
            ts = min(self._row_ts(row) for row in current_by_symbol.values())
            bars = {}
            emitted_symbols = set()

            for symbol, row in list(current_by_symbol.items()):
                row_ts = self._row_ts(row)
                if row_ts != ts:
                    continue
                exchange = market_data[symbol]["exchange"]
                if isinstance(row, list) and len(row) >= 6:
                    # iter_csv_bars already returns finite, normalized values.
                    bars.setdefault(exchange, {})[symbol] = row
                    emitted_symbols.add(symbol)
                else:
                    values = self._row_ohlcv_values(row)
                    if self._valid_bar_values(values):
                        bars.setdefault(exchange, {})[symbol] = [
                            ts,
                            float(values[0]),
                            float(values[1]),
                            float(values[2]),
                            float(values[3]),
                            float(values[4]),
                        ]
                        emitted_symbols.add(symbol)

                next_row = next(rows_by_symbol[symbol], None)
                if next_row is None:
                    current_by_symbol.pop(symbol, None)
                else:
                    current_by_symbol[symbol] = next_row

            if bars and self._has_tradable_pair_symbols(emitted_symbols, pair_specs):
                yield bars

    def _symbol_bar_iterator(self, item):
        start_ts = self._backtest_start_ts()
        end_ts = self._backtest_end_ts()
        frame = item.get("frame")
        if frame is not None:
            for row in frame.iter_rows():
                row_ts = self._row_ts(row)
                if start_ts is not None and row_ts < start_ts:
                    continue
                if end_ts is not None and row_ts > end_ts:
                    break
                yield row
            return

        path = item.get("path")
        if path is None:
            return
        for bar in iter_csv_bars(path):
            row_ts = self._row_ts(bar)
            if start_ts is not None and row_ts < start_ts:
                continue
            if end_ts is not None and row_ts > end_ts:
                break
            yield bar

    def _row_ohlcv_values(self, row):
        if isinstance(row, dict):
            return [row.get(column) for column in ["open", "high", "close", "low", "volume"]]
        return [row[1], row[2], row[3], row[4], row[5]]

    def _row_ts(self, row):
        return int(row["ts"] if isinstance(row, dict) else row[0])

    def _has_tradable_pair_symbols(self, symbols, pair_specs):
        if not pair_specs:
            return True
        return any(x_symbol in symbols and y_symbol in symbols for _pair_id, x_symbol, y_symbol in pair_specs)

    def _valid_bar_values(self, values):
        if any(value is None for value in values):
            return False
        try:
            return all(isfinite(float(value)) for value in values)
        except (TypeError, ValueError):
            return False

    def _bars_ts(self, bars):
        for symbols in bars.values():
            for bar in symbols.values():
                return bar[0]
        return None

    def _backtest_start_ts(self):
        return getattr(self.config, "backtest_start_ts", None)

    def _backtest_end_ts(self):
        return getattr(self.config, "backtest_end_ts", None)

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

    def _position_snapshot(self, ts, bars, exchange_result):
        equity = float(exchange_result["equity"])
        snapshot = {
            "ts": ts,
            "equity": equity,
            "cash": float(exchange_result["cash"]),
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
        }
        for symbol in self._active_symbols():
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
            exchange = self.exchange_manager.exchanges.get(exchange_name)
            bar = getattr(exchange, "last_bars", {}).get(symbol) if exchange is not None else None
        if bar is None:
            return None
        if isinstance(bar, dict):
            return float(bar["close"])
        return float(bar[3])

    def _alignment_pair_specs(self, market_data):
        specs = []
        for pair in getattr(self.config.strategy, "pairs", None) or []:
            if not pair.get("enabled", True):
                continue
            pair_id = pair.get("pair_id") or f"{pair.get('x_symbol', '')}_{pair.get('y_symbol', '')}"
            x_symbol = pair.get("x_symbol", pair.get("long_symbol"))
            y_symbol = pair.get("y_symbol", pair.get("short_symbol"))
            if x_symbol in market_data and y_symbol in market_data:
                specs.append((pair_id, x_symbol, y_symbol))

        if specs:
            return specs
        return specs

    def _record_pair_state(self, ts, force: bool = False):
        """Capture per-pair runtime state from MultiPairStrategy pipelines."""
        if not force:
            return
        pipelines = getattr(self.strategy, 'pipelines', None)
        if not pipelines:
            return
        for pair_id, pp in pipelines.items():
            st = pp.state
            row = {
                "ts": ts,
                "pair_id": pair_id,
                "bar_index": st.last_bar_index,
                "global_bar_index": getattr(self.strategy, '_global_bar_index', 0),
                "is_ready": st.is_ready,
                "alpha": st.estimator_state.alpha,
                "beta": st.estimator_state.spread_beta,
                "hedge_beta": st.estimator_state.hedge_beta,
                "spread_mean": st.estimator_state.spread_mean,
                "spread_std": st.estimator_state.spread_std,
                "half_life": st.estimator_state.residual_half_life_bars,
                "cointegration_pass": st.estimator_state.cointegration_pass,
                "cointegration_pvalue": st.estimator_state.cointegration_pvalue,
                "cointegration_stat": st.estimator_state.cointegration_stat,
                "cointegration_block_open": st.estimator_state.cointegration_block_open,
                "cointegration_reason": st.estimator_state.cointegration_reason,
                "next_model_update_index": st.estimator_state.next_model_update_index,
                "next_hedge_model_update_index": st.estimator_state.next_hedge_model_update_index,
                "last_model_update_skip_index": st.estimator_state.last_model_update_skip_index,
                "last_hedge_model_update_skip_index": st.estimator_state.last_hedge_model_update_skip_index,
                "block_reason": st.last_block_reason,
            }
            self.pair_curve.append(row)

    def _benchmark_returns(self, market_data):
        equity = pl.DataFrame(self.exchange_manager.equity_curve)
        if equity.is_empty() or "ts" not in equity.columns or "equity" not in equity.columns:
            return pl.DataFrame()

        # Multi-pair strategies don't have a single pair — skip per-pair benchmark
        frame = equity.select(["ts", "equity"]).with_columns(
            pl.col("equity").cast(pl.Float64, strict=False)
        )
        frame = frame.with_columns(
            (pl.col("equity") / pl.col("equity").shift(1) - 1).alias("portfolio_return")
        )

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
