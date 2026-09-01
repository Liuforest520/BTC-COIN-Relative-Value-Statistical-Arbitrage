"""MultiPairStrategy: orchestrates PairPipelines + PortfolioAllocator + OrderPlanner."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from core.modules.execution.order_planner import OrderPlanner
from core.modules.models import Order
from core.modules.models.pipeline_types import (
    AllocatedPairTarget,
    BarSnapshot,
    PairBarBundle,
    PortfolioAllocation,
    PortfolioState,
    RawPairTarget,
)
from core.modules.strategy.base import BaseStrategy
from core.modules.strategy.config import (
    EstimatorConfig,
    ExecutionConfig,
    PairDefinition,
    PortfolioConfig,
    SignalConfig,
    SizingConfig,
)
from core.modules.strategy.pair_pipeline import PairPipeline, _dataclass_to_dict, _merge_para


class MultiPairStrategy(BaseStrategy):
    """Top-level strategy: dispatch pair pipelines, allocate portfolio, create market orders."""

    def __init__(
        self,
        pairs: list[PairDefinition],
        estimator_ctor,
        estimator_cfg: EstimatorConfig,
        signal_ctor=None,
        signal_cfg: SignalConfig | None = None,
        sizing_ctor=None,
        sizing_cfg: SizingConfig | None = None,
        portfolio_ctor=None,
        portfolio_cfg: PortfolioConfig | None = None,
        execution_cfg: ExecutionConfig | None = None,
        name: str = "multi_pair",
    ):
        super().__init__()
        self.name = name
        self.pairs = pairs
        self._pair_defs_by_id = {pair_def.pair_id: pair_def for pair_def in pairs}
        self.pipelines: dict[str, PairPipeline] = {}
        self.portfolio_state = PortfolioState()
        self.order_planner = OrderPlanner(execution_cfg)

        for pair_def in pairs:
            if not pair_def.enabled:
                continue
            self.pipelines[pair_def.pair_id] = PairPipeline(
                pair_def,
                estimator_ctor,
                estimator_cfg,
                signal_ctor,
                signal_cfg,
                sizing_ctor,
                sizing_cfg,
                execution_cfg,
            )

        if portfolio_ctor and portfolio_cfg:
            self.portfolio = portfolio_ctor(**_dataclass_to_dict(portfolio_cfg))
            if hasattr(self.portfolio, "total_pair_count"):
                self.portfolio.total_pair_count = len(self.pipelines)
        else:
            self.portfolio = None

        self.estimator_cfg = estimator_cfg
        self.signal_cfg = signal_cfg
        self.sizing_cfg = sizing_cfg
        self.portfolio_cfg = portfolio_cfg
        self.max_entries_per_pair = max(1, int(getattr(portfolio_cfg, "max_entries_per_pair", 1) or 1))
        self.add_cooldown_bars = max(0, int(getattr(portfolio_cfg, "add_cooldown_bars", 0) or 0))
        self.min_hold_bars = max(0, int(getattr(portfolio_cfg, "min_hold_bars", 0) or 0))
        self._pair_bar_indices: dict[str, int] = {}

        self.signal = self
        self.last_state: dict[str, Any] = {}

    def snapshot_signal_state(self, ts, compact: bool = False):
        state = {"ts": ts}
        for pair_id, pipeline in self.pipelines.items():
            runtime_state = pipeline.state
            prefix = pair_id + "_"
            state[prefix + "zscore"] = getattr(pipeline, "_last_zscore", None)
            if compact:
                continue

            result = getattr(pipeline, "_last_result", None)
            estimator = getattr(result, "estimator", None)
            if estimator is not None:
                state[prefix + "alpha"] = estimator.alpha
                state[prefix + "beta"] = estimator.beta
                state[prefix + "spread_beta"] = estimator.spread_beta
                state[prefix + "spread_mean"] = estimator.spread_mean
                state[prefix + "spread_std"] = estimator.spread_std
                state[prefix + "latest_spread"] = estimator.latest_spread
            state[prefix + "position_exit_zscore"] = getattr(
                pipeline, "_last_position_exit_zscore", None
            )
            state[prefix + "position_exit_zscore_method"] = (
                pipeline.position_exit_zscore.method
            )
            state[prefix + "action"] = getattr(pipeline, "_last_action", "none")
            state[prefix + "side"] = runtime_state.sizing_state.position_side
            state[prefix + "entry_count"] = runtime_state.sizing_state.entry_count
            state[prefix + "pair_full"] = runtime_state.sizing_state.pair_full
            state[prefix + "cointegration_pass"] = runtime_state.estimator_state.cointegration_pass
            state[prefix + "cointegration_pvalue"] = runtime_state.estimator_state.cointegration_pvalue
            state[prefix + "cointegration_block_open"] = runtime_state.estimator_state.cointegration_block_open
            state[prefix + "next_model_update_index"] = runtime_state.estimator_state.next_model_update_index
            state[prefix + "last_model_update_skip_index"] = runtime_state.estimator_state.last_model_update_skip_index
            state[prefix + "current_gross_notional"] = runtime_state.sizing_state.current_gross_notional
            state[prefix + "pair_cap_gross"] = runtime_state.sizing_state.pair_cap_gross
            state[prefix + "entry_gross_cap"] = runtime_state.sizing_state.entry_gross_cap
            state[prefix + "remaining_pair_gross"] = runtime_state.sizing_state.remaining_pair_gross
            state[prefix + "add_cooldown_remaining_bars"] = runtime_state.sizing_state.add_cooldown_remaining_bars
            state[prefix + "min_hold_remaining_bars"] = runtime_state.sizing_state.min_hold_remaining_bars
            state[prefix + "block_reason"] = runtime_state.last_block_reason
        self.last_state = state
        return state

    def _record_signal_state(self, ts):
        return self.snapshot_signal_state(ts)

    def on_bar(self, bars: dict):
        ts = self._extract_ts(bars)
        self._sync_account_context()
        bar_index = getattr(self, "_global_bar_index", 0) + 1
        self._global_bar_index = bar_index

        results = {}
        bundles = {}
        for pair_id, pipeline in self.pipelines.items():
            pair_bar_index = self._pair_bar_indices.get(pair_id, 0) + 1
            bundle = self._build_bundle(pair_id, pipeline.pair_def, bars, ts, pair_bar_index)
            if bundle is None:
                continue
            bundles[pair_id] = bundle
            self._pair_bar_indices[pair_id] = pair_bar_index
            result = pipeline.on_bar(bundle, compute_sizing=False)
            results[pair_id] = result
            if result.signal:
                pipeline._last_zscore = result.signal.zscore
                pipeline._last_action = result.signal.action

        orders: list[Order] = []
        entry_candidates: dict[str, dict[str, Any]] = {}
        entry_targets: dict[str, RawPairTarget] = {}
        for pair_id, pipeline in self.pipelines.items():
            result = results.get(pair_id)
            if result is None or result.signal is None:
                continue
            pair_def = self._pair_defs_by_id.get(pair_id)
            if pair_def is None:
                continue

            signal = result.signal
            fallback_hedge = pipeline.state.estimator_state.hedge_beta or 1.0
            bundle = bundles.get(pair_id)
            self._refresh_pair_capacity(pair_id, pipeline, pair_def, bundle)

            if signal.action == "close":
                if self._can_close_pair(pipeline, signal.bar_index):
                    orders.extend(self._close_orders(pair_id, pipeline, pair_def, fallback_hedge))
            elif signal.action in ("open", "add") and signal.side:
                candidate = self._prepare_entry_candidate(
                    pair_id,
                    pipeline,
                    pair_def,
                    signal,
                    result,
                    fallback_hedge,
                    bundle,
                )
                if candidate is not None:
                    entry_candidates[pair_id] = candidate
                    entry_targets[pair_id] = candidate["raw_target"]

        allocation = self._allocate_entry_targets(entry_targets, bar_index)
        orders.extend(self._orders_from_entry_allocation(entry_candidates, allocation))

        self._update_portfolio_state(bundles, ts, bar_index, allocation)
        return orders

    def _sync_account_context(self):
        equity = getattr(self, "_current_equity", None)
        if equity is None or equity <= 0:
            return
        if self.portfolio is not None and hasattr(self.portfolio, "equity"):
            self.portfolio.equity = float(equity)

    def _prepare_entry_candidate(
        self,
        pair_id,
        pipeline,
        pair_def,
        signal,
        result,
        fallback_hedge,
        bundle,
    ) -> dict[str, Any] | None:
        bar_index = signal.bar_index
        sizing_state = pipeline.state.sizing_state
        cointegration_decision = self._cointegration_entry_decision(result)
        if not cointegration_decision["allowed"]:
            sizing_state.last_schedule_reason = cointegration_decision["reason"]
            pipeline.state.last_block_reason = cointegration_decision["reason"]
            return None

        decision = self._entry_schedule_decision(pipeline, signal, bar_index)
        if not decision["allowed"]:
            sizing_state.last_schedule_reason = decision["reason"]
            pipeline.state.last_block_reason = decision["reason"]
            return None

        signal_para = _merge_para(getattr(result, "para", {}), getattr(signal, "para", {}))
        signal_para = _merge_para(signal_para, {
            "portfolio": {
                "scheduled_action": decision["action"],
                "entry_count_before": sizing_state.entry_count,
                "max_entries_per_pair": self.max_entries_per_pair,
                "add_cooldown_bars": self.add_cooldown_bars,
                "min_hold_bars": self.min_hold_bars,
                "current_gross_notional": sizing_state.current_gross_notional,
                "pair_cap_gross": sizing_state.pair_cap_gross,
                "entry_gross_cap": sizing_state.entry_gross_cap,
                "remaining_pair_gross": sizing_state.remaining_pair_gross,
                "position_id": sizing_state.position_id,
            },
            "pair": {
                "pair_id": pair_id,
                "x_symbol": pair_def.x_symbol,
                "y_symbol": pair_def.y_symbol,
                "x_exchange": pair_def.x_exchange,
                "y_exchange": pair_def.y_exchange,
            },
        })
        sizing_signal = replace(
            signal,
            action="open",
            para=signal_para,
            reason=f"{signal.reason}; scheduled {decision['action']}",
        )
        raw_target = self._compute_direct_target(
            pair_id, pipeline, pair_def, sizing_signal, result, fallback_hedge, bundle
        )
        if raw_target is None:
            reason = "sizing returned no target"
            sizing_state.last_schedule_reason = reason
            pipeline.state.last_block_reason = reason
            return None

        raw_target.para = _merge_para(raw_target.para, signal_para)
        raw_target.reason = f"{raw_target.reason}; scheduled_{decision['action']}"
        result.raw_target = raw_target
        sizing_state.last_schedule_reason = decision["reason"]
        pipeline.state.last_block_reason = ""
        pipeline._last_action = decision["action"]
        return {
            "pair_id": pair_id,
            "pair_def": pair_def,
            "signal": signal,
            "scheduled_action": decision["action"],
            "raw_target": raw_target,
            "fallback_hedge": fallback_hedge,
            "position_id": sizing_state.position_id if decision["action"] == "add" else None,
        }

    @staticmethod
    def _cointegration_entry_decision(result) -> dict[str, Any]:
        estimator = getattr(result, "estimator", None)
        if estimator is None or not getattr(estimator, "cointegration_block_open", False):
            return {"allowed": True, "reason": "cointegration ok"}
        reason = getattr(estimator, "cointegration_reason", "") or "cointegration check failed"
        pvalue = getattr(estimator, "cointegration_pvalue", None)
        if pvalue is not None and "pvalue" not in reason:
            reason = f"{reason}; pvalue={float(pvalue):.4f}"
        return {"allowed": False, "reason": f"cointegration blocked open: {reason}"}

    def _entry_schedule_decision(self, pipeline, signal, bar_index) -> dict[str, Any]:
        state = pipeline.state.sizing_state
        side = signal.side
        if not side:
            return {"allowed": False, "action": "none", "reason": "entry signal has no side"}

        eps = 1e-9
        if state.remaining_pair_gross <= eps:
            state.pair_full = True
            return {"allowed": False, "action": "none", "reason": "pair gross cap reached"}

        if state.position_side is None:
            return {"allowed": True, "action": "open", "reason": "first entry"}

        if state.position_side != side:
            return {
                "allowed": False,
                "action": "none",
                "reason": f"opposite entry ignored while holding {state.position_side}",
            }

        if state.entry_count >= self.max_entries_per_pair:
            state.pair_full = True
            return {"allowed": False, "action": "none", "reason": "max entries per pair reached"}

        if state.add_cooldown_remaining_bars > 0:
            return {
                "allowed": False,
                "action": "none",
                "reason": f"add cooldown {state.add_cooldown_remaining_bars} bars remaining",
            }

        return {"allowed": True, "action": "add", "reason": f"add entry #{state.entry_count + 1}"}

    def _allocate_entry_targets(
        self,
        entry_targets: dict[str, RawPairTarget],
        bar_index: int,
    ):
        if not entry_targets:
            return None
        if self.portfolio is not None:
            return self.portfolio.allocate(self.portfolio_state, entry_targets, bar_index)

        pair_targets = {}
        for pair_id, raw in entry_targets.items():
            pair_targets[pair_id] = AllocatedPairTarget(
                pair_id=pair_id,
                selected=True,
                side=raw.side,
                hedge_ratio=raw.hedge_ratio,
                final_x_notional=raw.x_notional,
                final_y_notional=raw.y_notional,
                final_x_quantity=raw.x_quantity,
                final_y_quantity=raw.y_quantity,
                portfolio_weight=0.0,
                allocation_reason="no portfolio allocator",
                para=raw.para,
            )
        return PortfolioAllocation(
            bar_index=bar_index,
            selected_pair_ids=list(pair_targets),
            pair_targets=pair_targets,
            portfolio_gross_exposure=sum(
                at.final_x_notional + at.final_y_notional for at in pair_targets.values()
            ),
            portfolio_net_exposure=self._allocated_net_exposure(pair_targets),
            reason="no portfolio allocator",
        )

    def _orders_from_entry_allocation(self, entry_candidates, allocation) -> list[Order]:
        if allocation is None:
            return []

        orders: list[Order] = []
        for pair_id, candidate in entry_candidates.items():
            alloc_target = allocation.pair_targets.get(pair_id)
            if alloc_target is None or not alloc_target.selected:
                continue
            pair_def = candidate["pair_def"]
            raw_target = candidate["raw_target"]
            alloc_target.side = raw_target.side
            alloc_target.para = raw_target.para
            target_hedge = alloc_target.hedge_ratio or raw_target.hedge_ratio or candidate["fallback_hedge"]
            if alloc_target.final_x_quantity <= 0 or alloc_target.final_y_quantity <= 0:
                continue
            orders.extend(self.order_planner.orders_for_target(
                alloc_target,
                pair_def,
                x_exchange=pair_def.x_exchange,
                y_exchange=pair_def.y_exchange,
                hedge_ratio=target_hedge,
                action="open",
                position_id=candidate["position_id"],
            ))
        return orders

    def _refresh_pair_capacity(self, pair_id, pipeline, pair_def, bundle):
        state = pipeline.state.sizing_state
        if bundle is None:
            return

        equity = self._current_equity_value(pipeline)
        pair_count = self._pair_cap_count()
        pair_cap = equity / pair_count if equity > 0 else 0.0
        entry_cap = pair_cap / self.max_entries_per_pair if self.max_entries_per_pair > 0 else pair_cap

        x_price = bundle.x_bar.close if bundle.x_bar else 0.0
        y_price = bundle.y_bar.close if bundle.y_bar else 0.0
        current_gross = abs(state.x_quantity) * x_price + abs(state.y_quantity) * y_price
        remaining = max(0.0, pair_cap - current_gross)

        cooldown_remaining = 0
        min_hold_remaining = 0
        if state.position_side is not None and state.last_entry_bar_index is not None:
            elapsed = max(0, bundle.bar_index - state.last_entry_bar_index)
            cooldown_remaining = max(0, self.add_cooldown_bars - elapsed)
            min_hold_remaining = max(0, self.min_hold_bars - elapsed)

        eps = max(1e-9, pair_cap * 1e-9)
        state.current_gross_notional = current_gross
        state.pair_cap_gross = pair_cap
        state.entry_gross_cap = entry_cap
        state.remaining_pair_gross = remaining
        state.add_cooldown_remaining_bars = cooldown_remaining
        state.min_hold_remaining_bars = min_hold_remaining
        state.pair_full = (
            state.position_side is not None
            and (state.entry_count >= self.max_entries_per_pair or remaining <= eps)
        )

    def _pair_cap_count(self) -> int:
        return max(1, int(getattr(self.portfolio_cfg, "max_open_pairs", 1) or 1))

    def _current_equity_value(self, pipeline=None) -> float:
        equity = getattr(self, "_current_equity", None)
        if equity is not None and equity > 0:
            return float(equity)
        if self.portfolio is not None and getattr(self.portfolio, "equity", 0) > 0:
            return float(self.portfolio.equity)
        return 0.0

    def _update_portfolio_state(self, bundles, ts, bar_index, entry_allocation):
        existing_targets = self._existing_position_targets(bundles, ts, bar_index)
        selected_ids = list(existing_targets)
        gross = sum(t.gross_notional for t in existing_targets.values())
        net = self._raw_targets_net_exposure(existing_targets)
        symbol_exposure = self._raw_symbol_exposure(existing_targets)

        if entry_allocation is not None:
            for pair_id in entry_allocation.selected_pair_ids:
                if pair_id not in selected_ids:
                    selected_ids.append(pair_id)
            gross += entry_allocation.portfolio_gross_exposure
            net += entry_allocation.portfolio_net_exposure

        self.portfolio_state.ready = True
        self.portfolio_state.warmup_bars_seen += 1
        self.portfolio_state.selected_pair_ids = selected_ids
        self.portfolio_state.portfolio_gross_exposure = gross
        self.portfolio_state.portfolio_net_exposure = net
        self.portfolio_state.symbol_exposure_map = symbol_exposure
        self.portfolio_state.last_allocation_reason = (
            entry_allocation.reason if entry_allocation is not None else "existing positions only"
        )

    def _allocated_net_exposure(self, pair_targets: dict[str, AllocatedPairTarget]) -> float:
        total = 0.0
        for target in pair_targets.values():
            if target.side == "long_x":
                total += target.final_x_notional - target.final_y_notional
            elif target.side == "short_x":
                total += target.final_y_notional - target.final_x_notional
            else:
                total += target.final_x_notional - target.final_y_notional
        return total

    def _raw_targets_net_exposure(self, targets: dict[str, RawPairTarget]) -> float:
        total = 0.0
        for target in targets.values():
            if target.side == "long_x":
                total += target.x_notional - target.y_notional
            elif target.side == "short_x":
                total += target.y_notional - target.x_notional
            else:
                total += target.x_notional - target.y_notional
        return total

    def _raw_symbol_exposure(self, targets: dict[str, RawPairTarget]) -> dict[str, float]:
        exposure: dict[str, float] = {}
        for pair_id, target in targets.items():
            pipeline = self.pipelines.get(pair_id)
            if pipeline is None:
                continue
            pair_def = pipeline.pair_def
            exposure[pair_def.x_symbol] = exposure.get(pair_def.x_symbol, 0.0) + abs(target.x_notional)
            exposure[pair_def.y_symbol] = exposure.get(pair_def.y_symbol, 0.0) + abs(target.y_notional)
        exposure["_gross"] = sum(v for k, v in exposure.items() if k != "_gross")
        return exposure

    def _close_orders(self, pair_id, pipeline, pair_def, fallback_hedge) -> list[Order]:
        sizing_state = pipeline.state.sizing_state
        if sizing_state.x_quantity <= 0 or sizing_state.y_quantity <= 0:
            return []

        close_hedge = sizing_state.target_hedge_ratio or fallback_hedge
        target = AllocatedPairTarget(
            pair_id=pair_id,
            selected=True,
            side=sizing_state.position_side,
            hedge_ratio=close_hedge,
            final_x_quantity=sizing_state.x_quantity,
            final_y_quantity=sizing_state.y_quantity,
        )
        orders = self.order_planner.orders_for_target(
            target,
            pair_def,
            x_exchange=pair_def.x_exchange,
            y_exchange=pair_def.y_exchange,
            hedge_ratio=close_hedge,
            action="close",
        )
        for order in orders:
            order.position_id = sizing_state.position_id
        return orders

    def _can_close_pair(self, pipeline, bar_index) -> bool:
        sizing_state = pipeline.state.sizing_state
        if self.min_hold_bars <= 0 or sizing_state.position_side is None:
            return True
        if sizing_state.last_entry_bar_index is None:
            return True

        elapsed = max(0, int(bar_index) - int(sizing_state.last_entry_bar_index))
        remaining = max(0, self.min_hold_bars - elapsed)
        sizing_state.min_hold_remaining_bars = remaining
        if remaining <= 0:
            return True

        reason = f"min hold {remaining} bars remaining"
        sizing_state.last_schedule_reason = reason
        pipeline.state.last_block_reason = reason
        return False

    def _compute_direct_target(
        self,
        pair_id,
        pipeline,
        pair_def,
        signal,
        result,
        fallback_hedge,
        bundle,
    ):
        if not hasattr(pipeline, "sizing") or not pipeline.sizing:
            return None
        if bundle is None:
            return None

        x_price = bundle.x_bar.close if bundle.x_bar else 0
        y_price = bundle.y_bar.close if bundle.y_bar else 0
        target = pipeline.sizing.compute(
            pipeline.state.sizing_state,
            signal,
            result.estimator,
            x_price,
            y_price,
            x_history=pipeline.state.estimator_state.x_close_history,
            y_history=pipeline.state.estimator_state.y_close_history,
            para=getattr(result, "para", {}),
        )
        if not target.ready:
            return None
        if not target.hedge_ratio:
            target.hedge_ratio = fallback_hedge
        return target

    def on_trades_filled(self, trades: list):
        grouped = {}
        for trade in trades or []:
            pair_id = getattr(trade, "pair_id", None)
            group_id = str(getattr(trade, "group_id", ""))
            if pair_id is None or not group_id:
                continue
            grouped.setdefault((pair_id, group_id), []).append(trade)

        for (pair_id, _group_id), group in grouped.items():
            pipeline = self.pipelines.get(pair_id)
            if pipeline is None:
                continue
            sizing_state = pipeline.state.sizing_state
            action = str(getattr(group[0], "action", ""))

            if action == "open":
                pair_side = self._filled_pair_side(pipeline.pair_def, group)
                if pair_side is not None:
                    sizing_state.position_side = pair_side
                sizing_state.position_id = self._first_attr(group, "position_id", sizing_state.position_id)
                sizing_state.target_hedge_ratio = self._first_attr(
                    group, "target_hedge_ratio", sizing_state.target_hedge_ratio
                )
                for trade in group:
                    self._apply_pair_quantity_delta(pipeline, sizing_state, trade, sign=1.0)
                sizing_state.entry_count += 1
                sizing_state.last_entry_bar_index = pipeline.state.last_bar_index
                sizing_state.pair_full = sizing_state.entry_count >= self.max_entries_per_pair
                pipeline.on_position_opened()

            elif action == "close":
                for trade in group:
                    self._apply_pair_quantity_delta(pipeline, sizing_state, trade, sign=-1.0)
                if sizing_state.x_quantity < 1e-12 and sizing_state.y_quantity < 1e-12:
                    sizing_state.position_side = None
                    sizing_state.position_id = None
                    sizing_state.target_hedge_ratio = None
                    sizing_state.last_entry_bar_index = None
                    sizing_state.x_quantity = 0.0
                    sizing_state.y_quantity = 0.0
                    sizing_state.entry_count = 0
                    sizing_state.current_gross_notional = 0.0
                    sizing_state.remaining_pair_gross = sizing_state.pair_cap_gross
                    sizing_state.pair_full = False
                    sizing_state.add_cooldown_remaining_bars = 0
                    sizing_state.min_hold_remaining_bars = 0
                    pipeline.on_position_closed()

    def _filled_pair_side(self, pair_def, trades):
        x_side = None
        y_side = None
        for trade in trades:
            symbol = getattr(trade, "symbol", "")
            side = str(getattr(trade, "side", ""))
            if symbol == pair_def.x_symbol:
                x_side = side
            elif symbol == pair_def.y_symbol:
                y_side = side

        if x_side == "buy" and y_side == "sell":
            return "long_x"
        if x_side == "sell" and y_side == "buy":
            return "short_x"
        return None

    def _apply_pair_quantity_delta(self, pipeline, sizing_state, trade, sign: float):
        symbol = getattr(trade, "symbol", "")
        quantity = float(getattr(trade, "quantity", 0))
        if symbol == pipeline.pair_def.x_symbol:
            sizing_state.x_quantity = max(0.0, sizing_state.x_quantity + sign * quantity)
        elif symbol == pipeline.pair_def.y_symbol:
            sizing_state.y_quantity = max(0.0, sizing_state.y_quantity + sign * quantity)

    def _first_attr(self, items, attr, default=None):
        for item in items:
            value = getattr(item, attr, None)
            if value is not None:
                return value
        return default

    def on_orders_rejected(self, orders: list):
        for order in orders:
            pair_id = getattr(order, "pair_id", None)
            if pair_id is None:
                continue
            pipeline = self.pipelines.get(pair_id)
            if pipeline is None:
                continue
            pipeline.state.signal_state.last_open_reject_bar_index = pipeline.state.last_bar_index

    def _build_bundle(self, pair_id, pair_def, bars, ts, bar_index):
        x_bar = self._find_bar(bars, pair_def.x_exchange, pair_def.x_symbol)
        y_bar = self._find_bar(bars, pair_def.y_exchange, pair_def.y_symbol)
        if x_bar is None or y_bar is None:
            return None
        return PairBarBundle(pair_id=pair_id, ts=ts, bar_index=bar_index, x_bar=x_bar, y_bar=y_bar)

    def _existing_position_targets(self, bundles, ts, bar_index) -> dict[str, RawPairTarget]:
        targets: dict[str, RawPairTarget] = {}
        if not any(pipeline.state.sizing_state.position_side for pipeline in self.pipelines.values()):
            return targets
        for pair_id, pipeline in self.pipelines.items():
            sizing_state = pipeline.state.sizing_state
            side = sizing_state.position_side
            if side is None:
                continue
            bundle = bundles.get(pair_id)
            if bundle is None:
                continue
            x_price = bundle.x_bar.close if bundle.x_bar else 0.0
            y_price = bundle.y_bar.close if bundle.y_bar else 0.0
            x_notional = abs(sizing_state.x_quantity) * x_price
            y_notional = abs(sizing_state.y_quantity) * y_price
            targets[pair_id] = RawPairTarget(
                pair_id=pair_id,
                ts=ts,
                bar_index=bar_index,
                ready=True,
                side=side,
                x_notional=x_notional,
                y_notional=y_notional,
                x_quantity=abs(sizing_state.x_quantity),
                y_quantity=abs(sizing_state.y_quantity),
                hedge_ratio=sizing_state.target_hedge_ratio or pipeline.state.estimator_state.hedge_beta or 1.0,
                gross_notional=x_notional + y_notional,
                signal_strength=1e12,
                reason="existing_position",
                para={
                    "portfolio": {"existing_position": True},
                    "pair": {
                        "pair_id": pair_id,
                        "x_symbol": pipeline.pair_def.x_symbol,
                        "y_symbol": pipeline.pair_def.y_symbol,
                        "x_exchange": pipeline.pair_def.x_exchange,
                        "y_exchange": pipeline.pair_def.y_exchange,
                    },
                },
            )
        return targets

    @staticmethod
    def _find_bar(bars, exchange, symbol):
        exchange_bars = bars.get(exchange, {})
        bar = exchange_bars.get(symbol)
        if bar is None:
            return None
        if isinstance(bar, dict):
            return BarSnapshot(
                ts=bar.get("ts", 0),
                open=float(bar.get("open", 0)),
                high=float(bar.get("high", 0)),
                close=float(bar.get("close", 0)),
                low=float(bar.get("low", 0)),
                volume=float(bar.get("volume", 0)),
                exchange=exchange,
                symbol=symbol,
            )
        if isinstance(bar, list) and len(bar) >= 6:
            return BarSnapshot(
                bar[0], bar[1], bar[2], bar[3], bar[4], bar[5], exchange, symbol
            )
        return None

    @staticmethod
    def _extract_ts(bars):
        for exchange, exchange_bars in bars.items():
            if str(exchange).startswith("_") or not isinstance(exchange_bars, dict):
                continue
            for bar in exchange_bars.values():
                if isinstance(bar, dict):
                    return bar.get("ts", 0)
                if isinstance(bar, list) and bar:
                    return bar[0]
        return 0
