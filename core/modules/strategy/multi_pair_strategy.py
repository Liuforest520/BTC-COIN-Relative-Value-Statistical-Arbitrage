"""MultiPairStrategy: orchestrates PairPipelines + PortfolioAllocator + OrderPlanner."""
from __future__ import annotations

from math import ceil, isfinite
from uuid import uuid4
from dataclasses import replace
from typing import Any

from core.modules.data.resampler import timeframe_bars, timeframe_to_minutes
from core.modules.execution.order_planner import OrderPlanner
from core.modules.logger import logger
from core.modules.models import Order
from core.modules.models.pipeline_types import (
    AllocatedPairTarget,
    BarSnapshot,
    PairBarBundle,
    PortfolioAllocation,
    PortfolioState,
    RawPairTarget,
)
from core.modules.strategy.position_protection import (
    mark_net_pnl,
    net_pnl_at_x,
    solve_zero_net_x_price,
    supported as protection_supported,
)
from core.modules.strategy.protection import ProtectionContext, ProtectionManager
from core.modules.strategy.rebalance import (
    RebalanceCandidate,
    RebalanceManager,
    RebalancePosition,
)
from core.modules.strategy.base import BaseStrategy
from core.modules.strategy.config import (
    EstimatorConfig,
    ExecutionConfig,
    PairDefinition,
    PortfolioConfig,
    SignalConfig,
    SizingConfig,
    ProtectionConfig,
    RebalanceConfig,
)
from core.modules.strategy.pair_pipeline import PairPipeline, _dataclass_to_dict, _merge_para


def _decision_bars(value, timeframe_minutes: int, label: str) -> int:
    """Convert a legacy source-minute bar count to model decision bars."""
    raw = int(value or 0)
    if raw < 0:
        raise ValueError(f"{label} must be non-negative")
    if raw == 0:
        return 0
    return timeframe_bars(raw, timeframe_minutes, label=label)

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
        protection_cfg: ProtectionConfig | None = None,
        rebalance_cfg: RebalanceConfig | None = None,
        fee_rate: float = 0.0005,
        slippage_bps: float = 1.0,
        name: str = "multi_pair",
    ):
        super().__init__()
        self.name = name
        self.pairs = pairs
        self._pair_defs_by_id = {pair_def.pair_id: pair_def for pair_def in pairs}
        self.pipelines: dict[str, PairPipeline] = {}
        self.portfolio_state = PortfolioState()
        self.order_planner = OrderPlanner(execution_cfg)
        self.estimator_cfg = estimator_cfg
        self.signal_cfg = signal_cfg or SignalConfig()
        self.sizing_cfg = sizing_cfg
        self.portfolio_cfg = portfolio_cfg or PortfolioConfig()
        self.protection_cfg = protection_cfg or ProtectionConfig()
        self.rebalance_cfg = rebalance_cfg or RebalanceConfig()
        # Strategy bar counters advance only when a complete model bar is
        # available. Keep legacy bar-valued strategy settings in source-minute
        # units and convert them once for the active model timeframe.
        self.model_timeframe_minutes = timeframe_to_minutes(
            getattr(estimator_cfg, "model_timeframe", "1m")
        )
        raw_lookback = float(getattr(estimator_cfg, "model_lookback_bars", 10080))
        raw_update = float(getattr(estimator_cfg, "model_update_interval_bars", 240))
        if self.model_timeframe_minutes > 1:
            lookback_bars = timeframe_bars(
                raw_lookback,
                self.model_timeframe_minutes,
                label="estimator.model_lookback_bars",
            )
            update_bars = timeframe_bars(
                raw_update,
                self.model_timeframe_minutes,
                label="estimator.model_update_interval_bars",
            )
            logger.info(
                "model timeframe={} lookback {}m -> {} bars ({}m); update {}m -> {} bars ({}m)",
                getattr(estimator_cfg, "model_timeframe", "1m"),
                int(raw_lookback) if raw_lookback.is_integer() else raw_lookback,
                lookback_bars,
                lookback_bars * self.model_timeframe_minutes,
                int(raw_update) if raw_update.is_integer() else raw_update,
                update_bars,
                update_bars * self.model_timeframe_minutes,
            )
            if raw_lookback % self.model_timeframe_minutes or raw_update % self.model_timeframe_minutes:
                logger.warning(
                    "model timeframe={} uses ceil conversion for a non-divisible source-minute window",
                    getattr(estimator_cfg, "model_timeframe", "1m"),
                )
        self.portfolio_cfg = replace(
            self.portfolio_cfg,
            add_cooldown_bars=_decision_bars(
                self.portfolio_cfg.add_cooldown_bars,
                self.model_timeframe_minutes,
                "portfolio.add_cooldown_bars",
            ),
            min_hold_bars=_decision_bars(
                self.portfolio_cfg.min_hold_bars,
                self.model_timeframe_minutes,
                "portfolio.min_hold_bars",
            ),
        )
        self.protection_cfg = replace(
            self.protection_cfg,
            stop_loss_freeze_bars=_decision_bars(
                self.protection_cfg.stop_loss_freeze_bars,
                self.model_timeframe_minutes,
                "protection.stop_loss_freeze_bars",
            ),
            pair_loss_stop_freeze_bars=_decision_bars(
                self.protection_cfg.pair_loss_stop_freeze_bars,
                self.model_timeframe_minutes,
                "protection.pair_loss_stop_freeze_bars",
            ),
            take_profit_freeze_bars=_decision_bars(
                self.protection_cfg.take_profit_freeze_bars,
                self.model_timeframe_minutes,
                "protection.take_profit_freeze_bars",
            ),
            max_holding_time_freeze_bars=_decision_bars(
                self.protection_cfg.max_holding_time_freeze_bars,
                self.model_timeframe_minutes,
                "protection.max_holding_time_freeze_bars",
            ),
        )
        self.protection_manager = ProtectionManager(self.protection_cfg)
        self.rebalance_manager = RebalanceManager(self.rebalance_cfg)
        self.fee_rate = max(0.0, float(fee_rate))
        self.slippage_rate = max(0.0, float(slippage_bps)) / 10000.0
        self.frozen_model_supported = protection_supported(
            self.estimator_cfg.regression_method,
            self.estimator_cfg.position_update_policy,
            True,
        )
        self.protection_supported = protection_supported(
            self.estimator_cfg.regression_method,
            self.estimator_cfg.position_update_policy,
            self.protection_cfg.enabled
            or self.protection_cfg.max_holding_time_enabled
            or self.protection_cfg.pair_loss_stop_enabled,
        )

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

        # Validate the effective estimator lookback up front when the feature
        # is enabled. Pair-level lookback overrides are already applied by
        # PairPipeline, so each pair gets the same semantics as its model.
        if self.protection_cfg.max_holding_time_enabled:
            for pipeline in self.pipelines.values():
                self._max_holding_bars_for_pipeline(pipeline)

        if portfolio_ctor and portfolio_cfg:
            self.portfolio = portfolio_ctor(**_dataclass_to_dict(self.portfolio_cfg))
            if hasattr(self.portfolio, "total_pair_count"):
                self.portfolio.total_pair_count = len(self.pipelines)
        else:
            self.portfolio = None

        self.max_entries_per_pair = max(1, int(getattr(self.portfolio_cfg, "max_entries_per_pair", 1) or 1))
        self.add_cooldown_bars = max(0, int(getattr(self.portfolio_cfg, "add_cooldown_bars", 0) or 0))
        self.min_hold_bars = max(0, int(getattr(self.portfolio_cfg, "min_hold_bars", 0) or 0))
        self._pair_bar_indices: dict[str, int] = {}
        self._pending_open_pair_ids: set[str] = set()
        self._pending_rebalance_pair_ids: set[str] = set()

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
                state[prefix + "last_model_update_index"] = estimator.last_model_update_index
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
            state[prefix + "max_holding_bars"] = runtime_state.protection_state.max_holding_bars
            state[prefix + "max_holding_deadline_bar"] = runtime_state.protection_state.max_holding_deadline_bar
            state[prefix + "reopen_lock_pending"] = runtime_state.protection_state.reopen_lock_pending
            state[prefix + "pair_loss_stop_freeze_until_bar"] = runtime_state.protection_state.pair_loss_stop_freeze_until_bar
            state[prefix + "protection_rule"] = runtime_state.protection_state.protection_rule
            state[prefix + "freeze_rule"] = runtime_state.protection_state.freeze_rule
            state[prefix + "freeze_bars"] = runtime_state.protection_state.freeze_bars
            state[prefix + "freeze_until_bar"] = runtime_state.protection_state.freeze_until_bar
            state[prefix + "last_exit_class"] = runtime_state.protection_state.last_exit_class
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
            self._maybe_release_protection_lock(pipeline)
            if result.signal:
                signal_details = (getattr(result.signal, "para", {}) or {}).get("signal") or {}
                pipeline._last_zscore = signal_details.get(
                    "standard_zscore", result.signal.zscore
                )
                pipeline._last_action = result.signal.action

        orders: list[Order] = []
        entry_candidates: dict[str, dict[str, Any]] = {}
        entry_targets: dict[str, RawPairTarget] = {}
        self.portfolio_state.open_pair_ids = self._open_pair_ids()
        self.portfolio_state.open_pair_count = len(self.portfolio_state.open_pair_ids)
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

            # Protective exits have priority over the ordinary Z-score exit
            # and are evaluated on the current bar close.  The generated
            # market orders are still filled by Exchange on the next bar.
            protection_decision = self._protection_exit_decision(pipeline, bundle)
            if protection_decision is not None:
                protection_orders = self._close_orders(
                    pair_id,
                    pipeline,
                    pair_def,
                    fallback_hedge,
                    exit_reason=protection_decision["reason"],
                    protection_trigger=protection_decision["trigger"],
                )
                if protection_orders:
                    orders.extend(protection_orders)
                else:
                    pipeline.state.protection_state.pending_exit = False
                continue
            if pipeline.state.protection_state.pending_exit:
                continue

            if signal.action == "close":
                if self._can_close_pair(pipeline, signal.bar_index):
                    orders.extend(self._close_orders(pair_id, pipeline, pair_def, fallback_hedge))
            elif signal.action in ("open", "add") and signal.side:
                already_holding = self._pair_holds_position(pipeline)
                if pair_id in self._pending_open_pair_ids:
                    reason = "open order pending fill"
                    pipeline.state.sizing_state.last_schedule_reason = reason
                    pipeline.state.last_block_reason = reason
                    continue
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

        # First allocate every entry that can be funded normally. Those opens
        # are not limited to one candidate. Only candidates skipped for lack
        # of minimum capital are passed to the independent rebalance manager.
        allocation = self._allocate_entry_targets(entry_targets, bar_index)
        direct_orders = self._orders_from_entry_allocation(entry_candidates, allocation)
        allocation_report = getattr(allocation, "constraint_report", {}) or {}
        underfunded_ids = set(
            allocation_report.get("underfunded_pair_ids", []) or []
        )
        underfunded_candidates = {
            pair_id: candidate
            for pair_id, candidate in entry_candidates.items()
            if pair_id in underfunded_ids
        }
        if self.rebalance_cfg.enabled and underfunded_candidates:
            qualified_candidates = {}
            for pair_id, candidate in underfunded_candidates.items():
                rebalance_candidate = self._rebalance_candidate_from_entry(
                    pair_id, candidate
                )
                if (
                    rebalance_candidate is not None
                    and self.rebalance_manager.quality_allowed(rebalance_candidate)
                ):
                    qualified_candidates[pair_id] = candidate
                else:
                    pipeline = self.pipelines.get(pair_id)
                    if pipeline is not None:
                        reason = "rebalance candidate quality gate failed"
                        pipeline.state.sizing_state.last_schedule_reason = reason
                        pipeline.state.last_block_reason = reason
            underfunded_candidates = qualified_candidates
        available_after_direct = self._allocation_available_after(
            allocation, self.portfolio_state.available_balance
        )
        rebalance_orders, replacement_allocation = self._plan_rebalance(
            underfunded_candidates,
            bundles,
            bar_index,
            available_after_direct,
        )
        # Rebalance closes must be queued before both ordinary and replacement
        # opens so their actual released margin is visible at execution time.
        orders.extend(rebalance_orders)
        orders.extend(direct_orders)
        orders.extend(
            self._orders_from_entry_allocation(
                underfunded_candidates, replacement_allocation
            )
        )
        allocation = self._merge_allocations(
            allocation, replacement_allocation, bar_index
        )

        self._update_portfolio_state(bundles, ts, bar_index, allocation)
        return orders

    def _sync_account_context(self):
        equity = getattr(self, "_current_equity", None)
        if equity is None:
            return
        if self.portfolio is not None and hasattr(self.portfolio, "equity"):
            self.portfolio.equity = float(equity)
        self.portfolio_state.equity = float(equity or 0.0)
        self.portfolio_state.available_balance = max(0.0, float(getattr(self, "_current_available_balance", 0.0) or 0.0))

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
        if self.protection_supported:
            signal_para = _merge_para(signal_para, {
                "protection": {
                    "enabled": True,
                    "regression_method": self.estimator_cfg.regression_method,
                    "position_update_policy": self.estimator_cfg.position_update_policy,
                    "alpha": result.estimator.alpha,
                    "beta": result.estimator.beta,
                    "spread_mean": result.estimator.spread_mean,
                    "spread_std": result.estimator.spread_std,
                    "exit_z": signal.exit_z,
                    "signal_ts": signal.ts,
                    "signal_bar_index": signal.bar_index,
                }
            })
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
                "target_capital": pair_def.target_capital,
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

        # Compute replacement quality independently from the protection
        # feature switch. Rebalance must behave identically whether protective
        # exits are enabled or disabled.
        if self.frozen_model_supported and (
            self.protection_supported or self.rebalance_cfg.enabled
        ):
            try:
                estimated_entry_cost = (
                    abs(raw_target.x_notional) + abs(raw_target.y_notional)
                ) * (self.fee_rate + self.slippage_rate)
                boundary = solve_zero_net_x_price(
                    self.estimator_cfg.regression_method,
                    result.estimator.alpha,
                    result.estimator.beta,
                    result.estimator.spread_mean,
                    result.estimator.spread_std,
                    signal.exit_z,
                    signal.side,
                    bundle.x_bar.close,
                    bundle.y_bar.close,
                    raw_target.x_quantity,
                    raw_target.y_quantity,
                    estimated_entry_cost,
                    self.fee_rate,
                    self.slippage_rate,
                )
                x_move = None
                if boundary.x_price is not None and bundle.x_bar.close > 0:
                    x_move = abs(float(boundary.x_price) / float(bundle.x_bar.close) - 1.0)
                expected_net_pnl = None
                expected_net_return = None
                if boundary.target_residual is not None:
                    expected_net_pnl = net_pnl_at_x(
                        self.estimator_cfg.regression_method,
                        result.estimator.alpha,
                        result.estimator.beta,
                        boundary.target_residual,
                        bundle.x_bar.close,
                        bundle.x_bar.close,
                        bundle.y_bar.close,
                        signal.side,
                        raw_target.x_quantity,
                        raw_target.y_quantity,
                        estimated_entry_cost,
                        self.fee_rate,
                        self.slippage_rate,
                    )
                    gross = abs(raw_target.x_notional) + abs(raw_target.y_notional)
                    if expected_net_pnl is not None and gross > 1e-12:
                        expected_net_return = float(expected_net_pnl) / gross
                quality = {
                    "adf_pvalue": getattr(result.estimator, "cointegration_pvalue", None),
                    "theoretical_zero_return_x_move": x_move,
                    "theoretical_zero_return_x_price": boundary.x_price,
                    "theoretical_zero_return_x_direction": boundary.trigger_direction,
                    "theoretical_zero_return_reason": boundary.reason,
                    "expected_net_pnl": expected_net_pnl,
                    "expected_net_return": expected_net_return,
                    "signal_strength": float(getattr(signal, "signal_strength", 0.0) or 0.0),
                }
                signal_para = _merge_para(signal_para, {"rebalance_quality": quality})
                if self.protection_supported:
                    signal_para = _merge_para(signal_para, {"protection": {
                        "theoretical_zero_return_x_move": x_move,
                        "theoretical_zero_return_x_price": boundary.x_price,
                        "theoretical_zero_return_x_direction": boundary.trigger_direction,
                        "theoretical_zero_return_reason": boundary.reason,
                    }})
            except (TypeError, ValueError, OverflowError):
                pass
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
            "result": result,
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
        protection_state = pipeline.state.protection_state
        freeze_until = protection_state.freeze_until_bar
        if freeze_until is None:
            freeze_until = protection_state.pair_loss_stop_freeze_until_bar
        if freeze_until is not None and bar_index < freeze_until:
            rule = protection_state.freeze_rule or "pair loss stop"
            return {
                "allowed": False,
                "action": "none",
                "reason": f"{rule} freeze {freeze_until - bar_index} bars remaining",
            }
        if protection_state.reopen_lock_pending:
            return {
                "allowed": False,
                "action": "none",
                "reason": "waiting for model update after protective exit",
            }
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

    def _plan_rebalance(
        self,
        entry_candidates,
        bundles,
        bar_index,
        available_capital=None,
    ):
        """Plan one replacement target after ordinary allocation is complete."""
        if not self.rebalance_cfg.enabled or not entry_candidates:
            return [], None

        available = max(
            0.0,
            float(
                self.portfolio_state.available_balance
                if available_capital is None
                else available_capital
            ),
        )
        candidates = []
        for pair_id, candidate in entry_candidates.items():
            pipeline = self.pipelines.get(pair_id)
            if pipeline is None or self._pair_holds_position(pipeline):
                continue
            rebalance_candidate = self._rebalance_candidate_from_entry(
                pair_id, candidate
            )
            if rebalance_candidate is not None:
                candidates.append(rebalance_candidate)

        positions = []
        for pair_id, old_pipeline in self.pipelines.items():
            old_bundle = bundles.get(pair_id)
            if (
                pair_id in self._pending_rebalance_pair_ids
                or not self._pair_holds_position(old_pipeline)
                or old_pipeline.state.protection_state.pending_exit
                or old_bundle is None
                or old_bundle.x_bar is None
                or old_bundle.y_bar is None
            ):
                continue
            ledger = old_pipeline.state.position_ledger
            entry_bar = ledger.entry_bar_index
            held_bars = (
                max(0, int(old_bundle.bar_index) - int(entry_bar))
                if entry_bar is not None else 0
            )
            positions.append(RebalancePosition(
                pair_id=pair_id,
                net_return=self._pair_unrealized_return(old_pipeline, old_bundle),
                releasable_equity=self._pair_releasable_equity(old_pipeline, old_bundle),
                held_bars=held_bars,
                minimum_holding_bars=self._rebalance_min_holding_bars(old_pipeline),
                payload=old_pipeline,
            ))

        plan = self.rebalance_manager.plan(candidates, positions, available)
        if not plan.ready:
            return [], None

        rebalance_batch_id = str(uuid4())[:8]
        close_orders = []
        prepared_pipelines = []
        for position in plan.evictions:
            old_pipeline = position.payload
            old_state = old_pipeline.state.sizing_state
            pair_orders = self._close_orders(
                position.pair_id,
                old_pipeline,
                old_pipeline.pair_def,
                old_state.target_hedge_ratio
                or old_pipeline.state.estimator_state.hedge_beta
                or 1.0,
                exit_reason="rebalance_replacement",
                protection_trigger="rebalance_replacement",
                exit_class="rebalance_replacement",
                protection_freeze_bars=self._rebalance_freeze_bars(old_pipeline),
                rebalance_batch_id=rebalance_batch_id,
            )
            if not pair_orders:
                for prepared in prepared_pipelines:
                    prepared.state.protection_state.pending_exit = False
                return [], None
            close_orders.extend(pair_orders)
            prepared_pipelines.append(old_pipeline)

        for position in plan.evictions:
            self._pending_rebalance_pair_ids.add(position.pair_id)

        chosen = plan.candidate.payload
        chosen_raw = chosen["raw_target"]
        chosen_raw.para.setdefault("portfolio", {})[
            "rebalance_batch_id"
        ] = rebalance_batch_id
        actual_available = self.portfolio_state.available_balance
        self.portfolio_state.available_balance = available + plan.planned_release
        replacement_allocation = self._allocate_entry_targets(
            {plan.candidate.pair_id: chosen_raw}, bar_index
        )
        self.portfolio_state.available_balance = actual_available
        if (
            replacement_allocation is None
            or plan.candidate.pair_id not in replacement_allocation.pair_targets
        ):
            for prepared in prepared_pipelines:
                prepared.state.protection_state.pending_exit = False
            for position in plan.evictions:
                self._pending_rebalance_pair_ids.discard(position.pair_id)
            return [], None
        return close_orders, replacement_allocation

    def _rebalance_candidate_from_entry(self, pair_id, candidate):
        raw = candidate.get("raw_target")
        target = float(getattr(raw, "gross_notional", 0.0) or 0.0)
        if target <= 0:
            target = float(getattr(raw, "target_capital", 0.0) or 0.0)
        if target <= 0:
            return None
        para = getattr(raw, "para", {}) or {}
        quality = (
            para.get("rebalance_quality", {}) if isinstance(para, dict) else {}
        )
        estimator = getattr(candidate.get("result"), "estimator", None)
        return RebalanceCandidate(
            pair_id=pair_id,
            target_capital=target,
            minimum_capital=(
                target * float(self.rebalance_cfg.minimum_entry_capital_ratio)
            ),
            adf_pvalue=quality.get(
                "adf_pvalue", getattr(estimator, "cointegration_pvalue", None)
            ),
            theoretical_zero_return_x_move=quality.get(
                "theoretical_zero_return_x_move"
            ),
            expected_net_return=quality.get("expected_net_return"),
            signal_strength=float(
                quality.get(
                    "signal_strength",
                    getattr(raw, "signal_strength", 0.0) or 0.0,
                ) or 0.0
            ),
            payload=candidate,
        )

    def _pair_unrealized_return(self, pipeline, bundle):
        ledger = pipeline.state.position_ledger
        if bundle is None or ledger.entry_gross_notional <= 0:
            return 0.0
        pnl = self._pair_mark_net_pnl(pipeline, bundle)
        return float(pnl / max(ledger.entry_gross_notional, 1e-12))

    def _pair_releasable_equity(self, pipeline, bundle):
        pair_marked_equity = getattr(self, "_current_pair_marked_equity", {})
        pair_id = getattr(pipeline.pair_def, "pair_id", None)
        if pair_id in pair_marked_equity:
            try:
                return max(0.0, float(pair_marked_equity[pair_id]))
            except (TypeError, ValueError):
                pass

        # Compatibility fallback for direct strategy fixtures that do not
        # provide the exchange account snapshot. Production backtests use the
        # exchange-derived Pair marked equity above.
        ledger = pipeline.state.position_ledger
        if bundle is None or ledger.entry_gross_notional <= 0:
            return 0.0
        return max(
            0.0,
            float(ledger.entry_gross_notional) + self._pair_mark_net_pnl(pipeline, bundle),
        )

    def _pair_mark_net_pnl(self, pipeline, bundle):
        state = pipeline.state.sizing_state
        ledger = pipeline.state.position_ledger
        side = state.position_side or ledger.side
        x_entry = ledger.entry_x_price or bundle.x_bar.close
        y_entry = ledger.entry_y_price or bundle.y_bar.close
        return float(mark_net_pnl(
            side,
            float(bundle.x_bar.close),
            float(bundle.y_bar.close),
            x_entry,
            y_entry,
            state.x_quantity,
            state.y_quantity,
            ledger.entry_fee,
            ledger.funding_cost,
            self.fee_rate,
            self.slippage_rate,
        ))

    def _rebalance_min_holding_bars(self, pipeline) -> int:
        multiplier = float(
            self.rebalance_cfg.eviction_min_holding_model_lookback_multiplier
        )
        lookback = getattr(pipeline.estimator, "model_lookback_bars", None)
        if lookback is None:
            lookback = getattr(self.estimator_cfg, "model_lookback_bars", 0)
        lookback = float(lookback or 0.0)
        if multiplier > 0.0 and (not isfinite(lookback) or lookback <= 0.0):
            raise ValueError(
                "rebalance eviction holding multiplier requires a positive model lookback"
            )
        product = lookback * multiplier
        if not isfinite(product) or product < 0.0:
            raise ValueError("rebalance eviction minimum holding period is invalid")
        return max(0, int(ceil(product)))

    def _rebalance_freeze_bars(self, pipeline):
        multiplier = float(self.rebalance_cfg.closed_pair_freeze_model_lookback_multiplier)
        lookback = getattr(pipeline.estimator, "model_lookback_bars", None)
        if lookback is None:
            lookback = getattr(self.estimator_cfg, "model_lookback_bars", 0)
        lookback = float(lookback or 0.0)
        if multiplier > 0.0 and (not isfinite(lookback) or lookback <= 0.0):
            raise ValueError("rebalance freeze multiplier requires a positive model lookback")
        product = lookback * multiplier
        if not isfinite(product) or product < 0.0:
            raise ValueError("rebalance freeze period is invalid")
        return max(0, int(ceil(product)))

    def _allocate_entry_targets(
        self,
        entry_targets: dict[str, RawPairTarget],
        bar_index: int,
    ):
        if not entry_targets:
            return None
        if self.portfolio is not None:
            return self.portfolio.allocate(self.portfolio_state, entry_targets, bar_index)

        # Fallback path: preserve every candidate; Exchange enforces margin.
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

    @staticmethod
    def _allocation_available_after(allocation, available_before) -> float:
        available = max(0.0, float(available_before or 0.0))
        if allocation is None:
            return available
        report = getattr(allocation, "constraint_report", {}) or {}
        if "available_after_plan" in report:
            return max(0.0, float(report["available_after_plan"] or 0.0))
        allocated = sum(
            float(target.final_x_notional or 0.0)
            + float(target.final_y_notional or 0.0)
            for target in allocation.pair_targets.values()
            if target.selected
        )
        return max(0.0, available - allocated)

    @staticmethod
    def _merge_allocations(first, second, bar_index):
        if first is None:
            return second
        if second is None:
            return first
        pair_targets = dict(first.pair_targets)
        pair_targets.update(second.pair_targets)
        selected = list(dict.fromkeys(
            list(first.selected_pair_ids) + list(second.selected_pair_ids)
        ))
        return PortfolioAllocation(
            bar_index=bar_index,
            selected_pair_ids=selected,
            pair_targets=pair_targets,
            portfolio_gross_exposure=sum(
                target.final_x_notional + target.final_y_notional
                for target in pair_targets.values()
            ),
            portfolio_net_exposure=MultiPairStrategy._allocated_net_exposure(
                pair_targets
            ),
            constraint_report={
                "direct": getattr(first, "constraint_report", {}) or {},
                "replacement": getattr(second, "constraint_report", {}) or {},
            },
            reason=f"{first.reason}; {second.reason}",
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
            pair_orders = self.order_planner.orders_for_target(
                alloc_target,
                pair_def,
                x_exchange=pair_def.x_exchange,
                y_exchange=pair_def.y_exchange,
                hedge_ratio=target_hedge,
                action="open",
                position_id=candidate["position_id"],
            )
            orders.extend(pair_orders)
        return orders

    def _refresh_pair_capacity(self, pair_id, pipeline, pair_def, bundle):
        state = pipeline.state.sizing_state
        if bundle is None:
            return

        equity = self._current_equity_value(pipeline)
        pair_cap = self._pair_target_capital(pair_def, pipeline)
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

    def _pair_target_capital(self, pair_def, pipeline=None) -> float:
        value = getattr(pair_def, "target_capital", None)
        if value is not None:
            return max(0.0, float(value))
        state = pipeline.state.sizing_state if pipeline is not None else None
        current = float(getattr(state, "pair_cap_gross", 0.0) or 0.0) if state is not None else 0.0
        if current > 0:
            return current
        return max(0.0, float(getattr(self.sizing_cfg, "notional", 0.0) or 0.0))

    def _open_pair_count(self) -> int:
        """Return the number of currently occupied Pair positions for diagnostics."""
        return sum(
            1 for pipeline in self.pipelines.values() if self._pair_holds_position(pipeline)
        )

    def _open_pair_ids(self) -> list[str]:
        return [
            pair_id
            for pair_id, pipeline in self.pipelines.items()
            if self._pair_holds_position(pipeline)
        ]

    @staticmethod
    def _pair_holds_position(pipeline) -> bool:
        state = pipeline.state.sizing_state
        if state.position_side is not None:
            return True
        return abs(float(state.x_quantity)) > 1e-12 or abs(float(state.y_quantity)) > 1e-12

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

    def _close_orders(
        self,
        pair_id,
        pipeline,
        pair_def,
        fallback_hedge,
        exit_reason: str | None = None,
        protection_trigger: str | None = None,
        exit_class: str | None = None,
        protection_freeze_bars: int | None = None,
        rebalance_batch_id: str | None = None,
    ) -> list[Order]:
        sizing_state = pipeline.state.sizing_state
        if sizing_state.x_quantity <= 0 or sizing_state.y_quantity <= 0:
            return []

        exit_class = self._resolve_exit_class(exit_class, exit_reason, protection_trigger)
        reopen_lock_pending = self._should_wait_after_non_z_exit(exit_class)
        protection_state = pipeline.state.protection_state
        if exit_reason == "rebalance_replacement":
            protection_state.pending_exit = True
            protection_state.protection_rule = "rebalance_replacement"
            protection_state.freeze_rule = "rebalance_replacement"
            protection_state.freeze_bars = int(protection_freeze_bars or 0)

        close_hedge = sizing_state.target_hedge_ratio or fallback_hedge
        target = AllocatedPairTarget(
            pair_id=pair_id,
            selected=True,
            side=sizing_state.position_side,
            hedge_ratio=close_hedge,
            final_x_quantity=sizing_state.x_quantity,
            final_y_quantity=sizing_state.y_quantity,
            para={
                "protection": {
                    "exit_reason": exit_reason,
                    "protection_trigger": protection_trigger,
                    "exit_class": exit_class,
                }
            } if exit_reason else {},
        )
        orders = self.order_planner.orders_for_target(
            target,
            pair_def,
            x_exchange=pair_def.x_exchange,
            y_exchange=pair_def.y_exchange,
            hedge_ratio=close_hedge,
            action="close",
            exit_reason=exit_reason,
            protection_trigger=protection_trigger,
            exit_class=exit_class,
            reopen_lock_pending=reopen_lock_pending,
            protection_max_holding_bars=protection_state.max_holding_bars,
            protection_max_holding_deadline_bar=protection_state.max_holding_deadline_bar,
            protection_pair_loss_stop_return=protection_state.pair_loss_stop_return,
            protection_pair_loss_stop_freeze_bars=protection_state.pair_loss_stop_freeze_bars,
            protection_pair_loss_stop_freeze_until_bar=protection_state.pair_loss_stop_freeze_until_bar,
            protection_rule=protection_state.protection_rule,
            protection_freeze_bars=(protection_freeze_bars if protection_freeze_bars is not None else protection_state.freeze_bars),
            rebalance_batch_id=rebalance_batch_id,
            protection_freeze_until_bar=(
                (pipeline.state.last_bar_index + 1 + (protection_freeze_bars if protection_freeze_bars is not None else protection_state.freeze_bars))
                if (protection_freeze_bars if protection_freeze_bars is not None else protection_state.freeze_bars) > 0
                else protection_state.freeze_until_bar
            ),
        )
        for order in orders:
            order.position_id = sizing_state.position_id
        return orders

    def _protection_exit_decision(self, pipeline, bundle):
        if not self.protection_supported or bundle is None:
            return None
        state = pipeline.state.protection_state
        sizing_state = pipeline.state.sizing_state
        if not state.active or state.pending_exit or sizing_state.position_side is None:
            return None
        decision = self.protection_manager.evaluate(ProtectionContext(
            pipeline=pipeline, bundle=bundle, state=state,
            ledger=pipeline.state.position_ledger,
            sizing_state=sizing_state, config=self.protection_cfg,
            fee_rate=self.fee_rate, slippage_rate=self.slippage_rate,
        ))
        if decision is None:
            return None
        state.pending_exit = True
        state.last_trigger = decision.protection_trigger
        state.last_exit_reason = decision.diagnostic.get('net_return') and f"pair net return {decision.diagnostic['net_return']:.4%}" or None
        state.protection_rule = decision.rule_name
        state.freeze_rule = decision.rule_name
        state.freeze_bars = int(decision.freeze_bars)
        state.last_exit_class = decision.exit_class
        # Keep the old dict contract for callers and historical tests; the
        # complete decision is carried by the protection state/order metadata.
        return {"reason": decision.exit_reason, "trigger": decision.protection_trigger}

    def _pair_loss_stop_freeze_bars_for_pipeline(self, pipeline) -> int:
        configured = getattr(self.protection_cfg, "pair_loss_stop_freeze_model_lookback_multiplier", None)
        if configured is None:
            return max(0, int(self.protection_cfg.pair_loss_stop_freeze_bars))
        lookback = getattr(pipeline.estimator, "model_lookback_bars", None)
        try:
            lookback = float(lookback)
            product = lookback * float(configured)
        except (TypeError, ValueError):
            product = 0.0
        if not isfinite(product) or product <= 0.0:
            raise ValueError("protection.pair_loss_stop_freeze_model_lookback_multiplier requires a valid model lookback")
        return max(1, int(ceil(product)))

    def _forced_liquidation_freeze_bars_for_pipeline(self, pipeline) -> int:
        """Resolve forced-liquidation freeze from the shared stop-loss setting.

        Pair-equity-zero is the account-level last-resort stop-loss.  It uses
        the same model-relative multiplier as the ordinary Pair loss stop so
        one configuration controls both protective exits consistently.
        """
        return self._pair_loss_stop_freeze_bars_for_pipeline(pipeline)

    def _max_holding_bars_for_pipeline(self, pipeline) -> int | None:
        if not self.protection_cfg.max_holding_time_enabled:
            return None
        configured_lookback = getattr(self.estimator_cfg, "model_lookback_bars", None)
        try:
            configured_lookback = float(configured_lookback)
        except (TypeError, ValueError):
            configured_lookback = 0.0
        if not isfinite(configured_lookback) or configured_lookback <= 0.0:
            raise ValueError(
                "protection.max_holding_time_enabled requires a valid positive estimator model_lookback_bars"
            )
        override = getattr(pipeline.pair_def, "model_lookback_bars_override", None)
        if override is not None:
            try:
                override = float(override)
            except (TypeError, ValueError):
                override = 0.0
            if not isfinite(override) or override <= 0.0:
                raise ValueError(
                    "protection.max_holding_time_enabled requires a valid positive pair model_lookback_bars_override"
                )
        lookback = getattr(pipeline.estimator, "model_lookback_bars", None)
        try:
            lookback = float(lookback)
        except (TypeError, ValueError):
            lookback = 0.0
        if not isfinite(lookback) or lookback <= 0.0:
            raise ValueError(
                "protection.max_holding_time_enabled requires a valid positive estimator model_lookback_bars"
            )
        product = lookback * float(
            self.protection_cfg.max_holding_time_model_lookback_multiplier
        )
        if not isfinite(product):
            raise ValueError(
                "protection.max_holding_time_model_lookback_multiplier produces an invalid max holding period"
            )
        return max(1, int(ceil(product)))

    def _should_wait_after_non_z_exit(self, exit_class: str | None) -> bool:
        """Only a stop-loss (or max-holding) exit locks the pair.

        A take-profit exit deliberately reopens immediately: the frozen model
        is still the one that produced the entry, so waiting for the next refit
        only throws away re-entry opportunities.
        """
        if exit_class != "stop_loss":
            return False
        configured = self.protection_cfg.wait_for_model_update_after_non_z_exit
        # Missing key means a legacy config. Preserve the historical
        # unconditional protective-exit lock in that case.
        return True if configured is None else bool(configured)

    @staticmethod
    def _resolve_exit_class(exit_class, exit_reason, protection_trigger) -> str:
        """Classify a close: zscore_reversion / take_profit / stop_loss.

        Stop-loss and max-holding exits share the ``stop_loss`` class because
        both mean "the frozen model was wrong, wait for a refit"; a take-profit
        exit is its own class and never locks the pair.
        """
        if exit_class:
            return str(exit_class)
        trigger = str(protection_trigger or "")
        reason = str(exit_reason or "")
        if trigger == "take_profit" or reason == "protective_take_profit":
            return "take_profit"
        if trigger or reason:
            return "stop_loss"
        return "zscore_reversion"

    def _initialize_or_update_position_ledger(self, pipeline, group, pair_side):
        """Record every fill independently from protection configuration."""
        ledger = pipeline.state.position_ledger
        sizing_state = pipeline.state.sizing_state
        if not ledger.active:
            ledger.active = True
            ledger.side = pair_side
            ledger.position_id = sizing_state.position_id
            ledger.entry_bar_index = pipeline.state.last_bar_index + 1
            ledger.entry_ts = self._first_attr(group, "ts", None)
            ledger.entry_x_price = None
            ledger.entry_y_price = None
            ledger.entry_x_quantity = 0.0
            ledger.entry_y_quantity = 0.0
            ledger.entry_gross_notional = 0.0
            ledger.entry_fee = 0.0
            ledger.entry_slippage = 0.0
            ledger.funding_cost = 0.0

        old_x_qty = ledger.entry_x_quantity
        old_y_qty = ledger.entry_y_quantity
        for trade in group:
            symbol = getattr(trade, "symbol", "")
            quantity = abs(float(getattr(trade, "quantity", 0.0) or 0.0))
            price = float(getattr(trade, "price", 0.0) or 0.0)
            if symbol == pipeline.pair_def.x_symbol:
                total = old_x_qty + quantity
                ledger.entry_x_price = (
                    ((ledger.entry_x_price or 0.0) * old_x_qty + price * quantity) / total
                    if total else None
                )
                ledger.entry_x_quantity = total
                old_x_qty = total
            elif symbol == pipeline.pair_def.y_symbol:
                total = old_y_qty + quantity
                ledger.entry_y_price = (
                    ((ledger.entry_y_price or 0.0) * old_y_qty + price * quantity) / total
                    if total else None
                )
                ledger.entry_y_quantity = total
                old_y_qty = total
            ledger.entry_fee += float(getattr(trade, "fee", 0.0) or 0.0)
            ledger.entry_slippage += float(
                getattr(trade, "slippage", 0.0) or 0.0
            )
        ledger.entry_gross_notional = (
            ledger.entry_x_quantity * float(ledger.entry_x_price or 0.0)
            + ledger.entry_y_quantity * float(ledger.entry_y_price or 0.0)
        )

    def _apply_position_ledger_close_delta(self, pipeline, trades):
        ledger = pipeline.state.position_ledger
        if not ledger.active:
            return
        old_gross = float(ledger.entry_gross_notional or 0.0)
        for trade in trades:
            symbol = getattr(trade, "symbol", "")
            quantity = abs(float(getattr(trade, "quantity", 0.0) or 0.0))
            if symbol == pipeline.pair_def.x_symbol:
                ledger.entry_x_quantity = max(
                    0.0, ledger.entry_x_quantity - quantity
                )
            elif symbol == pipeline.pair_def.y_symbol:
                ledger.entry_y_quantity = max(
                    0.0, ledger.entry_y_quantity - quantity
                )
        ledger.entry_gross_notional = (
            ledger.entry_x_quantity * float(ledger.entry_x_price or 0.0)
            + ledger.entry_y_quantity * float(ledger.entry_y_price or 0.0)
        )
        remaining_ratio = (
            min(1.0, max(0.0, ledger.entry_gross_notional / old_gross))
            if old_gross > 1e-12 else 0.0
        )
        ledger.entry_fee *= remaining_ratio
        ledger.entry_slippage *= remaining_ratio
        ledger.funding_cost *= remaining_ratio

    @staticmethod
    def _clear_position_ledger(pipeline):
        ledger = pipeline.state.position_ledger
        ledger.active = False
        ledger.side = None
        ledger.position_id = None
        ledger.entry_bar_index = None
        ledger.entry_ts = None
        ledger.entry_x_price = None
        ledger.entry_y_price = None
        ledger.entry_x_quantity = 0.0
        ledger.entry_y_quantity = 0.0
        ledger.entry_gross_notional = 0.0
        ledger.entry_fee = 0.0
        ledger.entry_slippage = 0.0
        ledger.funding_cost = 0.0

    def _initialize_or_update_protection(self, pipeline, group, pair_side):
        if not self.protection_supported:
            return
        state = pipeline.state.protection_state
        sizing_state = pipeline.state.sizing_state
        para = self._first_attr(group, "para", {}) or {}
        protection = dict(para.get("protection") or {}) if isinstance(para, dict) else {}
        if not state.active:
            state.entry_x_quantity = 0.0
            state.entry_y_quantity = 0.0
            state.entry_x_price = None
            state.entry_y_price = None
            state.entry_gross_notional = 0.0
            state.entry_fee = 0.0
            state.entry_slippage = 0.0
            state.funding_cost = 0.0
            state.active = True
            state.pending_exit = False
            state.last_trigger = None
            state.pair_loss_stop_freeze_until_bar = None
            state.freeze_rule = None
            state.freeze_bars = 0
            state.freeze_until_bar = None
            state.protection_rule = None
            state.side = pair_side
            state.position_id = sizing_state.position_id
            # Orders created on the current signal bar are filled by the
            # exchange before the next strategy bar.  Therefore the first
            # actual fill is the next pair bar, not the signal bar.
            state.entry_bar_index = pipeline.state.last_bar_index + 1
            state.entry_ts = self._first_attr(group, "ts", None)
            state.alpha = protection.get("alpha")
            state.beta = protection.get("beta")
            state.spread_mean = protection.get("spread_mean")
            state.spread_std = protection.get("spread_std")
            state.exit_z = protection.get("exit_z")
            state.take_profit_return = float(self.protection_cfg.take_profit_return)
            state.pair_loss_stop_return = float(self.protection_cfg.pair_loss_stop_return)
            state.pair_loss_stop_freeze_bars = self._pair_loss_stop_freeze_bars_for_pipeline(pipeline)
            state.max_holding_bars = self._max_holding_bars_for_pipeline(pipeline)
            state.max_holding_deadline_bar = (
                state.entry_bar_index + state.max_holding_bars
                if state.entry_bar_index is not None and state.max_holding_bars is not None
                else None
            )

        old_x_qty = state.entry_x_quantity
        old_y_qty = state.entry_y_quantity
        for trade in group:
            symbol = getattr(trade, "symbol", "")
            qty = abs(float(getattr(trade, "quantity", 0.0) or 0.0))
            price = float(getattr(trade, "price", 0.0) or 0.0)
            if symbol == pipeline.pair_def.x_symbol:
                total = old_x_qty + qty
                state.entry_x_price = ((state.entry_x_price or 0.0) * old_x_qty + price * qty) / total if total else None
                state.entry_x_quantity = total
                old_x_qty = total
            elif symbol == pipeline.pair_def.y_symbol:
                total = old_y_qty + qty
                state.entry_y_price = ((state.entry_y_price or 0.0) * old_y_qty + price * qty) / total if total else None
                state.entry_y_quantity = total
                old_y_qty = total
            state.entry_fee += float(getattr(trade, "fee", 0.0) or 0.0)
            state.entry_slippage += float(getattr(trade, "slippage", 0.0) or 0.0)
        state.entry_gross_notional = (
            state.entry_x_quantity * float(state.entry_x_price or 0.0)
            + state.entry_y_quantity * float(state.entry_y_price or 0.0)
        )
        boundary = solve_zero_net_x_price(
            self.estimator_cfg.regression_method,
            state.alpha,
            state.beta,
            state.spread_mean,
            state.spread_std,
            state.exit_z,
            state.side,
            state.entry_x_price,
            state.entry_y_price,
            state.entry_x_quantity,
            state.entry_y_quantity,
            # The stored entry prices include execution slippage. Subtract
            # only the separately charged entry fee here; otherwise the
            # theoretical stop would double-count entry slippage.
            state.entry_fee,
            self.fee_rate,
            self.slippage_rate,
        )
        state.stop_loss_x_price = boundary.x_price
        state.stop_loss_direction = boundary.trigger_direction
        state.stop_loss_net_pnl = 0.0 if boundary.x_price is not None else None
        state.stop_loss_reason = boundary.reason
        for trade in group:
            trade.protection_stop_x_price = state.stop_loss_x_price
            trade.protection_take_profit_return = state.take_profit_return
            trade.protection_target_residual = boundary.target_residual
            trade.protection_stop_reason = boundary.reason
            trade.protection_max_holding_bars = state.max_holding_bars
            trade.protection_max_holding_deadline_bar = state.max_holding_deadline_bar
            trade.protection_pair_loss_stop_return = state.pair_loss_stop_return
            trade.protection_pair_loss_stop_freeze_bars = state.pair_loss_stop_freeze_bars

    def _maybe_release_protection_lock(self, pipeline):
        state = pipeline.state.protection_state
        if not state.reopen_lock_pending or state.close_bar_index is None:
            return
        estimator_state = pipeline.state.estimator_state
        if (
            estimator_state.last_model_update_index is not None
            and estimator_state.last_model_update_index > state.close_bar_index
        ):
            state.reopen_lock_pending = False

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
        # Pair target capital is the complete X+Y gross notional. Preserve the
        # beta-neutral weights produced by sizing while replacing only the
        # gross amount. Add-ons consume the remaining Pair target.
        target_capital = getattr(pair_def, "target_capital", None)
        if target_capital is not None:
            total = max(float(target.x_notional + target.y_notional), 1e-12)
            x_weight = float(target.x_notional) / total
            y_weight = float(target.y_notional) / total
            state = pipeline.state.sizing_state
            desired = max(0.0, float(target_capital))
            if state.position_side is not None:
                current = float(state.current_gross_notional or 0.0)
                desired = max(0.0, desired - current)
            target.x_notional = desired * x_weight
            target.y_notional = desired * y_weight
            target.x_quantity = target.x_notional / x_price if x_price > 0 else 0.0
            target.y_quantity = target.y_notional / y_price if y_price > 0 else 0.0
            target.gross_notional = desired
            target.target_capital = float(target_capital)
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
                self._pending_open_pair_ids.discard(pair_id)
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
                self._initialize_or_update_position_ledger(
                    pipeline, group, pair_side
                )
                self._initialize_or_update_protection(pipeline, group, pair_side)

            elif action == "close":
                exit_reason = self._first_attr(group, "exit_reason", None)
                protection_trigger = self._first_attr(group, "protection_trigger", None)
                exit_class = self._first_attr(group, "exit_class", None)
                exit_class = self._resolve_exit_class(exit_class, exit_reason, protection_trigger)
                for trade in group:
                    self._apply_pair_quantity_delta(pipeline, sizing_state, trade, sign=-1.0)
                self._apply_position_ledger_close_delta(pipeline, group)
                self._apply_protection_close_delta(pipeline, group)
                if sizing_state.x_quantity >= 1e-12 or sizing_state.y_quantity >= 1e-12:
                    fill_prices = {
                        getattr(trade, "symbol", ""): float(getattr(trade, "price", 0.0) or 0.0)
                        for trade in group
                    }
                    x_price = fill_prices.get(pipeline.pair_def.x_symbol)
                    y_price = fill_prices.get(pipeline.pair_def.y_symbol)
                    if x_price is not None and y_price is not None:
                        sizing_state.current_gross_notional = (
                            sizing_state.x_quantity * x_price
                            + sizing_state.y_quantity * y_price
                        )
                        sizing_state.remaining_pair_gross = max(
                            0.0,
                            float(sizing_state.pair_cap_gross or 0.0)
                            - sizing_state.current_gross_notional,
                        )
                        sizing_state.pair_full = sizing_state.remaining_pair_gross <= 1e-8
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
                    self._pending_rebalance_pair_ids.discard(pair_id)
                    self._clear_position_ledger(pipeline)
                    protection_state = pipeline.state.protection_state
                    was_protective = exit_class == "stop_loss"
                    protection_state.active = False
                    protection_state.pending_exit = False
                    protection_state.last_exit_reason = exit_reason
                    protection_state.last_trigger = protection_trigger
                    protection_state.last_exit_class = exit_class
                    # As with entry, the close callback runs before the next
                    # strategy bar is processed, so record the actual fill
                    # bar rather than the prior signal bar.
                    protection_state.close_bar_index = pipeline.state.last_bar_index + 1
                    protection_state.protection_rule = self._first_attr(group, "protection_rule", protection_state.last_trigger)
                    protection_state.freeze_rule = protection_state.protection_rule
                    protection_state.freeze_bars = int(self._first_attr(group, "protection_freeze_bars", 0) or 0)
                    if protection_state.protection_rule == "pair_equity_zero" and protection_state.freeze_bars <= 0:
                        protection_state.freeze_bars = self._forced_liquidation_freeze_bars_for_pipeline(pipeline)
                        for trade in group:
                            trade.protection_freeze_bars = protection_state.freeze_bars
                            trade.protection_freeze_until_bar = (
                                protection_state.close_bar_index + protection_state.freeze_bars
                            )
                    protection_state.freeze_until_bar = (
                        protection_state.close_bar_index + protection_state.freeze_bars
                        if protection_state.freeze_bars > 0 else None
                    )
                    if protection_state.last_trigger == "pair_loss_stop":
                        protection_state.pair_loss_stop_freeze_until_bar = (
                            protection_state.close_bar_index + protection_state.pair_loss_stop_freeze_bars
                        )
                    explicit_lock = self._first_attr(group, "reopen_lock_pending", None)
                    protection_state.reopen_lock_pending = (
                        bool(explicit_lock)
                        if explicit_lock is not None
                        else (was_protective and self._should_wait_after_non_z_exit(exit_class))
                    )
                    protection_state.position_id = None
                    protection_state.entry_x_quantity = 0.0
                    protection_state.entry_y_quantity = 0.0
                    protection_state.entry_x_price = None
                    protection_state.entry_y_price = None
                    protection_state.stop_loss_direction = None
                    protection_state.entry_gross_notional = 0.0
                    protection_state.entry_fee = 0.0
                    protection_state.entry_slippage = 0.0
                    protection_state.funding_cost = 0.0

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

    def _apply_protection_close_delta(self, pipeline, trades):
        """Keep protection accounting aligned after a partial Pair close."""
        state = pipeline.state.protection_state
        if not state.active:
            return

        old_gross = float(state.entry_gross_notional or 0.0)
        for trade in trades:
            symbol = getattr(trade, "symbol", "")
            quantity = abs(float(getattr(trade, "quantity", 0.0) or 0.0))
            if symbol == pipeline.pair_def.x_symbol:
                state.entry_x_quantity = max(0.0, state.entry_x_quantity - quantity)
            elif symbol == pipeline.pair_def.y_symbol:
                state.entry_y_quantity = max(0.0, state.entry_y_quantity - quantity)

        state.entry_gross_notional = (
            state.entry_x_quantity * float(state.entry_x_price or 0.0)
            + state.entry_y_quantity * float(state.entry_y_price or 0.0)
        )
        remaining_ratio = (
            min(1.0, max(0.0, state.entry_gross_notional / old_gross))
            if old_gross > 1e-12 else 0.0
        )
        state.entry_fee *= remaining_ratio
        state.entry_slippage *= remaining_ratio
        state.funding_cost *= remaining_ratio

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
            action = getattr(order, "action", "")
            action = getattr(action, "value", action)
            if str(action) == "open":
                self._pending_open_pair_ids.discard(pair_id)
            if str(action) == "close":
                trigger = getattr(order, "protection_trigger", None)
                if trigger == "rebalance_replacement":
                    self._pending_rebalance_pair_ids.discard(pair_id)
                    pipeline.state.protection_state.pending_exit = False
                elif trigger:
                    pipeline.state.protection_state.pending_exit = False

    def on_orders_accepted(self, orders: list[Order]):
        for order in orders or []:
            action = getattr(order, "action", "")
            action = getattr(action, "value", action)
            pair_id = getattr(order, "pair_id", None)
            if str(action) != "open" or not pair_id:
                continue
            pipeline = self.pipelines.get(pair_id)
            if pipeline is not None and not self._pair_holds_position(pipeline):
                self._pending_open_pair_ids.add(pair_id)

    def on_funding_payments(self, payments: list):
        """Allocate exchange funding payments to active Pair positions.

        Funding records are symbol-level.  When several Pair positions share a
        symbol, allocate each payment by that Pair's absolute quantity share;
        this keeps take-profit mark returns from counting the same funding
        payment once per Pair.
        """
        if not payments:
            return
        for payment in payments:
            symbol = str(getattr(payment, "symbol", "") or "")
            amount = float(getattr(payment, "payment", 0.0) or 0.0)
            if not symbol or amount == 0.0:
                continue
            pair_id = getattr(payment, "pair_id", None)
            position_id = getattr(payment, "position_id", None)
            if pair_id in self.pipelines:
                runtime = self.pipelines[pair_id].state
                ledger = runtime.position_ledger
                pstate = runtime.protection_state
                applied = False
                if ledger.active and (
                    position_id is None or ledger.position_id == position_id
                ):
                    ledger.funding_cost += amount
                    applied = True
                if pstate.active and (
                    position_id is None or pstate.position_id == position_id
                ):
                    pstate.funding_cost += amount
                    applied = True
                if applied:
                    continue
            active = []
            total = 0.0
            for pipeline in self.pipelines.values():
                ledger = pipeline.state.position_ledger
                pstate = pipeline.state.protection_state
                accounting = ledger if ledger.active else pstate
                if not accounting.active:
                    continue
                if symbol == pipeline.pair_def.x_symbol:
                    qty = accounting.entry_x_quantity
                elif symbol == pipeline.pair_def.y_symbol:
                    qty = accounting.entry_y_quantity
                else:
                    continue
                qty = abs(float(qty))
                if qty > 0:
                    active.append((pipeline, qty))
                    total += qty
            if total <= 0:
                continue
            for pipeline, qty in active:
                share = amount * qty / total
                ledger = pipeline.state.position_ledger
                pstate = pipeline.state.protection_state
                if ledger.active:
                    ledger.funding_cost += share
                if pstate.active:
                    pstate.funding_cost += share

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
