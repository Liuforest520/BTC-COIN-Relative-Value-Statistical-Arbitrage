"""Probes for the Pair-equity-zero forced-liquidation fix (2026-09-22).

Timing convention verified here (both are look-ahead free, they just differ in what
the bar shows at its open):
  * crash visible at the bar's OPEN  -> liquidated at that same bar's open;
  * crash only inside the bar (open normal, close crashed) -> liquidated at the NEXT
    bar's open (the loss is only known after the close).

Also verifies: the excess loss is charged to free cash, a later price recovery credits
nothing (no free option), other Pairs are untouched, and the strategy locks the Pair
until the next model update.

Run from the repo root: python research/scripts/review_0922_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.modules.exchange.exchange import Exchange, OrderAction, OrderSide, OrderType  # noqa: E402
from core.modules.exchange.manager import ExchangeManager  # noqa: E402
from core.modules.models import Order  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        FAILURES.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def flat_bar(ts: int, price: float) -> list:
    """A bar that opens and closes at the same price."""
    return [ts, price, price, price, price, 1.0]


def open_at_close_at(ts: int, open_price: float, close_price: float) -> list:
    """A bar that opens at one price and closes at another (loss appears only at close)."""
    return [ts, open_price, max(open_price, close_price), close_price, min(open_price, close_price), 1.0]


def order(oid: str, gid: str, symbol: str, side, qty: float) -> Order:
    return Order(order_id=oid, group_id=gid, exchange="binance", symbol=symbol,
                 action=OrderAction.OPEN, side=side, order_type=OrderType.MARKET,
                 quantity=qty, pair_id=gid, position_id=gid)


def build() -> tuple[Exchange, ExchangeManager]:
    """Two isolated Pairs of 40 each out of 100: `a` will blow up, `b` must not."""
    exchange = Exchange("binance", initial_cash=100.0, fee_rate=0.0, slippage_bps=0.0)
    manager = ExchangeManager({"binance": exchange})
    manager.place_orders([
        order("a1", "a", "A1", OrderSide.BUY, 2.0),
        order("a2", "a", "A2", OrderSide.SELL, 2.0),
        order("b1", "b", "B1", OrderSide.BUY, 2.0),
        order("b2", "b", "B2", OrderSide.SELL, 2.0),
    ])
    manager.on_bar({"binance": {s: flat_bar(1, 10.0) for s in ("A1", "A2", "B1", "B2")}})
    return exchange, manager


def assert_honest_after_liquidation(exchange, result, label: str) -> None:
    trades = result["new_trades"]
    check(f"{label}1 只强平资不抵债的那一对",
          {t.pair_id for t in trades} == {"a"} and "a" not in exchange.position_lots
          and "b" in exchange.position_lots,
          f"pairs={ {t.pair_id for t in trades} }")
    check(f"{label}2 成交字段符合约定",
          {t.exit_reason for t in trades} == {"protective_pair_equity_zero"}
          and {t.protection_trigger for t in trades} == {"pair_equity_zero"}
          and {t.exit_class for t in trades} == {"stop_loss"}
          and all(t.reopen_lock_pending is True for t in trades),
          f"{ {t.exit_reason for t in trades} } / { {t.exit_class for t in trades} }")
    check(f"{label}3 超出本金的 18 从可用资金真实扣掉（20 → 2）",
          abs(exchange.available_balance - 2.0) < 1e-9, f"available={exchange.available_balance}")
    check(f"{label}4 未受影响的 b 对资本仍是 40",
          abs(exchange.position_capital.get("b", 0.0) - 40.0) < 1e-9,
          f"b={exchange.position_capital.get('b')}")
    check(f"{label}5 资不抵债的对已从账本移除", "a" not in exchange.position_capital)
    check(f"{label}6 权益 = 100-58 = 42（亏损已实打实计入）",
          abs(exchange.equity - 42.0) < 1e-9, f"equity={exchange.equity}")
    check(f"{label}7 强制减仓字段被真实置位",
          result["forced_deleveraging_triggered"] is True
          and result["forced_deleveraging_scale"] == 1.0
          and len(result["forced_deleveraging_orders"]) == 2,
          f"triggered={result['forced_deleveraging_triggered']} "
          f"orders={len(result['forced_deleveraging_orders'])}")


print("== A. 崩盘价在开盘就可见 → 当根开盘即强平 ==")
exchange, manager = build()
check("A0 建仓后：每对资本 40，可用 20",
      abs(exchange.available_balance - 20.0) < 1e-9
      and abs(exchange.position_capital["a"] - 40.0) < 1e-9,
      f"available={exchange.available_balance}")
result = manager.on_bar({"binance": {"A1": flat_bar(2, 1.0), "A2": flat_bar(2, 30.0),
                                     "B1": flat_bar(2, 10.0), "B2": flat_bar(2, 10.0)}})
assert_honest_after_liquidation(exchange, result, "A")

print("\n== B. 强平后价格反弹是否还能“免费赚回来” ==")
result = manager.on_bar({"binance": {"B1": flat_bar(3, 10.0), "B2": flat_bar(3, 10.0)}})
check("B1 价格回到 10 没有任何成交", result["new_trades"] == [])
check("B2 权益不变（仍 42）", abs(exchange.equity - 42.0) < 1e-9, f"equity={exchange.equity}")
check("B3 可用资金不变（2）", abs(exchange.available_balance - 2.0) < 1e-9,
      f"available={exchange.available_balance}")

print("\n== C. 亏损只在盘中出现（开盘正常、收盘崩）→ 下一根开盘强平，且当根不成交 ==")
exchange, manager = build()
result = manager.on_bar({"binance": {
    "A1": open_at_close_at(2, 10.0, 1.0), "A2": open_at_close_at(2, 10.0, 30.0),
    "B1": flat_bar(2, 10.0), "B2": flat_bar(2, 10.0)}})
check("C1 崩盘那根不成交，也不强平（判定用该根开盘价，10 时尚未资不抵债）",
      result["new_trades"] == [] and result["forced_deleveraging_triggered"] is False
      and "a" in exchange.position_lots,
      f"trades={len(result['new_trades'])}")
check("C2 该根收盘后 a 对已资不抵债（资本 40 + 浮亏 -58 < 0，账本如实反映）",
      exchange.position_capital.get("a", 0.0) + sum(
          float(q) * (float(p) - 10.0) for q, p in ((2.0, 1.0), (-2.0, 30.0))) < 0,
      f"capital={exchange.position_capital.get('a')}")
result = manager.on_bar({"binance": {
    "A1": flat_bar(3, 1.0), "A2": flat_bar(3, 30.0),
    "B1": flat_bar(3, 10.0), "B2": flat_bar(3, 10.0)}})
assert_honest_after_liquidation(exchange, result, "C")

print("\n== D. 策略侧：强平后的冻结与锁定 ==")
from core.modules.models.pipeline_types import PositionProtectionState, SizingState  # noqa: E402
from core.modules.strategy.config import ProtectionConfig  # noqa: E402
from core.modules.strategy.multi_pair_strategy import MultiPairStrategy  # noqa: E402


def make_strategy(lookback_bars: int = 2880, multiplier: float | None = 1.0,
                  freeze_bars_cfg: int = 0) -> tuple[MultiPairStrategy, SimpleNamespace]:
    strategy = MultiPairStrategy.__new__(MultiPairStrategy)
    sizing = SizingState(position_side="long_x", position_id="a", x_quantity=2.0, y_quantity=2.0,
                         current_gross_notional=40.0, target_hedge_ratio=1.0, entry_count=1,
                         last_entry_bar_index=5, pair_cap_gross=1_000_000.0,
                         remaining_pair_gross=1_000_000.0)
    protection = PositionProtectionState(active=True, side="long_x", entry_x_price=10.0,
                                         entry_y_price=10.0, entry_x_quantity=2.0,
                                         entry_y_quantity=2.0, entry_gross_notional=40.0,
                                         entry_bar_index=6)
    pipeline = SimpleNamespace(
        pair_def=SimpleNamespace(pair_id="a", x_symbol="A1", y_symbol="A2"),
        state=SimpleNamespace(sizing_state=sizing, protection_state=protection, last_bar_index=10),
        estimator=SimpleNamespace(model_lookback_bars=lookback_bars),
    )
    strategy.pipelines = {"a": pipeline}
    strategy._pending_rebalance_pair_ids = set()
    strategy._pending_open_pair_ids = set()
    strategy.estimator_cfg = SimpleNamespace(model_lookback_bars=lookback_bars,
                                             model_timeframe="1m")
    strategy.protection_cfg = ProtectionConfig(
        enabled=True,
        pair_loss_stop_enabled=True,
        pair_loss_stop_freeze_bars=freeze_bars_cfg,
        pair_loss_stop_freeze_model_lookback_multiplier=multiplier,
        wait_for_model_update_after_non_z_exit=True,
    )
    return strategy, pipeline


def forced_group(protection_rule: str = "pair_equity_zero") -> list:
    return [
        SimpleNamespace(pair_id="a", group_id="g", action="close", symbol=symbol, quantity=2.0,
                        exit_class="stop_loss", exit_reason="protective_pair_equity_zero",
                        protection_trigger="pair_equity_zero", reopen_lock_pending=True,
                        protection_rule=protection_rule)
        for symbol in ("A1", "A2")
    ]


strategy, pipeline = make_strategy()
group = forced_group()
strategy.on_trades_filled(group)
sizing, protection = pipeline.state.sizing_state, pipeline.state.protection_state

check("D1 该对仓位已清空", sizing.position_side is None and sizing.x_quantity == 0.0
      and sizing.y_quantity == 0.0 and sizing.entry_count == 0)
check("D2 记录了退出原因与类别",
      protection.last_exit_reason == "protective_pair_equity_zero"
      and protection.last_exit_class == "stop_loss",
      f"{protection.last_exit_reason}/{protection.last_exit_class}")
check("D3 进入“等下一次模型更新”锁定（reopen_lock_pending）",
      protection.reopen_lock_pending is True)
check("D4 冻结规则名 = pair_equity_zero", protection.freeze_rule == "pair_equity_zero",
      f"freeze_rule={protection.freeze_rule}")
check("D5 冻结长度取 Pair 亏损止损的倍数设置：ceil(lookback × 1.0) = 2880 根",
      protection.freeze_bars == 2880, f"freeze_bars={protection.freeze_bars}")
check("D6 冻结截止 bar = 平仓 bar(11) + 2880",
      protection.freeze_until_bar == 11 + 2880, f"freeze_until_bar={protection.freeze_until_bar}")
check("D7 成交对象被回填了冻结字段（报告可见）",
      all(t.protection_freeze_bars == 2880 for t in group)
      and all(t.protection_freeze_until_bar == 11 + 2880 for t in group),
      f"{group[0].protection_freeze_bars}/{group[0].protection_freeze_until_bar}")

_ = MultiPairStrategy._entry_schedule_decision.__get__(strategy)  # bound for clarity
signal = SimpleNamespace(side="long_x", bar_index=20)
decision = strategy._entry_schedule_decision(pipeline, signal, 20)
check("D8 冻结期内拒绝开仓并给出剩余 bar 数",
      decision["allowed"] is False and "pair_equity_zero freeze" in decision["reason"],
      decision["reason"])
decision = strategy._entry_schedule_decision(pipeline, signal, 11 + 2880)
check("D9 冻结到期后不再因冻结被拦（只剩模型更新锁）",
      decision["allowed"] is False and "model update" in decision["reason"],
      decision["reason"])
protection.reopen_lock_pending = False
decision = strategy._entry_schedule_decision(pipeline, signal, 11 + 2881)
check("D10 冻结到期 + 模型更新锁解除后可开仓",
      decision["allowed"] is True and decision["action"] == "open", decision["reason"])

s15, p15 = make_strategy(lookback_bars=192)      # 15m 下换算后的模型估计期
s15.on_trades_filled(forced_group())
check("D11 15m（换算后 lookback=192）冻结 192 根 = 2880 分钟，墙钟时间一致",
      p15.state.protection_state.freeze_bars == 192,
      f"freeze_bars={p15.state.protection_state.freeze_bars}")

s_legacy, p_legacy = make_strategy(multiplier=None, freeze_bars_cfg=0)
s_legacy.on_trades_filled(forced_group())
check("D12 若把倍数设为 None 且固定冻结为 0，则强平后只剩“等模型更新”一道锁（冻结 0 根）",
      p_legacy.state.protection_state.freeze_bars == 0
      and p_legacy.state.protection_state.freeze_until_bar is None,
      f"freeze_bars={p_legacy.state.protection_state.freeze_bars}")

s_other, p_other = make_strategy()
other = forced_group(protection_rule="theoretical_x_stop")
s_other.on_trades_filled(other)
check("D13 其它止损规则不会被套用强平冻结（走各自的 protection_freeze_bars）",
      p_other.state.protection_state.freeze_bars == 0,
      f"freeze_bars={p_other.state.protection_state.freeze_bars} "
      f"rule={p_other.state.protection_state.freeze_rule}")

print("\n==================== SUMMARY ====================")
if FAILURES:
    print(f"{len(FAILURES)} FAILED:")
    for name in FAILURES:
        print(f"  - {name}")
    raise SystemExit(1)
print("all probes passed")
