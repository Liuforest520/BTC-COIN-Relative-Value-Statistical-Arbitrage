"""Ad-hoc probes for the 2026-09-18 pair-target-capital / rebalance changes.

Read-only: builds in-memory fakes only, never runs a backtest or writes data.
Run from the repo root:

    python research/scripts/review_0918_probe.py
"""
from __future__ import annotations

import sys
from math import ceil
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.modules.exchange import ExchangeManager
from core.modules.exchange.exchange import Exchange
from core.modules.models import Order, OrderAction, OrderSide, OrderType
from core.modules.models.pipeline_types import (
    PositionProtectionState,
    PortfolioState,
    RawPairTarget,
    SizingState,
)
from core.modules.portfolio.allocator import PairTargetCapitalAllocator
from core.modules.strategy.config import (
    ProfitablePositionReplacementConfig,
    RebalanceConfig,
)
from core.modules.strategy.multi_pair_strategy import MultiPairStrategy

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    if not ok:
        FAILURES.append(name)
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------- allocator
def probe_allocator() -> None:
    print("\n== A. PairTargetCapitalAllocator ==")
    alloc = PairTargetCapitalAllocator(equity=3_000_000, minimum_entry_capital_ratio=0.5)

    def raw(pid: str, gross: float = 600_000.0, xw: float = 0.6, yw: float = 0.4) -> RawPairTarget:
        return RawPairTarget(
            pair_id=pid, ready=True, side="long_x", x_weight=xw, y_weight=yw,
            x_price=100.0, y_price=50.0, gross_notional=gross, target_capital=gross,
        )

    def run(pid_gross: dict[str, RawPairTarget], available: float, ready: bool = True,
            equity: float = 3_000_000.0):
        state = PortfolioState(ready=ready, equity=equity, available_balance=available)
        return alloc.allocate(state, pid_gross, 1)

    t = run({"p": raw("p")}, 350_000.0).pair_targets["p"]
    check("A1 target 600k with 350k available allocates the whole 350k",
          abs(t.final_x_notional - 210_000.0) < 1e-6 and abs(t.final_y_notional - 140_000.0) < 1e-6,
          f"x={t.final_x_notional:.0f} y={t.final_y_notional:.0f}")

    check("A2 target 600k with 250k available is skipped",
          run({"p": raw("p")}, 250_000.0).pair_targets == {})

    a = run({"p": raw("p")}, 300_000.0)
    check("A3 exactly 300k (= 0.5 * whole-pair 600k) is filled",
          abs(a.pair_targets["p"].final_x_notional - 180_000.0) < 1e-6)

    check("A4 one cent below 300k is skipped",
          run({"p": raw("p")}, 299_999.99).pair_targets == {})

    check("A5 available_balance == 0 with 3M equity does NOT fall back to equity",
          run({"p": raw("p")}, 0.0).pair_targets == {})

    check("A6 ready=False + available 0 + equity>0 still does NOT fall back",
          run({"p": raw("p")}, 0.0, ready=False).pair_targets == {})

    check("A7 documented fixture fallback needs available==0 AND equity==0 AND ready==False",
          run({"p": raw("p")}, 0.0, ready=False, equity=0.0).pair_targets != {})

    a = run({"p1": raw("p1"), "p2": raw("p2")}, 700_000.0)
    check("A8 700k funds only the first 600k pair; the second is skipped in the same bar",
          list(a.pair_targets) == ["p1"], f"selected={list(a.pair_targets)}")

    r = raw("p3", gross=120_000.0)
    r.target_capital = 600_000.0
    a = run({"p3": r}, 3_000_000.0)
    check("A9 an add-on allocates the remaining gross_notional (120k), not target_capital again",
          abs(a.pair_targets["p3"].final_x_notional - 72_000.0) < 1e-6,
          f"x={a.pair_targets['p3'].final_x_notional:.0f}")


# ------------------------------------------------------- cross-exchange fill
def probe_cross_exchange_scale() -> None:
    print("\n== B. Cross-exchange common scale / margin ==")

    def bar(ts: int, price: float):
        return [ts, price, price, price, price, 1.0]

    def order(oid: str, gid: str, exch: str, symbol: str, side, qty: float) -> Order:
        return Order(order_id=oid, group_id=gid, exchange=exch, symbol=symbol,
                     action=OrderAction.OPEN, side=side, order_type=OrderType.MARKET,
                     quantity=qty, pair_id=gid, position_id=gid)

    ex_a = Exchange("binance", initial_cash=100.0, fee_rate=0.0, slippage_bps=0.0)
    ex_b = Exchange("okx", initial_cash=100.0, fee_rate=0.0, slippage_bps=0.0)
    manager = ExchangeManager({"binance": ex_a, "okx": ex_b})
    manager.place_orders([
        order("x", "p", "binance", "X", OrderSide.BUY, 8.0),
        order("y", "p", "okx", "Y", OrderSide.SELL, 12.0),
    ])
    result = manager.on_bar({"binance": {"X": bar(1, 10.0)}, "okx": {"Y": bar(1, 10.0)}})
    fills = {t.symbol: t for t in result["new_trades"]}

    check("B1 both legs receive the same scale (binding leg is the poorer exchange)",
          abs(fills["X"].fill_scale - fills["Y"].fill_scale) < 1e-12
          and abs(fills["Y"].quantity - 10.0) < 1e-9
          and abs(fills["X"].quantity - 8.0 * 100.0 / 120.0) < 1e-9,
          f"scale={fills['X'].fill_scale:.6f} x_qty={fills['X'].quantity:.6f} y_qty={fills['Y'].quantity:.6f}")

    check("B2 the short leg consumes margin and never credits wallet cash",
          abs(ex_b.wallet_balance - 100.0) < 1e-9
          and abs(ex_b.available_balance) < 1e-9
          and abs(ex_a.available_balance - (100.0 - 8.0 * 100.0 / 120.0 * 10.0)) < 1e-9,
          f"a_avail={ex_a.available_balance:.6f} b_avail={ex_b.available_balance:.6f} b_wallet={ex_b.wallet_balance:.6f}")


# ------------------------------------------------------------ strategy logic
def make_pipeline(pid: str, side: str | None = "long_x", gross: float = 150_000.0,
                  x_qty: float = 1000.0, y_qty: float = 1000.0,
                  x_entry: float = 100.0, y_entry: float = 50.0,
                  lookback: float = 2880.0):
    sizing = SizingState(position_side=side, position_id=pid, x_quantity=x_qty,
                         y_quantity=y_qty, current_gross_notional=gross,
                         target_hedge_ratio=1.0, entry_count=1 if side else 0)
    protection = PositionProtectionState(
        active=side is not None, side=side, entry_x_price=x_entry, entry_y_price=y_entry,
        entry_gross_notional=gross, entry_x_quantity=x_qty, entry_y_quantity=y_qty,
    )
    return SimpleNamespace(
        pair_def=SimpleNamespace(pair_id=pid, x_symbol=f"{pid}X", y_symbol=f"{pid}Y",
                                 target_capital=gross),
        state=SimpleNamespace(sizing_state=sizing, protection_state=protection, last_bar_index=1000),
        estimator=SimpleNamespace(model_lookback_bars=lookback),
    )


def make_candidate(pid: str = "new", target: float = 600_000.0,
                   pvalue: float | None = 0.3, move: float | None = 0.25) -> dict:
    raw = RawPairTarget(pair_id=pid, ready=True, side="long_x", x_weight=0.5, y_weight=0.5,
                        x_price=100.0, y_price=50.0, gross_notional=target, target_capital=target)
    raw.para = {"protection": {"theoretical_zero_return_x_move": move}}
    result = SimpleNamespace(estimator=SimpleNamespace(cointegration_pvalue=pvalue))
    return {"pair_id": pid, "raw_target": raw, "result": result}


def make_strategy(available: float = 100_000.0, replacement_enabled: bool = True,
                  min_move: float = 0.20, max_pvalue: float = 0.4) -> MultiPairStrategy:
    strategy = MultiPairStrategy.__new__(MultiPairStrategy)
    strategy.rebalance_cfg = RebalanceConfig(
        enabled=True,
        minimum_entry_capital_ratio=0.5,
        closed_pair_freeze_model_lookback_multiplier=0.5,
        profitable_position_replacement=ProfitablePositionReplacementConfig(
            enabled=replacement_enabled,
            min_theoretical_zero_return_x_move=min_move,
            max_adf_pvalue=max_pvalue,
        ),
    )
    strategy.estimator_cfg = SimpleNamespace(model_lookback_bars=2880)
    strategy.fee_rate = 0.0005
    strategy.slippage_rate = 0.0001
    strategy.portfolio_state = PortfolioState(ready=True, equity=3_000_000, available_balance=available)
    strategy.pipelines = {}
    strategy._pending_rebalance_pair_ids = set()
    strategy._pending_rebalance_release = 0.0
    strategy.closed_log = []

    def fake_close(pair_id, pipeline, pair_def, fallback_hedge, **kwargs):
        strategy.closed_log.append((pair_id, kwargs.get("exit_reason"),
                                    kwargs.get("protection_freeze_bars")))
        return [SimpleNamespace(pair_id=pair_id)]

    strategy._close_orders = fake_close
    return strategy


def bundle(x_close: float, y_close: float = 50.0):
    return SimpleNamespace(x_bar=SimpleNamespace(close=x_close),
                           y_bar=SimpleNamespace(close=y_close))


def probe_unrealized_return() -> None:
    print("\n== C. _pair_unrealized_return ==")
    strategy = make_strategy()
    flat = make_pipeline("flat")

    strategy.fee_rate = 0.0005
    strategy.slippage_rate = 0.0001

    long_win = strategy._pair_unrealized_return(flat, bundle(110.0))
    check("C1 a long_x pair up +10% on the X leg reports a positive net return",
          abs(long_win - (10_000.0 - 96.0) / 150_000.0) < 1e-12, f"{long_win:.6f}")

    short_pipe = make_pipeline("short", side="short_x")
    short_win = strategy._pair_unrealized_return(short_pipe, bundle(90.0))
    # X down 10%: gross +10000, close cost is marked on the CURRENT (90) price -> 84
    check("C2 a short_x pair is marked with the correct sign (X down = profit)",
          abs(short_win - (10_000.0 - 84.0) / 150_000.0) < 1e-12, f"{short_win:.6f}")

    short_lose = strategy._pair_unrealized_return(short_pipe, bundle(110.0))
    long_lose = strategy._pair_unrealized_return(flat, bundle(90.0))
    check("C2b a short_x pair losing on an X rally mirrors a losing long (same gross, "
          "close cost marked on its own current price)",
          abs(short_lose - (-(10_000.0 + 96.0) / 150_000.0)) < 1e-12
          and abs(long_lose - (-(10_000.0 + 84.0) / 150_000.0)) < 1e-12
          and short_lose < 0.0 and long_lose < 0.0,
          f"short_lose={short_lose:.6f} long_lose={long_lose:.6f}")

    no_move = strategy._pair_unrealized_return(flat, bundle(100.0))
    check("C3 a flat position still reports a negative net return (close cost counted once)",
          abs(no_move - (-90.0 / 150_000.0)) < 1e-12, f"{no_move:.8f}")

    check("C4 the short leg's own PnL is not double counted (Y unchanged contributes 0)",
          abs(strategy._pair_unrealized_return(flat, bundle(100.0, 50.0)) - no_move) < 1e-15)


def probe_freeze_bars() -> None:
    print("\n== D. _rebalance_freeze_bars ==")
    strategy = make_strategy()
    default_pipe = make_pipeline("d")
    check("D1 ceil(2880 * 0.5) == 1440 bars",
          strategy._rebalance_freeze_bars(default_pipe) == ceil(2880 * 0.5),
          f"{strategy._rebalance_freeze_bars(default_pipe)}")

    override_pipe = make_pipeline("o", lookback=1000.0)
    check("D2 a pair-level model_lookback_bars_override (1000) drives the freeze (500)",
          strategy._rebalance_freeze_bars(override_pipe) == 500,
          f"{strategy._rebalance_freeze_bars(override_pipe)}")

    strategy.rebalance_cfg.closed_pair_freeze_model_lookback_multiplier = 0.0
    check("D3 multiplier 0 disables the freeze without raising",
          strategy._rebalance_freeze_bars(default_pipe) == 0)
    strategy.rebalance_cfg.closed_pair_freeze_model_lookback_multiplier = 0.5

    strategy.estimator_cfg = SimpleNamespace(model_lookback_bars=0)
    zero_pipe = make_pipeline("z", lookback=0.0)
    try:
        strategy._rebalance_freeze_bars(zero_pipe)
        check("D4 a non-positive lookback raises instead of silently freezing 0 bars", False)
    except ValueError:
        check("D4 a non-positive lookback raises instead of silently freezing 0 bars", True)


def probe_plan_rebalance() -> None:
    print("\n== E. _plan_rebalance ==")
    # available 100k, candidate target 600k -> minimum 300k -> release_needed 200k
    strategy = make_strategy(available=100_000.0)
    losing = make_pipeline("losing")      # X 100 -> 90  (worst)
    flat = make_pipeline("flat")          # X 100 -> 100 (mid)
    winning = make_pipeline("winning")    # X 100 -> 110 (best)
    strategy.pipelines = {
        "new": make_pipeline("new", side=None, gross=0.0, x_qty=0.0, y_qty=0.0),
        "losing": losing, "flat": flat, "winning": winning,
    }
    bundles = {"losing": bundle(90.0), "flat": bundle(100.0), "winning": bundle(110.0)}
    orders = strategy._plan_rebalance({"new": make_candidate()}, bundles, 1000)

    closed = [row[0] for row in strategy.closed_log]
    check("E1 worst-first batching closes exactly the losers needed (losing, then flat)",
          closed == ["losing", "flat"], f"closed={closed}")
    check("E2 the profitable pair is kept",
          "winning" not in closed)
    check("E3 released capital equals the summed closed gross (300k >= 200k needed)",
          abs(strategy._pending_rebalance_release - 300_000.0) < 1e-9,
          f"{strategy._pending_rebalance_release:.0f}")
    check("E4 close orders carry the rebalance reason and the model-lookback freeze",
          all(row[1] == "rebalance_replacement" and row[2] == 1440 for row in strategy.closed_log),
          f"{strategy.closed_log}")
    check("E5 closed pairs are pinned so the same bar cannot re-close them",
          strategy._pending_rebalance_pair_ids == {"losing", "flat"})

    # one big loser is enough -> only one close
    strategy2 = make_strategy(available=100_000.0)
    big = make_pipeline("big", gross=250_000.0)
    strategy2.pipelines = {
        "new": make_pipeline("new", side=None, gross=0.0, x_qty=0.0, y_qty=0.0),
        "big": big, "winning": make_pipeline("winning"),
    }
    strategy2._plan_rebalance({"new": make_candidate()},
                              {"big": bundle(90.0), "winning": bundle(110.0)}, 1000)
    check("E6 batching stops as soon as the shortfall is covered",
          [row[0] for row in strategy2.closed_log] == ["big"],
          f"closed={[row[0] for row in strategy2.closed_log]}")

    # already-planned pairs must not be planned again
    strategy2._pending_rebalance_release = 0.0
    strategy2.closed_log.clear()
    strategy2._plan_rebalance({"new": make_candidate()},
                              {"big": bundle(90.0), "winning": bundle(110.0)}, 1001)
    check("E7 a pair already pending a rebalance close is not selected again",
          [row[0] for row in strategy2.closed_log] == ["winning"],
          f"closed={[row[0] for row in strategy2.closed_log]}")

    # fully funded candidate -> no liquidation at all
    strategy3 = make_strategy(available=3_000_000.0)
    strategy3.pipelines = {"new": make_pipeline("new", side=None, gross=0.0, x_qty=0.0, y_qty=0.0),
                           "winning": make_pipeline("winning")}
    strategy3._plan_rebalance({"new": make_candidate()}, {"winning": bundle(110.0)}, 1000)
    check("E8 a fully funded candidate liquidates nothing",
          strategy3.closed_log == [] and strategy3._pending_rebalance_release == 0.0)

    # partially fundable candidate (>= minimum) -> no liquidation
    strategy4 = make_strategy(available=350_000.0)
    strategy4.pipelines = {"new": make_pipeline("new", side=None, gross=0.0, x_qty=0.0, y_qty=0.0),
                           "winning": make_pipeline("winning")}
    strategy4._plan_rebalance({"new": make_candidate()}, {"winning": bundle(110.0)}, 1000)
    check("E9 a candidate that clears its minimum but not its target liquidates nothing",
          strategy4.closed_log == [])

    # every eligible pair is profitable -> the replacement gate must veto
    strategy5 = make_strategy(available=100_000.0, replacement_enabled=True, min_move=0.20)
    strategy5.pipelines = {
        "new": make_pipeline("new", side=None, gross=0.0, x_qty=0.0, y_qty=0.0),
        "w1": make_pipeline("w1"), "w2": make_pipeline("w2"),
    }
    bundles5 = {"w1": bundle(110.0), "w2": bundle(110.0)}
    strategy5._plan_rebalance({"new": make_candidate(move=0.10)}, bundles5, 1000)
    check("E10 all-profitable book + weak candidate (move 0.10 < 0.20) liquidates nothing",
          strategy5.closed_log == [], f"closed={[r[0] for r in strategy5.closed_log]}")

    strategy6 = make_strategy(available=100_000.0, replacement_enabled=True, min_move=0.20)
    strategy6.pipelines = {
        "new": make_pipeline("new", side=None, gross=0.0, x_qty=0.0, y_qty=0.0),
        "w1": make_pipeline("w1"), "w2": make_pipeline("w2"),
    }
    strategy6._plan_rebalance({"new": make_candidate(move=0.25, pvalue=0.30)}, bundles5, 1000)
    check("E11 same book + strong candidate (move 0.25) closes the profitable pairs",
          len(strategy6.closed_log) > 0, f"closed={[r[0] for r in strategy6.closed_log]}")

    strategy7 = make_strategy(available=100_000.0, replacement_enabled=True)
    strategy7.pipelines = {
        "new": make_pipeline("new", side=None, gross=0.0, x_qty=0.0, y_qty=0.0),
        "w1": make_pipeline("w1"), "w2": make_pipeline("w2"),
    }
    strategy7._plan_rebalance({"new": make_candidate(move=0.25, pvalue=0.50)}, bundles5, 1000)
    check("E12 weak cointegration (pvalue 0.50 > 0.40) vetoes liquidation of profitable pairs",
          strategy7.closed_log == [])

    strategy8 = make_strategy(available=100_000.0, replacement_enabled=False)
    strategy8.pipelines = {
        "new": make_pipeline("new", side=None, gross=0.0, x_qty=0.0, y_qty=0.0),
        "w1": make_pipeline("w1"), "w2": make_pipeline("w2"),
    }
    strategy8._plan_rebalance({"new": make_candidate(move=0.25)}, bundles5, 1000)
    check("E13 replacement gate disabled -> profitable pairs are never liquidated",
          strategy8.closed_log == [])

    # losing pairs are exempt from the quality gate
    strategy9 = make_strategy(available=100_000.0, replacement_enabled=False)
    strategy9.pipelines = {
        "new": make_pipeline("new", side=None, gross=0.0, x_qty=0.0, y_qty=0.0),
        "l1": make_pipeline("l1"), "l2": make_pipeline("l2"),
    }
    strategy9._plan_rebalance({"new": make_candidate(move=None, pvalue=None)},
                              {"l1": bundle(90.0), "l2": bundle(90.0)}, 1000)
    check("E14 losing pairs are closed even with the gate disabled and no candidate quality data",
          len(strategy9.closed_log) == 2, f"closed={[r[0] for r in strategy9.closed_log]}")


def probe_capacity() -> None:
    """How many pairs can actually be open at once now that slots are gone."""
    print("\n== F. Implied concurrency from cash / target_capital ==")
    import yaml

    with open(ROOT / "config" / "config.yaml", "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    setup = (cfg.get("setups") or {}).get(cfg.get("active_setup")) or {}
    pairs = setup.get("pairs", []) or []
    cash = float((cfg.get("backtest") or {}).get("initial_cash", 0.0) or 0.0)
    ratio = float((setup.get("pipeline", {}).get("portfolio", {}) or {})
                  .get("minimum_entry_capital_ratio", 0.5) or 0.5)

    ordered = [(p.get("pair_id"), float(p.get("target_capital") or 0.0),
                "tradfi" in (p.get("tags") or [])) for p in pairs if p.get("enabled", True)]
    crypto = [row for row in ordered if not row[2]]
    tradfi = [row for row in ordered if row[2]]
    check("F1 every enabled pair declares target_capital",
          bool(ordered) and all(row[1] > 0 for row in ordered),
          f"pairs={len(ordered)} missing={[row[0] for row in ordered if row[1] <= 0]}")

    # Mirror PairTargetCapitalAllocator.allocate() in config order for one bar
    # on which every crypto pair signals at the same time.
    available = cash
    funded = []
    for pid, target, _tradfi in crypto:
        if available < target * ratio - 1e-9:
            continue
        available -= min(target, available)
        funded.append(pid)
        if available <= 1e-9:
            break
    print(f"    cash={cash:,.0f}  crypto pairs={len(crypto)}  tradfi pairs={len(tradfi)}")
    print(f"    sum(target_capital) crypto={sum(r[1] for r in crypto):,.0f} "
          f"tradfi={sum(r[1] for r in tradfi):,.0f}")
    print(f"    simultaneous crypto fills possible if all signal at once: {len(funded)}")
    check("F2 cash alone still bounds concurrency (no slot limit needed)", 1 <= len(funded) <= len(crypto),
          f"{len(funded)} pairs")


def probe_missing_target_capital() -> None:
    """What a pair without target_capital silently falls back to."""
    print("\n== G. Pair without target_capital ==")
    from core.modules.portfolio.allocator import _raw_target_capital

    # beta_neutral sizing returns weights only, so a pair without target_capital
    # reaches the allocator with gross_notional == 0.
    raw = RawPairTarget(pair_id="p", ready=True, side="long_x", x_weight=0.6, y_weight=0.4,
                        x_price=100.0, y_price=50.0, gross_notional=0.0, target_capital=None)
    raw.para = {"pair": {"target_capital": None},
                "portfolio": {"remaining_pair_gross": 10_000.0, "entry_gross_cap": 10_000.0}}
    check("G1 allocator falls back to sizing.notional (10k) as the WHOLE-PAIR capital",
          abs(_raw_target_capital(raw) - 10_000.0) < 1e-9,
          f"{_raw_target_capital(raw):.0f}")

    strategy = MultiPairStrategy.__new__(MultiPairStrategy)
    strategy.sizing_cfg = SimpleNamespace(notional=10_000.0)
    idle = make_pipeline("idle", side=None, gross=0.0, x_qty=0.0, y_qty=0.0)
    pair_def = SimpleNamespace(pair_id="idle", target_capital=None)
    check("G2 _pair_target_capital returns the same 10k pair cap",
          abs(strategy._pair_target_capital(pair_def, idle) - 10_000.0) < 1e-9,
          f"{strategy._pair_target_capital(pair_def, idle):.0f}")
    print("    -> a pair that forgets target_capital silently trades at 1/60 of a 600k pair "
          "instead of raising a config error")


def probe_config_loading() -> None:
    """Load the real config and build the real strategy (no market data needed)."""
    print("\n== H. Real config -> strategy construction ==")
    from core.modules.config import load_config
    from core.modules.strategy.factory import build_strategy

    cfg = load_config(ROOT / "config" / "config.yaml")
    check("H1 active setup pairs loaded", len(cfg.strategy.pairs) == 73, f"{len(cfg.strategy.pairs)}")
    check("H2 initial_cash is 3,000,000", abs(cfg.initial_cash - 3_000_000.0) < 1e-9,
          f"{cfg.initial_cash:,.0f}")

    strategy = build_strategy(cfg.strategy, cfg.symbols, fee_rate=cfg.fee_rate,
                              slippage_bps=cfg.slippage_bps)
    check("H3 allocator is PairTargetCapitalAllocator",
          type(strategy.portfolio).__name__ == "PairTargetCapitalAllocator",
          type(strategy.portfolio).__name__)
    check("H4 portfolio.minimum_entry_capital_ratio == 0.5",
          abs(strategy.portfolio.minimum_entry_capital_ratio - 0.5) < 1e-12)
    check("H5 rebalance block is read from pipeline.rebalance",
          strategy.rebalance_cfg.enabled is True
          and abs(strategy.rebalance_cfg.minimum_entry_capital_ratio - 0.5) < 1e-12,
          f"enabled={strategy.rebalance_cfg.enabled}")
    rule = strategy.rebalance_cfg.profitable_position_replacement
    check("H6 nested profitable_position_replacement is parsed",
          rule.enabled is True
          and abs(rule.min_theoretical_zero_return_x_move - 0.20) < 1e-12
          and abs(rule.max_adf_pvalue - 0.4) < 1e-12,
          f"enabled={rule.enabled} move={rule.min_theoretical_zero_return_x_move} p={rule.max_adf_pvalue}")
    check("H7 closed_pair_freeze_model_lookback_multiplier == 0.5",
          abs(strategy.rebalance_cfg.closed_pair_freeze_model_lookback_multiplier - 0.5) < 1e-12)
    missing = [p.pair_id for p in strategy.pairs if not p.target_capital]
    check("H8 every PairDefinition carries target_capital", missing == [], f"missing={missing}")

    pipelines = getattr(strategy, "pipelines", {}) or {}
    check("H9 pipelines were built", len(pipelines) == 73, f"{len(pipelines)}")
    if pipelines:
        pid = next(iter(pipelines))
        pipe = pipelines[pid]
        lookback = getattr(pipe.estimator, "model_lookback_bars", None)
        freeze = strategy._rebalance_freeze_bars(pipe)
        check("H10 real estimator lookback drives the freeze (2880 -> 1440)",
              lookback == 2880 and freeze == 1440, f"lookback={lookback} freeze={freeze}")


def main() -> int:
    probe_allocator()
    probe_cross_exchange_scale()
    probe_unrealized_return()
    probe_freeze_bars()
    probe_plan_rebalance()
    probe_capacity()
    probe_missing_target_capital()
    probe_config_loading()
    print("\n==================== SUMMARY ====================")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("all probes passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
