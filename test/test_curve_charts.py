"""Curve chart export: portfolio image, one image per traded pair, none for idle pairs."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from core.modules.reporting.curve_charts import export_curve_charts

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MINUTE = 60_000
START = 1_700_000_000_000  # ms epoch


def _write_symbol_csv(path: Path, base_price: float, drift: float) -> None:
    lines = ["open_time,open,high,low,close,volume"]
    for minute in range(180):
        price = base_price + drift * minute
        ts = START + minute * MINUTE
        lines.append(f"{ts},{price},{price},{price},{price},1.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _trade(ts, symbol, side, quantity, price, action, position_id, pair_id, fee=None):
    notional = quantity * price
    return SimpleNamespace(
        ts=ts,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        notional=notional,
        fee=notional * 0.0005 if fee is None else fee,
        action=action,
        position_id=position_id,
        pair_id=pair_id,
        group_id=position_id,
    )


@pytest.fixture()
def run_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_symbol_csv(data_dir / "AAAUSDT-1m.csv", 100.0, 0.10)
    _write_symbol_csv(data_dir / "BBBUSDT-1m.csv", 50.0, -0.02)

    config = {
        "data": {
            "symbols": {
                "AAAUSDT": {"path": str(data_dir / "AAAUSDT-1m.csv")},
                "BBBUSDT": {"path": str(data_dir / "BBBUSDT-1m.csv")},
                "CCCUSDT": {"path": str(data_dir / "AAAUSDT-1m.csv")},
                "DDDUSDT": {"path": str(data_dir / "BBBUSDT-1m.csv")},
            }
        },
        "active_setup": "s",
        "setups": {
            "s": {
                "pairs": [
                    {"pair_id": "aaa_bbb", "x_symbol": "AAAUSDT", "y_symbol": "BBBUSDT"},
                    {"pair_id": "idle_pair", "x_symbol": "CCCUSDT", "y_symbol": "DDDUSDT"},
                ]
            }
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    # aaa_bbb: open at minute 5, close at minute 100 (BBB falls -> long_x profits)
    trades = [
        _trade(START + 5 * MINUTE, "AAAUSDT", "buy", 10.0, 100.5, "open", "pos1", "aaa_bbb"),
        _trade(START + 5 * MINUTE, "BBBUSDT", "sell", 20.0, 49.9, "open", "pos1", "aaa_bbb"),
        _trade(START + 100 * MINUTE, "AAAUSDT", "sell", 10.0, 110.0, "close", "pos1", "aaa_bbb"),
        _trade(START + 100 * MINUTE, "BBBUSDT", "buy", 20.0, 48.0, "close", "pos1", "aaa_bbb"),
    ]
    equity_curve = [
        {"ts": START + minute * MINUTE, "equity": 100_000.0 + minute * 5.0}
        for minute in range(180)
    ]
    funding = [
        SimpleNamespace(
            ts=START + 60 * MINUTE, symbol="BBBUSDT", funding_rate=0.0001,
            mark_price=48.8, payment=-20.0 * 48.8 * 0.0001, quantity=-20.0,
        )
    ]
    result = SimpleNamespace(
        metrics={
            "initial_equity": 100_000.0,
            "final_equity": 100_895.0,
            "total_return": 0.00895,
            "annualized_return": 0.5,
            "sharpe": 1.23,
            "calmar": 2.0,
            "max_drawdown": -0.01,
            "trade_count": 1,
            "win_rate": 1.0,
            "profit_loss_ratio": 3.0,
            "average_holding_minutes": 95.0,
            "total_fee": 2.0,
            "total_slippage": 1.0,
            "funding_fee": -0.1,
        },
        equity_curve=equity_curve,
        trades=trades,
        funding_payments=funding,
    )
    out_dir = tmp_path / "results" / "run_x"
    out_dir.mkdir(parents=True)
    return result, config_path, out_dir


def test_curve_charts_write_portfolio_and_only_traded_pairs(run_dir):
    result, config_path, out_dir = run_dir

    written = export_curve_charts(result, config_path, out_dir, initial_equity=100_000.0)

    chart_dir = out_dir / "curve_charts"
    portfolio = chart_dir / "portfolio_curve.png"
    pair_chart = chart_dir / "pairs" / "aaa_bbb_curve.png"

    assert written["portfolio"] == portfolio
    assert portfolio.exists() and portfolio.read_bytes().startswith(PNG_MAGIC)
    assert pair_chart.exists() and pair_chart.read_bytes().startswith(PNG_MAGIC)
    # The idle pair has no fills, so it must not get an image.
    assert not (chart_dir / "pairs" / "idle_pair_curve.png").exists()
    assert [entry["pair_id"] for entry in written["pairs"]] == ["aaa_bbb"]


def test_pair_metrics_carry_sharpe_calmar_trades_and_drawdown(run_dir):
    result, config_path, out_dir = run_dir

    written = export_curve_charts(result, config_path, out_dir, initial_equity=100_000.0)
    stats = written["pairs"][0]["metrics"]

    assert stats["pair_id"] == "aaa_bbb"
    assert stats["closed_positions"] == 1
    assert stats["open_positions"] == 0
    assert stats["win_rate"] == 1.0
    assert stats["sharpe"] is not None and stats["sharpe"] > 0
    assert stats["calmar"] is not None
    assert stats["max_drawdown"] is not None and stats["max_drawdown"] <= 0
    assert stats["fee"] == pytest.approx(sum(trade.fee for trade in result.trades))
    # long AAA + short BBB with BBB falling: the pair must end profitable
    assert stats["net_pnl"] > 0
    assert stats["funding"] == pytest.approx(-20.0 * 48.8 * 0.0001, rel=1e-9)


def test_charts_survive_without_funding_and_without_trades(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_symbol_csv(data_dir / "AAAUSDT-1m.csv", 100.0, 0.0)
    config = {
        "data": {"symbols": {"AAAUSDT": {"path": str(data_dir / "AAAUSDT-1m.csv")}}},
        "active_setup": "s",
        "setups": {"s": {"pairs": [{"pair_id": "p", "x_symbol": "AAAUSDT", "y_symbol": "AAAUSDT"}]}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = SimpleNamespace(
        metrics={"initial_equity": 10_000.0, "final_equity": 10_000.0},
        equity_curve=[{"ts": START + i * MINUTE, "equity": 10_000.0} for i in range(10)],
        trades=[],
        funding_payments=[],
    )
    out_dir = tmp_path / "run_y"
    out_dir.mkdir()

    written = export_curve_charts(result, config_path, out_dir)

    assert (out_dir / "curve_charts" / "portfolio_curve.png").exists()
    assert written["pairs"] == []
