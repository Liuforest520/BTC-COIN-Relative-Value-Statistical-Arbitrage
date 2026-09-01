from core.modules.data.stream import iter_csv_bars
from core.modules.exchange.manager import ExchangeManager


def test_polars_csv_stream_normalizes_and_keeps_last_duplicate(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text(
        "open_time,open,high,low,close,volume,unused\n"
        "1000,1,3,0.5,2,10,x\n"
        "1000,2,4,1,3,11,y\n"
        "2000,bad,4,1,3,12,z\n"
        "3000,3,5,2,4,13,w\n",
        encoding="utf-8",
    )

    assert list(iter_csv_bars(path)) == [
        [1_000_000, 2.0, 4.0, 3.0, 1.0, 11.0],
        [3_000_000, 3.0, 5.0, 4.0, 2.0, 13.0],
    ]


def test_compact_exchange_manager_skips_cumulative_trade_snapshot():
    manager = ExchangeManager.from_names(["binance"])
    manager.compact_equity_curve = True
    manager.equity_curve = {"ts": [], "equity": []}
    manager.include_trade_history_on_bar = False

    def fail_if_called(_results):
        raise AssertionError("cumulative trade history should not be rebuilt")

    manager._all_trades = fail_if_called
    result = manager.on_bar(
        {"binance": {"BTCUSDT": [1_000, 100.0, 101.0, 100.5, 99.0, 10.0]}}
    )

    assert result["trades"] == []
    assert manager.equity_curve == {"ts": [1_000], "equity": [100_000.0]}
