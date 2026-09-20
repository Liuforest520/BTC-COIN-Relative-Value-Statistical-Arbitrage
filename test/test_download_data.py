from datetime import datetime

import pytest

from core.modules.data.download_data import DatasetSpec, DownloadRequest, MySQLDataDownloader


def _spec(*, table_name: str, dataset: str | None = None, timeframe_column: str | None = None) -> DatasetSpec:
    return DatasetSpec(
        dataset=dataset or "binance_usd_margin_kline_1m",
        table_name=table_name,
        columns=["open_time", "symbol", "open", "close"],
        time_column="open_time",
        time_column_type="bigint",
        time_unit="ms",
        symbol_column="symbol",
        timeframe_column=timeframe_column,
    )


def _request(timeframe: str) -> DownloadRequest:
    return DownloadRequest(
        dataset="binance_usd_margin_kline_1m",
        start="20240101",
        end="20240101",
        symbol="BTCUSDT",
        timeframe=timeframe,
    )


def _build_query(spec: DatasetSpec, request: DownloadRequest):
    downloader = object.__new__(MySQLDataDownloader)
    return downloader._build_select_query(
        spec,
        request,
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
    )


def test_timeframe_filter_is_skipped_when_kline_table_name_encodes_same_timeframe():
    spec = _spec(table_name="binance_usd_margin_kline_1m_20240101_20240107")

    sql, params = _build_query(spec, _request("1m"))

    assert "timeframe" not in sql
    assert params[-1] == "BTCUSDT"
    assert "1m" not in params


def test_timeframe_filter_is_kept_when_table_has_timeframe_column():
    spec = _spec(
        table_name="binance_usd_margin_kline_20240101_20240107",
        dataset="binance_usd_margin_kline",
        timeframe_column="timeframe",
    )

    sql, params = _build_query(spec, _request("1m"))

    assert "`timeframe` = %s" in sql
    assert params[-1] == "1m"


def test_missing_timeframe_column_still_rejects_mismatched_requested_timeframe():
    spec = _spec(table_name="binance_usd_margin_kline_1m_20240101_20240107")

    with pytest.raises(ValueError, match="does not encode requested timeframe"):
        _build_query(spec, _request("5m"))
