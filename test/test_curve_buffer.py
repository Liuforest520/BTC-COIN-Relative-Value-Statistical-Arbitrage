from pathlib import Path

import polars as pl

from core.backtest.curve_buffer import ColumnarRowBuffer, NumericCurveBuffer


def test_numeric_curve_buffer_preserves_rows_and_writes_in_chunks(tmp_path: Path):
    buffer = NumericCurveBuffer()
    buffer.append({"ts": 1, "equity": 100.0})
    buffer.append({"ts": 2, "equity": 101.0, "zscore": 2.5})

    assert buffer == [
        {"ts": 1, "equity": 100.0},
        {"ts": 2, "equity": 101.0, "zscore": 2.5},
    ]

    output = tmp_path / "numeric.csv"
    buffer.write_csv(output, chunk_rows=1)
    frame = pl.read_csv(output)

    assert frame.height == 2
    assert frame["ts"].to_list() == [1, 2]
    assert frame["equity"].to_list() == [100.0, 101.0]
    assert frame["zscore"][1] == 2.5


def test_columnar_row_buffer_writes_nullable_mixed_schema(tmp_path: Path):
    schema = [("ts", pl.Int64), ("pair_id", pl.String), ("beta", pl.Float64)]
    buffer = ColumnarRowBuffer(schema)
    buffer.append({"ts": 1, "pair_id": "btc_coin", "beta": None})
    buffer.append({"ts": 2, "pair_id": "btc_coin", "beta": 1.25})

    output = tmp_path / "mixed.csv"
    buffer.write_csv(output, chunk_rows=1)
    frame = pl.read_csv(output)

    assert frame.to_dicts() == [
        {"ts": 1, "pair_id": "btc_coin", "beta": None},
        {"ts": 2, "pair_id": "btc_coin", "beta": 1.25},
    ]
