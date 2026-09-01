import polars as pl


def test_polars_parquet_round_trip(tmp_path):
    path = tmp_path / "bars.parquet"
    expected = pl.DataFrame(
        {"ts": [1, 2], "open": [10.0, 11.0], "close": [10.5, 11.5]}
    )
    expected.write_parquet(path)

    assert pl.read_parquet(path).equals(expected)
