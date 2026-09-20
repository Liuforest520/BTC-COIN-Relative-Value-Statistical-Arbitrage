"""Memory-efficient buffers for long detailed backtests.

Minute-level multi-pair diagnostics are naturally columnar.  Keeping them as
one Python dictionary per minute repeats hundreds of dictionary keys and can
exhaust memory before the report is written.  These buffers preserve the
existing iterable row interface while storing numeric curves in compact C
arrays and mixed diagnostic rows by column.
"""

from __future__ import annotations

from array import array
from math import isnan
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl


CSV_CHUNK_ROWS = 25_000


class NumericCurveBuffer:
    """Columnar numeric rows with ``None`` represented as NaN."""

    def __init__(self) -> None:
        self._columns: dict[str, array] = {}
        self._kinds: dict[str, str] = {}
        self._length = 0

    def append(self, row: dict) -> None:
        for name in row:
            if name in self._columns:
                continue
            kind = "q" if name == "ts" else "d"
            fill = 0 if kind == "q" else float("nan")
            self._columns[name] = array(kind, [fill]) * self._length
            self._kinds[name] = kind

        for name, values in self._columns.items():
            value = row.get(name)
            if self._kinds[name] == "q":
                values.append(int(value) if value is not None else 0)
            else:
                try:
                    values.append(float(value) if value is not None else float("nan"))
                except (TypeError, ValueError):
                    values.append(float("nan"))
        self._length += 1

    def clear(self) -> None:
        self._columns.clear()
        self._kinds.clear()
        self._length = 0

    def write_csv(self, path: str | Path, chunk_rows: int = CSV_CHUNK_ROWS) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._length:
            path.write_text("", encoding="utf-8")
            return

        views = {
            name: np.frombuffer(values, dtype=np.int64 if self._kinds[name] == "q" else np.float64)
            for name, values in self._columns.items()
        }
        with path.open("wb") as handle:
            for start in range(0, self._length, max(1, int(chunk_rows))):
                end = min(self._length, start + max(1, int(chunk_rows)))
                frame = pl.DataFrame({name: values[start:end] for name, values in views.items()})
                frame.write_csv(handle, include_header=start == 0)

    def __len__(self) -> int:
        return self._length

    def __bool__(self) -> bool:
        return self._length > 0

    def __iter__(self) -> Iterator[dict]:
        names = list(self._columns)
        for index in range(self._length):
            row = {}
            for name in names:
                value = self._columns[name][index]
                if self._kinds[name] == "d" and isnan(value):
                    value = None
                row[name] = value
            yield row

    def __eq__(self, other) -> bool:
        if isinstance(other, NumericCurveBuffer):
            return self._columns == other._columns
        if isinstance(other, list):
            normalized = [
                {key: value for key, value in row.items() if value is not None}
                for row in self
            ]
            return normalized == other
        return NotImplemented


class ColumnarRowBuffer:
    """Fixed-schema mixed rows stored as one Python list per column."""

    def __init__(self, schema: list[tuple[str, pl.DataType]]) -> None:
        self.schema = list(schema)
        self._columns = {name: [] for name, _dtype in self.schema}
        self._length = 0

    def append(self, row: dict) -> None:
        for name in self._columns:
            self._columns[name].append(row.get(name))
        self._length += 1

    def clear(self) -> None:
        for values in self._columns.values():
            values.clear()
        self._length = 0

    def write_csv(self, path: str | Path, chunk_rows: int = CSV_CHUNK_ROWS) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._length:
            path.write_text("", encoding="utf-8")
            return

        with path.open("wb") as handle:
            for start in range(0, self._length, max(1, int(chunk_rows))):
                end = min(self._length, start + max(1, int(chunk_rows)))
                frame = pl.DataFrame(
                    {name: values[start:end] for name, values in self._columns.items()},
                    schema=self.schema,
                )
                frame.write_csv(handle, include_header=start == 0)

    def __len__(self) -> int:
        return self._length

    def __bool__(self) -> bool:
        return self._length > 0

    def __iter__(self) -> Iterator[dict]:
        names = list(self._columns)
        for index in range(self._length):
            yield {name: self._columns[name][index] for name in names}

    def __eq__(self, other) -> bool:
        if isinstance(other, ColumnarRowBuffer):
            return self._columns == other._columns
        if isinstance(other, list):
            return list(self) == other
        return NotImplemented


PAIR_STATE_SCHEMA = [
    ("ts", pl.Int64),
    ("pair_id", pl.String),
    ("bar_index", pl.Int64),
    ("global_bar_index", pl.Int64),
    ("is_ready", pl.Boolean),
    ("alpha", pl.Float64),
    ("beta", pl.Float64),
    ("hedge_beta", pl.Float64),
    ("spread_mean", pl.Float64),
    ("spread_std", pl.Float64),
    ("half_life", pl.Float64),
    ("cointegration_pass", pl.Boolean),
    ("cointegration_pvalue", pl.Float64),
    ("cointegration_stat", pl.Float64),
    ("cointegration_block_open", pl.Boolean),
    ("cointegration_reason", pl.String),
    ("last_model_update_index", pl.Int64),
    ("last_hedge_model_update_index", pl.Int64),
    ("next_model_update_index", pl.Int64),
    ("next_hedge_model_update_index", pl.Int64),
    ("last_model_update_skip_index", pl.Int64),
    ("last_hedge_model_update_skip_index", pl.Int64),
    ("block_reason", pl.String),
]
