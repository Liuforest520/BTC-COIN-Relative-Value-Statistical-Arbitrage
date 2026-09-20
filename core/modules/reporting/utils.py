from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable

import polars as pl


# These fields are emitted by orders/trades and are allowed to be null for
# ordinary opens/adds, then become strings on a protective close.  Supplying
# their dtypes up front prevents Polars from inferring a Null builder from an
# early all-null chunk and failing when a later row contains a value.
REPORT_STRING_COLUMNS = {
    "order_id",
    "group_id",
    "exchange",
    "symbol",
    "action",
    "side",
    "order_type",
    "status",
    "cancel_order_id",
    "position_id",
    "pair_id",
    "exit_reason",
    "protection_trigger",
    "exit_class",
    "protection_stop_reason",
    "protection_rule",
}
REPORT_BOOL_COLUMNS = {"reopen_lock_pending"}
REPORT_INT_COLUMNS = {
    "ts",
    "protection_max_holding_bars",
    "protection_freeze_bars",
    "protection_freeze_until_bar",
    "protection_max_holding_deadline_bar",
    "entry_count",
    "last_entry_bar_index",
    "add_cooldown_remaining_bars",
    "min_hold_remaining_bars",
}


def normalize_rows(rows: Iterable[Any] | None) -> list[dict]:
    """Convert report objects to flat, schema-stable dictionaries.

    Report rows may be dataclasses, dictionaries, or lightweight objects.  A
    nested ``para`` payload is strategy metadata and is intentionally omitted
    from tabular output; other nested values are JSON encoded so one unusual
    row cannot change a column's Polars dtype.
    """
    converted: list[dict] = []
    for item in rows or []:
        if is_dataclass(item):
            row = asdict(item)
        elif isinstance(item, dict):
            row = dict(item)
        else:
            row = dict(vars(item))
        normalized = {}
        for key, value in row.items():
            if key == "para":
                continue
            normalized[key] = _normalize_value(value)
        converted.append(normalized)

    if not converted:
        return []

    columns: list[str] = []
    for row in converted:
        for key in row:
            if key not in columns:
                columns.append(key)
    return [{column: row.get(column) for column in columns} for row in converted]


def reporting_frame(rows: Iterable[Any] | None) -> pl.DataFrame:
    """Build a Polars frame with stable types for sparse report rows."""
    normalized = normalize_rows(rows)
    if not normalized:
        return pl.DataFrame()

    overrides = {}
    for column in normalized[0].keys():
        if column in REPORT_STRING_COLUMNS:
            overrides[column] = pl.Utf8
        elif column in REPORT_BOOL_COLUMNS:
            overrides[column] = pl.Boolean
        elif column in REPORT_INT_COLUMNS:
            overrides[column] = pl.Int64

    return pl.from_dicts(
        normalized,
        schema_overrides=overrides or None,
        infer_schema_length=None,
        strict=False,
    )


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, (dict, list, tuple)):
        # ``para`` is removed by normalize_rows; this fallback keeps any other
        # nested diagnostic value CSV-safe and deterministic.
        try:
            return json.dumps(value, ensure_ascii=False, default=str, allow_nan=False)
        except (TypeError, ValueError):
            return str(value)
    return value
