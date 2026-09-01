"""
Preview downloadable MySQL market data before running a large download.

This script is read-only. It checks table prefixes, time coverage, symbol
coverage, and prints candidate download commands.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.modules.data.download_data import MySQLConfig, MySQLDataDownloader  # noqa: E402


DEFAULT_PREFIXES = [
    "binance_usd_margin_funding_rate_daily",
    "binance_usd_margin_kline_1m",
    "binance_usd_margin_kline_3m",
]


def parse_args(argv: list[str] | None = None):
    today = date.today()
    default_end = today.strftime("%Y%m%d")
    default_start = (today - timedelta(days=365 * 3)).strftime("%Y%m%d")

    parser = argparse.ArgumentParser(
        description="Preview MySQL data coverage by table prefix before downloading.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prefix", action="append", default=None, help="table prefix; can be passed multiple times")
    parser.add_argument("--start", default=default_start, help="inclusive start date YYYYMMDD")
    parser.add_argument("--end", default=default_end, help="inclusive end date YYYYMMDD")
    parser.add_argument("--mysql-config", default=str(Path("config/mysql.config")))
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--top-symbols", type=int, default=30)
    parser.add_argument(
        "--symbol-mode",
        choices=["none", "latest", "union"],
        default="latest",
        help="latest samples symbols from the latest overlapping table; union scans all overlapping tables",
    )
    parser.add_argument("--sample-rows", type=int, default=3, help="latest rows to show for each prefix")
    parser.add_argument("--deep", action="store_true", help="query real MIN/MAX time for every matched table; slower")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--charset", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = MySQLConfig.from_file_or_env(args.mysql_config)
    for field, value in [
        ("host", args.host),
        ("port", args.port),
        ("user", args.user),
        ("password", args.password),
        ("database", args.database),
        ("charset", args.charset),
    ]:
        if value is not None:
            setattr(config, field, value)
    config.__post_init__()

    downloader = MySQLDataDownloader(config)
    start_dt = parse_yyyymmdd(args.start)
    end_exclusive = parse_yyyymmdd(args.end) + timedelta(days=1)
    prefixes = args.prefix or DEFAULT_PREFIXES

    print(f"数据库: {config.database}")
    print(f"时间窗口: {args.start} -> {args.end}（end 按整天包含）")
    print(f"前缀数量: {len(prefixes)}")

    for prefix in prefixes:
        preview_prefix(downloader, prefix, start_dt, end_exclusive, args)
    return 0


def preview_prefix(
    downloader: MySQLDataDownloader,
    prefix: str,
    start_dt: datetime,
    end_exclusive: datetime,
    args,
) -> None:
    print("\n" + "=" * 88)
    print(f"前缀: {prefix}")

    tables = downloader.list_datasets(prefix + "%")
    if not tables:
        print("匹配表数: 0")
        print("结论: 当前数据库没有这个前缀的表。请确认前缀是否改名或数据是否尚未入库。")
        return

    table_infos = []
    first_spec = inspect_table_fast(downloader, tables[0])
    approx_rows_total = 0
    row_counts = approx_table_rows_map(downloader, tables)

    for table in tables:
        row_count = row_counts.get(table)
        approx_rows_total += int(row_count or 0)

        if args.deep:
            spec = inspect_table_fast(downloader, table)
            mn, mx = time_bounds_in_window(downloader, spec, start_dt, end_exclusive)
            if mn is None or mx is None:
                continue
            mn_dt = to_dt(downloader, mn, spec.time_unit)
            mx_dt = to_dt(downloader, mx, spec.time_unit)
        else:
            spec = None
            table_range = table_date_range(table)
            if table_range is None:
                spec = inspect_table_fast(downloader, table)
                mn, mx = time_bounds_in_window(downloader, spec, start_dt, end_exclusive)
                if mn is None or mx is None:
                    continue
                mn_dt = to_dt(downloader, mn, spec.time_unit)
                mx_dt = to_dt(downloader, mx, spec.time_unit)
            else:
                table_start, table_end_exclusive = table_range
                if table_end_exclusive <= start_dt or table_start >= end_exclusive:
                    continue
                mn_dt = max(table_start, start_dt)
                mx_dt = min(table_end_exclusive, end_exclusive) - timedelta(seconds=1)

        if mn_dt is None or mx_dt is None:
            continue
        table_infos.append({"table": table, "spec": spec, "mn_dt": mn_dt, "mx_dt": mx_dt, "rows": row_count})

    print(f"匹配表数: {len(tables)}")
    print(f"第一张表: {tables[0]}")
    print(f"最后一张表: {tables[-1]}")
    print(f"全表近似行数: {approx_rows_total:,}")

    if first_spec is not None:
        print(f"时间列: {first_spec.time_column} ({first_spec.time_column_type} -> {first_spec.time_unit})")
        print(f"symbol列: {first_spec.symbol_column or '无'}")
        print(f"周期列: {first_spec.timeframe_column or '无'}")
        print(f"字段: {', '.join(first_spec.columns[:20])}" + (" ..." if len(first_spec.columns) > 20 else ""))

    if not table_infos:
        print("窗口内可用表数: 0")
        print("结论: 这个前缀有表，但在指定时间窗口内没有数据。")
        return

    first_available = min(table_infos, key=lambda item: item["mn_dt"])
    last_available = max(table_infos, key=lambda item: item["mx_dt"])
    if last_available["spec"] is None:
        last_available["spec"] = inspect_table_fast(downloader, last_available["table"])
    print(f"窗口内可用表数: {len(table_infos)}")
    print(
        "窗口内时间范围: "
        f"{first_available['mn_dt'].strftime('%Y-%m-%d %H:%M:%S UTC')}"
        " -> "
        f"{last_available['mx_dt'].strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    print(f"窗口内第一张表: {first_available['table']}")
    print(f"窗口内最后一张表: {last_available['table']}")

    latest = last_available
    if args.symbol_mode != "none" and latest["spec"].symbol_column:
        symbols = symbols_for_tables(downloader, table_infos, args.symbol_mode, args.top_symbols)
        print(f"symbols({args.symbol_mode}): 共 {symbols['count']} 个")
        if symbols["sample"]:
            print(f"symbols样例: {', '.join(symbols['sample'])}")

    samples = latest_rows(downloader, latest["spec"], args.sample_rows)
    if samples:
        print(f"最近 {len(samples)} 行样例（来自 {latest['table']}）:")
        for row in samples:
            print(f"  {row}")

    print("下载命令:")
    print(
        "  全量: "
        f"python scripts/download_data.py --dataset {prefix} --start {args.start} --end {args.end} "
        f"--output-dir {args.output_dir}"
    )
    if latest["spec"].symbol_column:
        print(
            "  单symbol示例: "
            f"python scripts/download_data.py --dataset {prefix} --symbol BTCUSDT --start {args.start} --end {args.end} "
            f"--output-dir {args.output_dir}"
        )


def inspect_table_fast(downloader: MySQLDataDownloader, table: str):
    columns = [
        {
            "column_name": item.get("column_name") or item.get("COLUMN_NAME"),
            "data_type": item.get("data_type") or item.get("DATA_TYPE"),
        }
        for item in downloader._load_columns(table)
    ]
    time_column, time_column_type = downloader._resolve_time_column(columns)
    time_unit = downloader._resolve_time_unit(table, time_column, time_column_type)
    return type("PreviewSpec", (), {
        "table_name": table,
        "columns": [item["column_name"] for item in columns],
        "time_column": time_column,
        "time_column_type": time_column_type,
        "time_unit": time_unit,
        "symbol_column": downloader._first_matching_column(columns, ["symbol", "inst_id", "instrument", "ticker"]),
        "timeframe_column": downloader._first_matching_column(columns, ["timeframe", "interval", "period"]),
    })()


def table_date_range(table: str) -> tuple[datetime, datetime] | None:
    weekly = re.search(r"_(\d{8})_(\d{8})$", table)
    if weekly:
        start = parse_yyyymmdd(weekly.group(1))
        end_exclusive = parse_yyyymmdd(weekly.group(2)) + timedelta(days=1)
        return start, end_exclusive

    yearly = re.search(r"_(\d{4})$", table)
    if yearly:
        year = int(yearly.group(1))
        return datetime(year, 1, 1), datetime(year + 1, 1, 1)
    return None


def approx_table_rows(downloader: MySQLDataDownloader, table: str) -> int | None:
    sql = (
        "SELECT TABLE_ROWS AS n FROM information_schema.tables "
        "WHERE table_schema = %s AND table_name = %s"
    )
    row = fetch_one(downloader, sql, (downloader.config.database, table))
    return row["n"] if row else None


def approx_table_rows_map(downloader: MySQLDataDownloader, tables: list[str]) -> dict[str, int | None]:
    if not tables:
        return {}
    placeholders = ", ".join(["%s"] * len(tables))
    sql = (
        "SELECT table_name, TABLE_ROWS AS n FROM information_schema.tables "
        f"WHERE table_schema = %s AND table_name IN ({placeholders})"
    )
    rows = fetch_all(downloader, sql, [downloader.config.database, *tables])
    result = {}
    for row in rows:
        table_name = row.get("table_name") or row.get("TABLE_NAME")
        count = row.get("n") if "n" in row else row.get("TABLE_ROWS")
        if table_name is not None:
            result[str(table_name)] = count
    return result


def time_bounds_in_window(
    downloader: MySQLDataDownloader,
    spec,
    start_dt: datetime,
    end_exclusive: datetime,
) -> tuple[Any, Any]:
    lower = downloader._time_value_for_query(start_dt, spec.time_unit)
    upper = downloader._time_value_for_query(end_exclusive, spec.time_unit)
    sql = (
        f"SELECT MIN(`{spec.time_column}`) AS mn, MAX(`{spec.time_column}`) AS mx "
        f"FROM `{spec.table_name}` "
        f"WHERE `{spec.time_column}` >= %s AND `{spec.time_column}` < %s"
    )
    row = fetch_one(downloader, sql, (lower, upper))
    return (row["mn"], row["mx"]) if row else (None, None)


def symbols_for_tables(downloader: MySQLDataDownloader, table_infos: list[dict], mode: str, top: int) -> dict:
    if mode == "latest":
        table_infos = [table_infos[-1]]

    symbols = set()
    for info in table_infos:
        if info["spec"] is None:
            info["spec"] = inspect_table_fast(downloader, info["table"])
        spec = info["spec"]
        sql = f"SELECT DISTINCT `{spec.symbol_column}` AS symbol FROM `{spec.table_name}` ORDER BY `{spec.symbol_column}`"
        for row in fetch_all(downloader, sql):
            if row["symbol"] is not None:
                symbols.add(str(row["symbol"]))
    ordered = sorted(symbols)
    return {"count": len(ordered), "sample": ordered[:top]}


def latest_rows(downloader: MySQLDataDownloader, spec, limit: int) -> list[dict]:
    if limit <= 0:
        return []
    columns = [spec.time_column]
    if spec.symbol_column:
        columns.append(spec.symbol_column)
    for key in ["open", "high", "low", "close", "volume", "funding_rate", "rate"]:
        if key in spec.columns and key not in columns:
            columns.append(key)
    selected = ", ".join(f"`{column}`" for column in columns[:8])
    sql = f"SELECT {selected} FROM `{spec.table_name}` LIMIT %s"
    rows = fetch_all(downloader, sql, (int(limit),))
    for row in rows:
        row[spec.time_column] = fmt_value_time(downloader, row[spec.time_column], spec.time_unit)
    return rows


def fetch_one(downloader: MySQLDataDownloader, sql: str, params=None) -> dict | None:
    rows = fetch_all(downloader, sql, params)
    return rows[0] if rows else None


def fetch_all(downloader: MySQLDataDownloader, sql: str, params=None) -> list[dict]:
    with downloader.connection() as conn:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(sql, tuple(params or ()))
            return cur.fetchall()
        finally:
            cur.close()


def parse_yyyymmdd(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d")


def to_dt(downloader: MySQLDataDownloader, value, unit: str) -> datetime | None:
    return downloader._time_value_to_datetime(value, unit)


def fmt_value_time(downloader: MySQLDataDownloader, value, unit: str) -> str:
    dt = to_dt(downloader, value, unit)
    if dt is None:
        return repr(value)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


if __name__ == "__main__":
    raise SystemExit(main())
