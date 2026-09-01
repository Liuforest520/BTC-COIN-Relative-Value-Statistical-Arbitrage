"""
探查 MySQL 行情库中可用的数据集（只读，不下载数据）。

典型用法：
  # 1) 列出库里所有 binance_usd_margin_* 表（先看全貌）
  python maintenance/data/inspect_mysql_data.py --pattern binance_usd_margin_%

  # 2) 只看关心的几个数据集（表名前缀，可重复传）
  python maintenance/data/inspect_mysql_data.py ^
      --prefix binance_usd_margin_kline_1m ^
      --prefix binance_usd_margin_kline_3m ^
      --prefix binance_usd_margin_funding_rate_daily

  # 3) 想看每个 symbol 分别有多少行（判断哪些品种数据全；大表聚合较慢，可选）
  python maintenance/data/inspect_mysql_data.py --prefix binance_usd_margin_kline_1m --symbol-rows

  # 4) 连接参数临时覆盖（不修改 config/mysql.config）
  python maintenance/data/inspect_mysql_data.py --host 1.2.3.4 --database my_db --user u --password p

MySQL 连接配置：默认读 config/mysql.config（[mysql] 段），缺失时读 MYSQL_* 环境变量。
本脚本只执行 information_schema 查询和 SELECT COUNT/MIN/MAX/DISTINCT 等轻量统计，
不会把行情数据拉下来。确认数据可用后再用 scripts/download_data.py 下载。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台/重定向时强制 UTF-8 输出，避免 GBK 编码错误
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 复用现有下载器（MySQLConfig / MySQLDataDownloader）
from core.modules.data.download_data import (  # noqa: E402
    MySQLConfig,
    MySQLDataDownloader,
)

DAY_SECONDS = 86400
DAY_UNIT_DIV = {
    "s": 86400,
    "ms": 86400_000,
    "us": 86400_000_000,
    "ns": 86400_000_000_000,
}


# ---------------------------------------------------------------------------
# 轻量查询辅助
# ---------------------------------------------------------------------------

def fetch_one(downloader: MySQLDataDownloader, sql: str, params=None) -> dict | None:
    with downloader.connection() as conn:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(sql, params or ())
            return cur.fetchone()
        finally:
            cur.close()


def fetch_all(downloader: MySQLDataDownloader, sql: str, params=None, limit: int | None = None) -> list[dict]:
    if limit is not None:
        sql = sql.rstrip().rstrip(";") + " LIMIT %s"
        params = list(params or ()) + [int(limit)]
    with downloader.connection() as conn:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(sql, tuple(params or ()))
            return cur.fetchall()
        finally:
            cur.close()


def fmt_ts(downloader: MySQLDataDownloader, value, time_unit: str) -> str:
    """把 MIN/MAX/DISTINCT 查出来的原始时间值转成可读 UTC 字符串。"""
    dt = downloader._time_value_to_datetime(value, time_unit)
    if dt is None:
        return repr(value)
    return dt.strftime("%Y-%m-%d %H:%M:%S") + " UTC"


# ---------------------------------------------------------------------------
# 每张表的统计
# ---------------------------------------------------------------------------

def table_row_stats(downloader: MySQLDataDownloader, table: str, exact: bool) -> int | None:
    """行数：--exact-count 时 COUNT(*)（大表慢），否则 information_schema 近似值。"""
    if exact:
        row = fetch_one(downloader, f"SELECT COUNT(*) AS n FROM `{table}`")
        return row["n"] if row else None
    row = fetch_one(
        downloader,
        "SELECT TABLE_ROWS AS n FROM information_schema.tables "
        "WHERE table_schema = %s AND table_name = %s",
        (downloader.config.database, table),
    )
    return row["n"] if row else None


def time_bounds(downloader: MySQLDataDownloader, spec) -> tuple:
    sql = (
        f"SELECT MIN(`{spec.time_column}`) AS mn, MAX(`{spec.time_column}`) AS mx "
        f"FROM `{spec.table_name}`"
    )
    row = fetch_one(downloader, sql)
    return (row["mn"], row["mx"]) if row else (None, None)


def distinct_values(downloader: MySQLDataDownloader, spec, column: str, limit: int | None = None):
    """返回 (总数, 前 limit 个值)。limit=None 时只数总数。"""
    sql = f"SELECT DISTINCT `{column}` AS v FROM `{spec.table_name}`"
    rows = fetch_all(downloader, sql, limit=limit)
    values = [row["v"] for row in rows]
    total = len(values)
    if limit is not None and total == limit:
        cnt = fetch_one(
            downloader,
            f"SELECT COUNT(DISTINCT `{column}`) AS n FROM `{spec.table_name}`",
        )
        total = cnt["n"] if cnt else total
    return total, values


def symbol_row_counts(downloader: MySQLDataDownloader, spec, top: int) -> list[dict]:
    """每个 symbol 的行数 Top N（全表 GROUP BY，大表较慢，仅 --symbol-rows 时执行）。"""
    sql = (
        f"SELECT `{spec.symbol_column}` AS symbol, COUNT(*) AS n "
        f"FROM `{spec.table_name}` "
        f"GROUP BY `{spec.symbol_column}` "
        f"ORDER BY n DESC"
    )
    return fetch_all(downloader, sql, limit=top)


def funding_profile(downloader: MySQLDataDownloader, spec, days: int) -> dict:
    """
    费率表特征：按天统计记录条数，判断粒度。
      - 每天约 3 条 -> 8 小时结算粒度（币安标准，可直接喂回测 funding 结算）
      - 每天 1 条  -> 日粒度（回测 8h 网格会对不齐，需注意）
    """
    col = spec.time_column
    if spec.time_unit == "datetime":
        day_expr = f"DATE(`{col}`)"
        day_div = None
    else:
        day_div = DAY_UNIT_DIV.get(spec.time_unit)
        day_expr = f"FLOOR(`{col}` / {day_div})" if day_div else f"`{col}`"

    rows = fetch_all(
        downloader,
        f"SELECT {day_expr} AS day, COUNT(*) AS cnt "
        f"FROM `{spec.table_name}` "
        f"GROUP BY {day_expr} ORDER BY day DESC",
        limit=days,
    )
    profile = []
    for row in rows:
        if day_div is not None:
            day_str = datetime.fromtimestamp(int(row["day"]) * DAY_SECONDS, tz=timezone.utc).strftime(
                "%Y-%m-%d"
            )
        else:
            day_str = str(row["day"])
        profile.append(f"{day_str}: {row['cnt']} 条")

    # 最近几条原始记录（含费率值），确认列内容
    rate_col = next((c for c in spec.columns if "rate" in c.lower()), None)
    sample_cols = [spec.time_column]
    if spec.symbol_column:
        sample_cols.append(spec.symbol_column)
    if rate_col:
        sample_cols.append(rate_col)
    sample = fetch_all(
        downloader,
        "SELECT " + ", ".join(f"`{c}`" for c in sample_cols) +
        f" FROM `{spec.table_name}` ORDER BY `{spec.time_column}` DESC",
        limit=3,
    )
    return {"profile": profile, "sample": sample, "rate_col": rate_col}


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------

def print_spec_report(downloader: MySQLDataDownloader, spec, args, index: int, total: int) -> None:
    print(f"\n============ [{index}/{total}] {spec.table_name} ============")
    print(f"时间列   : {spec.time_column}  (类型 {spec.time_column_type or '?'} -> 单位 {spec.time_unit})")
    print(f"symbol列 : {spec.symbol_column or '无'}")
    print(f"周期列   : {spec.timeframe_column or '无'}")

    rows_n = table_row_stats(downloader, spec.table_name, args.exact_count)
    print(f"行数     : {rows_n:,}" if rows_n is not None else "行数     : 未知")

    mn, mx = time_bounds(downloader, spec)
    print(f"时间范围 : {fmt_ts(downloader, mn, spec.time_unit)}  ->  {fmt_ts(downloader, mx, spec.time_unit)}")

    if spec.symbol_column:
        total_sym, syms = distinct_values(downloader, spec, spec.symbol_column, limit=args.top_symbols)
        shown = ", ".join(str(s) for s in syms)
        print(f"symbols  : 共 {total_sym} 个")
        if syms:
            print(f"          前 {len(syms)} 个: {shown}")

    if spec.timeframe_column:
        total_tf, tfs = distinct_values(downloader, spec, spec.timeframe_column, limit=20)
        print(f"周期值   : 共 {total_tf} 个 -> {', '.join(str(v) for v in tfs)}")

    if args.symbol_rows and spec.symbol_column:
        print(f"各 symbol 行数 Top {args.top_symbols}（全表聚合，可能较慢）:")
        for row in symbol_row_counts(downloader, spec, args.top_symbols):
            print(f"          {row['symbol']}: {row['n']:,}")

    if "funding" in spec.table_name.lower():
        prof = funding_profile(downloader, spec, args.funding_days)
        print("费率表特征（最近 %d 天每天条数）:" % args.funding_days)
        for line in prof["profile"]:
            print(f"          {line}")
        if prof["profile"]:
            counts = {int(line.split(": ")[1].split(" ")[0]) for line in prof["profile"]}
            if counts == {3}:
                print("          -> 8 小时结算粒度（币安标准），可直接用于回测 funding 结算")
            elif counts == {1}:
                print("          -> 日粒度，回测的 8h 结算网格对不齐，只有每天 1 个结算点有费率")
            else:
                print(f"          -> 每天条数不一致 {sorted(counts)}，请人工核对")
        print("最近记录样例:")
        for row in prof["sample"]:
            print(f"          {row}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="探查 MySQL 行情库数据集（只读）：表清单 / 时间范围 / symbol / 粒度",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pattern", default="binance_usd_margin_%",
                        help="information_schema 表名 LIKE 模式，默认只关注币安 U 本位合约")
    parser.add_argument("--prefix", action="append", default=None,
                        help="只探查这些表前缀（可多次传），如 binance_usd_margin_kline_1m")
    parser.add_argument("--top-symbols", type=int, default=30, help="最多显示多少个 symbol")
    parser.add_argument("--symbol-rows", action="store_true",
                        help="额外统计每个 symbol 的行数（全表 GROUP BY，大表会慢）")
    parser.add_argument("--exact-count", action="store_true",
                        help="用 COUNT(*) 精确统计行数（大表会慢），默认用 information_schema 近似值")
    parser.add_argument("--funding-days", type=int, default=14, help="费率表按天统计最近多少天")

    parser.add_argument("--mysql-config", default=str(Path("config/mysql.config")))
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--charset", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        config = MySQLConfig.from_file_or_env(args.mysql_config)
    except Exception as exc:  # 配置文件缺字段 / 环境变量缺失
        print(f"[错误] MySQL 配置读取失败: {exc}")
        return 1

    for field, value in [
        ("host", args.host), ("port", args.port), ("user", args.user),
        ("password", args.password), ("database", args.database), ("charset", args.charset),
    ]:
        if value is not None:
            setattr(config, field, value)
    config.__post_init__()

    try:
        downloader = MySQLDataDownloader(config)
    except Exception as exc:
        print(f"[错误] 无法创建 MySQL 连接池: {exc}")
        return 1

    try:
        tables = downloader.list_datasets(args.pattern)
    except Exception as exc:
        print(f"[错误] 列数据集失败: {exc}")
        return 1

    if args.prefix:
        wanted = [p for p in args.prefix if p]
        tables = [t for t in tables if any(t == p or t.startswith(p) for p in wanted)]

    if not tables:
        print(f"没有匹配 '{args.pattern}' 的表" +
              (f"（前缀 {args.prefix}）" if args.prefix else ""))
        print("可以先用 --pattern binance_% 或 --pattern % 看库里都有什么表。")
        return 0

    print(f"匹配到 {len(tables)} 张表:")
    for t in tables:
        print(f"  - {t}")

    for index, table in enumerate(tables, 1):
        try:
            spec = downloader.inspect_dataset(table)
        except Exception as exc:
            print(f"\n[跳过] {table}: 探查失败: {exc}")
            continue
        print_spec_report(downloader, spec, args, index, len(tables))

    print("\n探查完成。确认数据可用后，用 scripts/download_data.py 下载（见脚本顶部注释）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
