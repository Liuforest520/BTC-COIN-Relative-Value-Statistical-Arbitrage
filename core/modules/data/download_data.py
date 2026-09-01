from __future__ import annotations

import argparse
import asyncio
import csv
import configparser
import hashlib
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

try:
    from mysql.connector import Error as MySQLError
    from mysql.connector.pooling import MySQLConnectionPool
except Exception:  # pragma: no cover - optional dependency before installation
    MySQLError = Exception
    MySQLConnectionPool = None

from core.modules.logger import logger


TIME_COLUMN_CANDIDATES = [
    "open_time",
    "funding_time",
    "fundingTime",
    "ts",
    "timestamp",
    "event_time",
    "time",
    "date",
    "bar_time",
    "kline_time",
    "created_at",
]
SYMBOL_COLUMN_CANDIDATES = ["symbol", "inst_id", "instrument", "ticker"]
TIMEFRAME_COLUMN_CANDIDATES = ["timeframe", "interval", "bar_interval", "period"]
TEMP_FILE_SUFFIX = ".part"

# 并行下载时保护 manifest.json 的读-改-写（多线程写同一个文件会互相覆盖丢条目）
_MANIFEST_LOCK = threading.Lock()


@dataclass
class MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    charset: str = "utf8mb4"
    pool_name: str = "market_data_pool"
    pool_size: int = 5
    connection_timeout: int = 30
    read_timeout: int = 600
    write_timeout: int = 600
    retry_attempts: int = 20
    retry_delay_seconds: float = 5.0

    def __post_init__(self):
        missing = [
            name
            for name, value in [
                ("host", self.host),
                ("user", self.user),
                ("password", self.password),
                ("database", self.database),
            ]
            if value in {None, ""}
        ]
        if missing:
            raise ValueError(f"missing MySQL config fields: {', '.join(missing)}")

    @classmethod
    def from_env(cls) -> "MySQLConfig":
        host = os.getenv("MYSQL_HOST")
        user = os.getenv("MYSQL_USER")
        password = os.getenv("MYSQL_PASSWORD")
        database = os.getenv("MYSQL_DATABASE")
        port = os.getenv("MYSQL_PORT", "3306")
        charset = os.getenv("MYSQL_CHARSET", "utf8mb4")

        missing = [
            name
            for name, value in [
                ("MYSQL_HOST", host),
                ("MYSQL_USER", user),
                ("MYSQL_PASSWORD", password),
                ("MYSQL_DATABASE", database),
            ]
            if not value
        ]
        if missing:
            raise ValueError(f"missing MySQL env vars: {', '.join(missing)}")

        return cls(
            host=host,
            port=int(port),
            user=user,
            password=password,
            database=database,
            charset=charset,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "MySQLConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"MySQL config file not found: {path}")

        parser = configparser.ConfigParser(interpolation=None)
        parser.read(path, encoding="utf-8")
        if not parser.has_section("mysql"):
            raise ValueError(f"MySQL config file must contain [mysql]: {path}")

        section = parser["mysql"]
        return cls(
            host=section.get("host", fallback="").strip(),
            port=section.getint("port", fallback=3306),
            user=section.get("user", fallback="").strip(),
            password=section.get("password", fallback=""),
            database=section.get("database", fallback="").strip(),
            charset=section.get("charset", fallback="utf8mb4").strip(),
            pool_name=section.get("pool_name", fallback="market_data_pool").strip(),
            pool_size=section.getint("pool_size", fallback=5),
            connection_timeout=section.getint("connection_timeout", fallback=30),
            read_timeout=section.getint("read_timeout", fallback=600),
            write_timeout=section.getint("write_timeout", fallback=600),
            retry_attempts=section.getint("retry_attempts", fallback=20),
            retry_delay_seconds=section.getfloat("retry_delay_seconds", fallback=5.0),
        )

    @classmethod
    def from_file_or_env(cls, path: str | Path = "config/mysql.config") -> "MySQLConfig":
        path = Path(path)
        if path.exists():
            return cls.from_file(path)
        return cls.from_env()


@dataclass
class DatasetSpec:
    dataset: str
    table_name: str
    columns: list[str]
    time_column: str
    time_column_type: str
    time_unit: str
    symbol_column: str | None = None
    timeframe_column: str | None = None
    min_time_value: Any = None
    table_start: datetime | None = None
    table_end_exclusive: datetime | None = None


@dataclass
class DownloadRequest:
    dataset: str
    start: str
    end: str
    output_dir: str | Path = "data"
    symbol: str | None = None
    timeframe: str | None = None
    fetch_size: int = 5000
    window_days: int = 0
    overwrite: bool = False


@dataclass
class DownloadResult:
    dataset: str
    table_name: str
    matched_tables: list[str]
    output_path: Path
    output_paths: list[Path]
    row_count: int
    started_at: datetime
    finished_at: datetime
    start_value: Any
    end_value: Any


class MySQLDataDownloader:
    def __init__(self, config: MySQLConfig):
        _ensure_mysql_dependency()
        self.config = config
        self._pool_serial = 0
        self._pool = self._create_pool()

    def test_connection(self):
        logger.info(
            "connecting to mysql host={} port={} database={} user={}",
            self.config.host,
            self.config.port,
            self.config.database,
            self.config.user,
        )
        with self.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT DATABASE(), USER()")
                row = cursor.fetchone()
            finally:
                cursor.close()
        database = row[0] if row else self.config.database
        user = row[1] if row and len(row) > 1 else self.config.user
        logger.info("mysql connected database={} user={}", database, user)

    def list_datasets(self, pattern: str = "binance_%") -> list[str]:
        sql = (
            "SELECT table_name "
            "FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name LIKE %s "
            "ORDER BY table_name"
        )
        with self.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, (self.config.database, pattern))
                return [row[0] for row in cursor.fetchall()]
            finally:
                cursor.close()

    def inspect_dataset(self, dataset: str) -> DatasetSpec:
        specs = self.inspect_datasets(dataset)
        if not specs:
            raise ValueError(f"dataset {dataset!r} not found in schema {self.config.database}")
        return specs[0]

    def inspect_datasets(self, dataset: str) -> list[DatasetSpec]:
        table_names = self._resolve_table_names(dataset)
        specs = []
        for table_name in table_names:
            columns = self._load_columns(table_name)
            time_column, time_column_type = self._resolve_time_column(columns)
            time_unit = self._resolve_time_unit(table_name, time_column, time_column_type)
            table_range = _table_date_range_from_name(table_name)
            specs.append(
                DatasetSpec(
                    dataset=dataset,
                    table_name=table_name,
                    columns=[item["column_name"] for item in columns],
                    time_column=time_column,
                    time_column_type=time_column_type,
                    time_unit=time_unit,
                    symbol_column=self._first_matching_column(columns, SYMBOL_COLUMN_CANDIDATES),
                    timeframe_column=self._first_matching_column(columns, TIMEFRAME_COLUMN_CANDIDATES),
                    min_time_value=None if table_range else self._table_min_time_value(table_name, time_column),
                    table_start=table_range[0] if table_range else None,
                    table_end_exclusive=table_range[1] if table_range else None,
                )
            )
        return sorted(specs, key=self._dataset_spec_sort_key)

    def inspect_download_datasets(
        self,
        dataset: str,
        start_dt: datetime,
        end_dt_exclusive: datetime,
    ) -> tuple[list[DatasetSpec], int]:
        table_names = self._resolve_table_names(dataset)
        active_table_names = []
        for table_name in table_names:
            table_range = _table_date_range_from_name(table_name)
            if table_range is not None:
                table_start, table_end_exclusive = table_range
                if table_end_exclusive <= start_dt or table_start >= end_dt_exclusive:
                    continue
            active_table_names.append(table_name)

        if not active_table_names:
            return [], len(table_names)

        first_table = active_table_names[0]
        columns = self._load_columns(first_table)
        time_column, time_column_type = self._resolve_time_column(columns)
        time_unit = self._resolve_time_unit(first_table, time_column, time_column_type)
        column_names = [item["column_name"] for item in columns]
        symbol_column = self._first_matching_column(columns, SYMBOL_COLUMN_CANDIDATES)
        timeframe_column = self._first_matching_column(columns, TIMEFRAME_COLUMN_CANDIDATES)

        specs = []
        for table_name in active_table_names:
            table_range = _table_date_range_from_name(table_name)
            specs.append(
                DatasetSpec(
                    dataset=dataset,
                    table_name=table_name,
                    columns=column_names,
                    time_column=time_column,
                    time_column_type=time_column_type,
                    time_unit=time_unit,
                    symbol_column=symbol_column,
                    timeframe_column=timeframe_column,
                    min_time_value=None,
                    table_start=table_range[0] if table_range else None,
                    table_end_exclusive=table_range[1] if table_range else None,
                )
            )
        return sorted(specs, key=self._dataset_spec_sort_key), len(table_names)

    def download(self, request: DownloadRequest) -> DownloadResult:
        self._validate_download_request(request)
        logger.info(
            "starting download dataset={} symbol={} start={} end={} output_dir={} window_days={} fetch_size={}",
            request.dataset,
            request.symbol or "ALL",
            request.start,
            request.end,
            request.output_dir,
            request.window_days,
            request.fetch_size,
        )
        start_dt, end_dt_exclusive = _parse_date_range(request.start, request.end)
        logger.info("scanning matched tables for dataset prefix {}", request.dataset)
        active_specs, matched_table_count = self.inspect_download_datasets(
            request.dataset,
            start_dt,
            end_dt_exclusive,
        )
        if matched_table_count <= 0:
            raise ValueError(f"dataset prefix {request.dataset!r} did not match any table")
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("output directory will be {}", output_dir)

        started_at = datetime.now()
        row_count = 0
        output_paths: list[Path] = []

        logger.info(
            "dataset prefix {} matched {} tables, {} overlap requested range",
            request.dataset,
            matched_table_count,
            len(active_specs),
        )
        logger.info(
            "active tables: {}",
            ", ".join(spec.table_name for spec in active_specs) if active_specs else "none",
        )
        if not active_specs:
            logger.warning(
                "no matched tables overlap requested range {} -> {}",
                start_dt.strftime("%Y-%m-%d"),
                (end_dt_exclusive - timedelta(days=1)).strftime("%Y-%m-%d"),
            )

        if request.symbol:
            return self._download_symbol_dataset(
                request=request,
                active_specs=active_specs,
                matched_table_count=matched_table_count,
                output_dir=output_dir,
                start_dt=start_dt,
                end_dt_exclusive=end_dt_exclusive,
                started_at=started_at,
            )

        logger.info("writing one csv file per source table")
        for spec in active_specs:
            output_path = self._resolve_table_output_path(output_dir, request, spec)
            temp_path = output_path.with_name(output_path.name + TEMP_FILE_SUFFIX)
            output_paths.append(output_path)
            logger.info("table {} output file will be {}", spec.table_name, output_path)

            if output_path.exists() and not request.overwrite:
                raise FileExistsError(f"output file already exists: {output_path}")

            for stale_path in [output_path, temp_path]:
                if request.overwrite and stale_path.exists():
                    logger.info("removing stale file {}", stale_path)
                    stale_path.unlink()

            table_row_count = 0
            last_written_time = None
            last_time_signatures: set[str] = set()
            with temp_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=spec.columns, extrasaction="ignore")
                writer.writeheader()

                for window_start, window_end in self._iter_download_windows(
                    spec=spec,
                    start_dt=start_dt,
                    end_dt_exclusive=end_dt_exclusive,
                    window_days=request.window_days,
                ):
                    logger.info(
                        "downloading {} window {} -> {}",
                        spec.table_name,
                        window_start.strftime("%Y-%m-%d"),
                        window_end.strftime("%Y-%m-%d"),
                    )

                    last_written_time, last_time_signatures, rows_in_window = self._download_window(
                        writer=writer,
                        spec=spec,
                        request=request,
                        window_start=window_start,
                        window_end=window_end,
                        last_written_time=last_written_time,
                        last_time_signatures=last_time_signatures,
                        file_handle=f,
                    )
                    row_count += rows_in_window
                    table_row_count += rows_in_window
                    logger.info(
                        "finished {} window {} -> {} rows={} total_rows={}",
                        spec.table_name,
                        window_start.strftime("%Y-%m-%d"),
                        window_end.strftime("%Y-%m-%d"),
                        rows_in_window,
                        row_count,
                    )

            logger.info("renaming temporary file to final output {}", output_path)
            temp_path.replace(output_path)
            logger.info("saved table={} rows={} file={}", spec.table_name, table_row_count, output_path)

        finished_at = datetime.now()
        time_unit = active_specs[0].time_unit if active_specs else "datetime"
        logger.info(
            "download finished tables={} rows={} output_dir={}",
            len(output_paths),
            row_count,
            output_dir,
        )
        return DownloadResult(
            dataset=request.dataset,
            table_name=request.dataset,
            matched_tables=[spec.table_name for spec in active_specs],
            output_path=output_dir,
            output_paths=output_paths,
            row_count=row_count,
            started_at=started_at,
            finished_at=finished_at,
            start_value=self._time_value_for_query(start_dt, time_unit),
            end_value=self._time_value_for_query(end_dt_exclusive, time_unit),
        )

    def download_symbols(self, request: DownloadRequest, symbols: list[str]) -> list[DownloadResult]:
        self._validate_download_request(request)
        if not symbols:
            return [self.download(request)]

        start_dt, end_dt_exclusive = _parse_date_range(request.start, request.end)
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "starting multi-symbol download dataset={} symbols={} start={} end={} output_dir={} window_days={} fetch_size={}",
            request.dataset,
            ",".join(symbols),
            request.start,
            request.end,
            request.output_dir,
            request.window_days,
            request.fetch_size,
        )
        logger.info("scanning matched tables once for dataset prefix {}", request.dataset)
        active_specs, matched_table_count = self.inspect_download_datasets(
            request.dataset,
            start_dt,
            end_dt_exclusive,
        )
        if matched_table_count <= 0:
            raise ValueError(f"dataset prefix {request.dataset!r} did not match any table")

        logger.info(
            "dataset prefix {} matched {} tables, {} overlap requested range",
            request.dataset,
            matched_table_count,
            len(active_specs),
        )

        results = []
        for symbol in symbols:
            symbol_request = DownloadRequest(
                dataset=request.dataset,
                start=request.start,
                end=request.end,
                output_dir=request.output_dir,
                symbol=symbol,
                timeframe=request.timeframe,
                fetch_size=request.fetch_size,
                window_days=request.window_days,
                overwrite=request.overwrite,
            )
            results.append(
                self._download_symbol_dataset(
                    request=symbol_request,
                    active_specs=active_specs,
                    matched_table_count=matched_table_count,
                    output_dir=output_dir,
                    start_dt=start_dt,
                    end_dt_exclusive=end_dt_exclusive,
                    started_at=datetime.now(),
                )
            )
        return results

    def _validate_download_request(self, request: DownloadRequest):
        if request.fetch_size <= 0:
            raise ValueError("fetch_size must be positive")
        if request.window_days < 0:
            raise ValueError("window_days must be non-negative")

    def _download_symbol_dataset(
        self,
        request: DownloadRequest,
        active_specs: list[DatasetSpec],
        matched_table_count: int,
        output_dir: Path,
        start_dt: datetime,
        end_dt_exclusive: datetime,
        started_at: datetime,
    ) -> DownloadResult:
        symbol = request.symbol
        if not symbol:
            raise ValueError("symbol is required for symbol-organized download")

        label = _dataset_file_label(request.dataset, request.timeframe)
        output_path = self._resolve_symbol_output_path(output_dir, symbol, label)
        temp_path = output_path.with_name(output_path.name + TEMP_FILE_SUFFIX)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = _load_manifest(output_dir)
        entry = _manifest_entry(manifest, symbol, label)
        existing_ranges = _entry_ranges(entry) if output_path.exists() and not request.overwrite else []

        logger.info(
            "symbol-organized output symbol={} data_type={} file={}",
            symbol,
            label,
            output_path,
        )

        if output_path.exists() and not entry and not request.overwrite:
            raise FileExistsError(
                f"output file exists but manifest has no record: {output_path}; "
                "use --overwrite or move the file before downloading"
            )

        if output_path.exists() and request.overwrite:
            logger.info("overwrite enabled, removing existing symbol file {}", output_path)
            output_path.unlink()
            existing_ranges = []

        if temp_path.exists():
            logger.info("removing stale temporary file {}", temp_path)
            temp_path.unlink()

        time_unit = active_specs[0].time_unit if active_specs else "datetime"
        if not active_specs:
            logger.warning(
                "no source tables available for symbol={} data_type={}, nothing to download",
                symbol,
                label,
            )
            finished_at = datetime.now()
            return DownloadResult(
                dataset=request.dataset,
                table_name=request.dataset,
                matched_tables=[],
                output_path=output_path,
                output_paths=[],
                row_count=0,
                started_at=started_at,
                finished_at=finished_at,
                start_value=self._time_value_for_query(start_dt, time_unit),
                end_value=self._time_value_for_query(end_dt_exclusive, time_unit),
            )

        if existing_ranges and _ranges_cover(existing_ranges, start_dt, end_dt_exclusive):
            logger.info(
                "manifest shows local data already covers {} {} {} -> {}, skip download",
                symbol,
                label,
                start_dt.strftime("%Y-%m-%d"),
                (end_dt_exclusive - timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            finished_at = datetime.now()
            return DownloadResult(
                dataset=request.dataset,
                table_name=request.dataset,
                matched_tables=[spec.table_name for spec in active_specs],
                output_path=output_path,
                output_paths=[output_path],
                row_count=int(entry.get("rows") or 0),
                started_at=started_at,
                finished_at=finished_at,
                start_value=self._time_value_for_query(start_dt, time_unit),
                end_value=self._time_value_for_query(end_dt_exclusive, time_unit),
            )

        missing_ranges = _missing_ranges(existing_ranges, start_dt, end_dt_exclusive)
        if not missing_ranges and output_path.exists():
            logger.info("no missing ranges for {} {}, skip download", symbol, label)
            finished_at = datetime.now()
            return DownloadResult(
                dataset=request.dataset,
                table_name=request.dataset,
                matched_tables=[spec.table_name for spec in active_specs],
                output_path=output_path,
                output_paths=[output_path],
                row_count=int(entry.get("rows") or 0),
                started_at=started_at,
                finished_at=finished_at,
                start_value=self._time_value_for_query(start_dt, time_unit),
                end_value=self._time_value_for_query(end_dt_exclusive, time_unit),
            )

        logger.info(
            "missing ranges for {} {}: {}",
            symbol,
            label,
            ", ".join(_format_range_for_log(start, end) for start, end in missing_ranges),
        )

        if output_path.exists() and _ranges_are_after_existing(existing_ranges, missing_ranges):
            logger.info("appending missing data to existing file {}", output_path)
            with output_path.open("a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=active_specs[0].columns, extrasaction="ignore")
                downloaded_rows = self._download_ranges_to_writer(
                    writer=writer,
                    active_specs=active_specs,
                    request=request,
                    ranges=missing_ranges,
                    file_handle=f,
                )
            row_count = int(entry.get("rows") or 0) + downloaded_rows
        else:
            row_count = self._rewrite_symbol_file(
                output_path=output_path,
                temp_path=temp_path,
                active_specs=active_specs,
                request=request,
                existing_ranges=existing_ranges,
                missing_ranges=missing_ranges,
            )

        new_ranges = _merge_ranges([*existing_ranges, *missing_ranges])
        with _MANIFEST_LOCK:
            # 锁内重新读取最新 manifest 再更新本 symbol 条目，避免并发写互相覆盖
            manifest = _load_manifest(output_dir)
            _update_manifest_entry(
                manifest=manifest,
                output_dir=output_dir,
                symbol=symbol,
                label=label,
                output_path=output_path,
                dataset=request.dataset,
                request_start=start_dt,
                request_end_exclusive=end_dt_exclusive,
                ranges=new_ranges,
                rows=row_count,
                spec=active_specs[0] if active_specs else None,
                source_tables=[spec.table_name for spec in active_specs],
            )
            _save_manifest(output_dir, manifest)

        finished_at = datetime.now()
        logger.info(
            "download finished symbol={} data_type={} rows={} file={} manifest={}",
            symbol,
            label,
            row_count,
            output_path,
            _manifest_path(output_dir),
        )
        return DownloadResult(
            dataset=request.dataset,
            table_name=request.dataset,
            matched_tables=[spec.table_name for spec in active_specs],
            output_path=output_path,
            output_paths=[output_path],
            row_count=row_count,
            started_at=started_at,
            finished_at=finished_at,
            start_value=self._time_value_for_query(start_dt, time_unit),
            end_value=self._time_value_for_query(end_dt_exclusive, time_unit),
        )

    def _rewrite_symbol_file(
        self,
        output_path: Path,
        temp_path: Path,
        active_specs: list[DatasetSpec],
        request: DownloadRequest,
        existing_ranges: list[tuple[datetime, datetime]],
        missing_ranges: list[tuple[datetime, datetime]],
    ) -> int:
        row_count = 0
        columns = active_specs[0].columns if active_specs else []
        time_column = active_specs[0].time_column if active_specs else ""
        time_unit = active_specs[0].time_unit if active_specs else "datetime"
        segments = [
            *[("existing", start, end) for start, end in existing_ranges],
            *[("download", start, end) for start, end in missing_ranges],
        ]
        segments.sort(key=lambda item: item[1])

        logger.info("writing rebuilt symbol file {}", temp_path)
        with temp_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for source, start, end in segments:
                if source == "existing":
                    copied = self._copy_existing_rows_in_range(
                        source_path=output_path,
                        writer=writer,
                        time_column=time_column,
                        time_unit=time_unit,
                        start_dt=start,
                        end_dt_exclusive=end,
                    )
                    row_count += copied
                    logger.info(
                        "copied existing rows range {} rows={} total_rows={}",
                        _format_range_for_log(start, end),
                        copied,
                        row_count,
                    )
                    continue

                downloaded = self._download_ranges_to_writer(
                    writer=writer,
                    active_specs=active_specs,
                    request=request,
                    ranges=[(start, end)],
                    file_handle=f,
                )
                row_count += downloaded

        logger.info("renaming rebuilt temporary file to final output {}", output_path)
        temp_path.replace(output_path)
        return row_count

    def _download_ranges_to_writer(
        self,
        writer: csv.DictWriter,
        active_specs: list[DatasetSpec],
        request: DownloadRequest,
        ranges: list[tuple[datetime, datetime]],
        file_handle,
    ) -> int:
        row_count = 0
        last_written_time = None
        last_time_signatures: set[str] = set()
        for range_start, range_end in ranges:
            for spec in active_specs:
                for window_start, window_end in self._iter_download_windows(
                    spec=spec,
                    start_dt=range_start,
                    end_dt_exclusive=range_end,
                    window_days=request.window_days,
                ):
                    logger.info(
                        "downloading {} symbol={} window {} -> {}",
                        spec.table_name,
                        request.symbol or "ALL",
                        window_start.strftime("%Y-%m-%d"),
                        window_end.strftime("%Y-%m-%d"),
                    )
                    last_written_time, last_time_signatures, rows_in_window = self._download_window(
                        writer=writer,
                        spec=spec,
                        request=request,
                        window_start=window_start,
                        window_end=window_end,
                        last_written_time=last_written_time,
                        last_time_signatures=last_time_signatures,
                        file_handle=file_handle,
                    )
                    row_count += rows_in_window
                    logger.info(
                        "finished {} symbol={} window {} -> {} rows={} range_total_rows={}",
                        spec.table_name,
                        request.symbol or "ALL",
                        window_start.strftime("%Y-%m-%d"),
                        window_end.strftime("%Y-%m-%d"),
                        rows_in_window,
                        row_count,
                    )
        return row_count

    def _copy_existing_rows_in_range(
        self,
        source_path: Path,
        writer: csv.DictWriter,
        time_column: str,
        time_unit: str,
        start_dt: datetime,
        end_dt_exclusive: datetime,
    ) -> int:
        copied = 0
        with source_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_dt = self._time_value_to_datetime(row.get(time_column), time_unit)
                if row_dt is None or row_dt < start_dt or row_dt >= end_dt_exclusive:
                    continue
                writer.writerow(row)
                copied += 1
        return copied

    @contextmanager
    def connection(self):
        connection = None
        try:
            connection = self._pool.get_connection()
            connection.autocommit = True
            connection.ping(reconnect=True, attempts=3, delay=2)
            yield connection
        except MySQLError:
            self._rebuild_pool()
            raise
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def _download_window(
        self,
        writer: csv.DictWriter,
        spec: DatasetSpec,
        request: DownloadRequest,
        window_start: datetime,
        window_end: datetime,
        last_written_time: Any,
        last_time_signatures: set[str],
        file_handle,
    ) -> tuple[Any, set[str], int]:
        rows_in_window = 0
        attempts = 0

        while True:
            lower_bound = self._window_resume_lower_bound(window_start, last_written_time, spec.time_unit)
            sql, params = self._build_select_query(
                spec=spec,
                request=request,
                lower_bound=lower_bound,
                upper_bound=window_end,
            )

            try:
                with self.connection() as conn:
                    cursor = conn.cursor(dictionary=True, buffered=False)
                    try:
                        cursor.execute(sql, params)
                        while True:
                            batch = cursor.fetchmany(size=request.fetch_size)
                            if not batch:
                                return last_written_time, last_time_signatures, rows_in_window

                            filtered_batch, last_written_time, last_time_signatures = self._dedupe_resume_rows(
                                rows=batch,
                                time_column=spec.time_column,
                                last_written_time=last_written_time,
                                last_time_signatures=last_time_signatures,
                            )
                            if not filtered_batch:
                                continue

                            writer.writerows(filtered_batch)
                            file_handle.flush()
                            rows_in_window += len(filtered_batch)
                    finally:
                        cursor.close()
            except MySQLError as exc:
                attempts += 1
                if attempts > self.config.retry_attempts:
                    raise RuntimeError(
                        f"download failed after {self.config.retry_attempts} retries for {spec.table_name}"
                    ) from exc
                logger.warning(
                    "download interrupted for {} retry={}/{} reason={}",
                    spec.table_name,
                    attempts,
                    self.config.retry_attempts,
                    exc,
                )
                time.sleep(self.config.retry_delay_seconds)
                self._rebuild_pool()

    def _active_specs_for_range(
        self,
        specs: list[DatasetSpec],
        start_dt: datetime,
        end_dt_exclusive: datetime,
    ) -> list[DatasetSpec]:
        active_specs = []
        for spec in specs:
            if spec.table_start is None or spec.table_end_exclusive is None:
                active_specs.append(spec)
                continue
            if spec.table_end_exclusive <= start_dt or spec.table_start >= end_dt_exclusive:
                continue
            active_specs.append(spec)
        return active_specs

    def _iter_download_windows(
        self,
        spec: DatasetSpec,
        start_dt: datetime,
        end_dt_exclusive: datetime,
        window_days: int,
    ) -> Iterator[tuple[datetime, datetime]]:
        table_start = spec.table_start or start_dt
        table_end_exclusive = spec.table_end_exclusive or end_dt_exclusive
        window_start = max(start_dt, table_start)
        clipped_end = min(end_dt_exclusive, table_end_exclusive)

        if window_start >= clipped_end:
            return

        if window_days == 0:
            yield window_start, clipped_end
            return

        while window_start < clipped_end:
            window_end = min(window_start + timedelta(days=window_days), clipped_end)
            yield window_start, window_end
            window_start = window_end

    def _build_select_query(self, spec: DatasetSpec, request: DownloadRequest, lower_bound: datetime, upper_bound: datetime):
        conditions = [f"`{spec.time_column}` >= %s", f"`{spec.time_column}` < %s"]
        params: list[Any] = [
            self._time_value_for_query(lower_bound, spec.time_unit),
            self._time_value_for_query(upper_bound, spec.time_unit),
        ]

        if request.symbol is not None:
            if spec.symbol_column is None:
                raise ValueError(f"dataset {spec.table_name} has no symbol column, cannot filter by symbol")
            conditions.append(f"`{spec.symbol_column}` = %s")
            params.append(request.symbol)

        if request.timeframe is not None:
            if spec.timeframe_column is None:
                raise ValueError(f"dataset {spec.table_name} has no timeframe column, cannot filter by timeframe")
            conditions.append(f"`{spec.timeframe_column}` = %s")
            params.append(request.timeframe)

        selected_columns = ", ".join(f"`{column}`" for column in spec.columns)
        where_clause = " AND ".join(conditions)
        sql = (
            f"SELECT {selected_columns} "
            f"FROM `{spec.table_name}` "
            f"WHERE {where_clause} "
            f"ORDER BY `{spec.time_column}` ASC"
        )
        return sql, tuple(params)

    def _resolve_output_path(self, output_dir: Path, request: DownloadRequest) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [request.symbol or "ALL", request.dataset]
        if request.timeframe:
            parts.append(request.timeframe)
        parts.extend([request.start, request.end, stamp])
        safe_name = "_".join(_sanitize_filename_part(part) for part in parts if part)
        return output_dir / f"{safe_name}.csv"

    def _resolve_table_output_path(self, output_dir: Path, request: DownloadRequest, spec: DatasetSpec) -> Path:
        parts = [spec.table_name]
        if request.symbol:
            parts.append(request.symbol)
        if request.timeframe:
            parts.append(request.timeframe)
        safe_name = "_".join(_sanitize_filename_part(part) for part in parts if part)
        return output_dir / f"{safe_name}.csv"

    def _resolve_symbol_output_path(self, output_dir: Path, symbol: str, label: str) -> Path:
        safe_symbol = _sanitize_filename_part(symbol)
        safe_label = _sanitize_filename_part(label)
        return output_dir / safe_symbol / f"{safe_symbol}-{safe_label}.csv"

    def _matching_output_files(self, output_dir: Path, request: DownloadRequest) -> list[Path]:
        fragments = [request.dataset, request.start, request.end]
        if request.symbol:
            fragments.insert(0, request.symbol)
        if request.timeframe:
            fragments.append(request.timeframe)

        matches = []
        for path in output_dir.glob("*.csv"):
            name = path.name.lower()
            if all(fragment.lower() in name for fragment in fragments):
                matches.append(path)
        matches.sort()
        return matches

    def _resolve_table_names(self, dataset: str) -> list[str]:
        exact_sql = "SHOW TABLES LIKE %s"
        with self.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(exact_sql, (_mysql_like_escape(dataset),))
                exact = [row[0] for row in cursor.fetchall()]
            finally:
                cursor.close()
        if exact:
            return exact

        prefix_sql = "SHOW TABLES LIKE %s"
        with self.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(prefix_sql, (_mysql_like_escape(dataset) + "%",))
                prefix_matches = [row[0] for row in cursor.fetchall()]
            finally:
                cursor.close()
        if prefix_matches:
            return prefix_matches
        raise ValueError(f"dataset {dataset!r} not found in schema {self.config.database}")

    def _load_columns(self, table_name: str) -> list[dict[str, Any]]:
        sql = f"SHOW COLUMNS FROM {_quote_identifier(table_name)}"
        with self.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql)
                columns = cursor.fetchall()
            finally:
                cursor.close()
        if not columns:
            raise ValueError(f"table has no columns: {table_name}")
        normalized_columns = []
        for column in columns:
            if isinstance(column, dict):
                column_name = column.get("column_name") or column.get("COLUMN_NAME") or column.get("Field")
                data_type = column.get("data_type") or column.get("DATA_TYPE") or column.get("Type")
            else:
                column_name = column[0]
                data_type = str(column[1]).split("(", 1)[0]
            if column_name is None or data_type is None:
                keys = sorted(column.keys()) if isinstance(column, dict) else list(range(len(column)))
                raise KeyError(f"unexpected column metadata keys for {table_name}: {keys}")
            normalized_columns.append(
                {
                    "column_name": column_name,
                    "data_type": data_type,
                }
            )
        return normalized_columns

    def _resolve_time_column(self, columns: list[dict[str, Any]]) -> tuple[str, str]:
        for candidate in TIME_COLUMN_CANDIDATES:
            for column in columns:
                if column["column_name"].lower() == candidate.lower():
                    return column["column_name"], str(column["data_type"]).lower()
        available = ", ".join(item["column_name"] for item in columns)
        raise ValueError(f"could not infer time column, available columns: {available}")

    def _resolve_time_unit(self, table_name: str, time_column: str, time_column_type: str) -> str:
        normalized_type = time_column_type.lower().split("(", 1)[0].strip()
        normalized_column = time_column.lower()

        if normalized_type in {"datetime", "timestamp", "date"}:
            return "datetime"
        if normalized_column in {"open_time", "close_time", "event_time", "bar_time", "kline_time"}:
            return "ms"
        if normalized_column in {"fundingtime", "funding_time"}:
            return "ms"

        sql = (
            f"SELECT `{time_column}` "
            f"FROM `{table_name}` "
            f"WHERE `{time_column}` IS NOT NULL "
            f"ORDER BY `{time_column}` ASC "
            f"LIMIT 1"
        )
        with self.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql)
                row = cursor.fetchone()
            finally:
                cursor.close()

        if not row:
            logger.warning("table {} is empty, fallback time unit to milliseconds", table_name)
            return "ms"

        value = row[0]
        if not isinstance(value, (int, float)):
            return "datetime"
        abs_value = abs(int(value))
        if abs_value >= 10**17:
            return "ns"
        if abs_value >= 10**14:
            return "us"
        if abs_value >= 10**11:
            return "ms"
        return "s"

    def _table_min_time_value(self, table_name: str, time_column: str):
        sql = (
            f"SELECT MIN(`{time_column}`) "
            f"FROM `{table_name}` "
            f"WHERE `{time_column}` IS NOT NULL"
        )
        with self.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql)
                row = cursor.fetchone()
            finally:
                cursor.close()
        if not row:
            return None
        return row[0]

    def _time_value_for_query(self, dt: datetime, time_unit: str):
        if time_unit == "datetime":
            return dt.strftime("%Y-%m-%d %H:%M:%S")

        epoch_seconds = dt.timestamp()
        if time_unit == "s":
            return int(epoch_seconds)
        if time_unit == "ms":
            return int(epoch_seconds * 1000)
        if time_unit == "us":
            return int(epoch_seconds * 1_000_000)
        if time_unit == "ns":
            return int(epoch_seconds * 1_000_000_000)
        raise ValueError(f"unsupported time unit: {time_unit}")

    def _window_resume_lower_bound(self, window_start: datetime, last_written_time: Any, time_unit: str) -> datetime:
        if last_written_time is None:
            return window_start
        last_dt = self._time_value_to_datetime(last_written_time, time_unit)
        if last_dt is None:
            return window_start
        return max(window_start, last_dt)

    def _time_value_to_datetime(self, value: Any, time_unit: str) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.replace(microsecond=0)
        if isinstance(value, str):
            parsed = _parse_datetime_text(value)
            if parsed is not None:
                return parsed.replace(microsecond=0)
            try:
                numeric = float(value)
            except ValueError:
                return None
            return self._numeric_time_value_to_datetime(numeric, time_unit)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return self._numeric_time_value_to_datetime(numeric, time_unit)

    def _numeric_time_value_to_datetime(self, numeric: float, time_unit: str) -> datetime | None:
        if time_unit == "s":
            return datetime.utcfromtimestamp(numeric)
        if time_unit == "ms":
            return datetime.utcfromtimestamp(numeric / 1000.0)
        if time_unit == "us":
            return datetime.utcfromtimestamp(numeric / 1_000_000.0)
        if time_unit == "ns":
            return datetime.utcfromtimestamp(numeric / 1_000_000_000.0)
        return None

    def _dedupe_resume_rows(
        self,
        rows: list[dict[str, Any]],
        time_column: str,
        last_written_time: Any,
        last_time_signatures: set[str],
    ) -> tuple[list[dict[str, Any]], Any, set[str]]:
        filtered_rows = []
        signatures = set(last_time_signatures)
        current_time = last_written_time

        for row in rows:
            row_time = row.get(time_column)
            row_signature = _row_signature(row)
            if current_time is not None and row_time == current_time and row_signature in signatures:
                continue

            filtered_rows.append(row)
            if current_time is None or row_time != current_time:
                current_time = row_time
                signatures = {row_signature}
            else:
                signatures.add(row_signature)

        return filtered_rows, current_time, signatures

    def _first_matching_column(self, columns: list[dict[str, Any]], candidates: list[str]) -> str | None:
        lowered = {item["column_name"].lower(): item["column_name"] for item in columns}
        for candidate in candidates:
            if candidate.lower() in lowered:
                return lowered[candidate.lower()]
        return None

    def _dataset_spec_sort_key(self, spec: DatasetSpec):
        if spec.table_start is not None:
            return spec.table_start, spec.table_name
        dt = self._time_value_to_datetime(spec.min_time_value, spec.time_unit)
        if dt is None:
            return datetime.max, spec.table_name
        return dt, spec.table_name

    def _merged_columns(self, specs: list[DatasetSpec]) -> list[str]:
        merged = []
        for spec in specs:
            for column in spec.columns:
                if column not in merged:
                    merged.append(column)
        return merged

    def _create_pool(self) -> MySQLConnectionPool:
        self._pool_serial += 1
        return MySQLConnectionPool(
            pool_name=f"{self.config.pool_name}_{self._pool_serial}",
            pool_size=self.config.pool_size,
            pool_reset_session=False,
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            charset=self.config.charset,
            use_pure=True,
            connection_timeout=self.config.connection_timeout,
            read_timeout=self.config.read_timeout,
            write_timeout=self.config.write_timeout,
            autocommit=True,
        )

    def _rebuild_pool(self):
        logger.warning("rebuilding MySQL connection pool {}", self.config.pool_name)
        self._pool = self._create_pool()


def download_data(
    dataset: str,
    start: str,
    end: str,
    *,
    output_dir: str | Path = "data",
    symbol: str | None = None,
    timeframe: str | None = None,
    fetch_size: int = 5000,
    window_days: int = 0,
    overwrite: bool = False,
    mysql_config: MySQLConfig | None = None,
    mysql_config_path: str | Path = "config/mysql.config",
) -> DownloadResult:
    config = mysql_config or MySQLConfig.from_file_or_env(mysql_config_path)
    downloader = MySQLDataDownloader(config)
    downloader.test_connection()
    request = DownloadRequest(
        dataset=dataset,
        start=start,
        end=end,
        output_dir=output_dir,
        symbol=symbol,
        timeframe=timeframe,
        fetch_size=fetch_size,
        window_days=window_days,
        overwrite=overwrite,
    )
    return downloader.download(request)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Download market data from MySQL.")
    parser.add_argument("--dataset", required=False, help="table name or category prefix, e.g. binance_spot_kline")
    parser.add_argument("--start", required=False, help="inclusive start date, format YYYYMMDD")
    parser.add_argument("--end", required=False, help="inclusive end date, format YYYYMMDD")
    parser.add_argument("--symbol", default=None, help="optional symbol filter, e.g. BTCUSDT")
    parser.add_argument("--symbols", default=None, help="optional comma-separated symbols, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--timeframe", default=None, help="optional timeframe filter, e.g. 1m")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--fetch-size", type=int, default=5000)
    parser.add_argument(
        "--window-days",
        type=int,
        default=0,
        help="optional sub-window size inside each matched table; 0 means download by table range",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="number of symbols to download in parallel (only takes effect when multiple symbols are given)",
    )
    parser.add_argument("--list-datasets", action="store_true", help="list available tables and exit")
    parser.add_argument("--mysql-config", default=os.getenv("MYSQL_CONFIG", "config/mysql.config"))
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--charset", default=None)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--retry-attempts", type=int, default=None)
    parser.add_argument("--retry-delay-seconds", type=float, default=None)
    parser.add_argument("--read-timeout", type=int, default=None)
    parser.add_argument("--write-timeout", type=int, default=None)
    parser.add_argument("--connection-timeout", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    config = MySQLConfig.from_file_or_env(args.mysql_config)
    overrides = {
        "host": args.host,
        "port": args.port,
        "user": args.user,
        "password": args.password,
        "database": args.database,
        "charset": args.charset,
        "pool_size": args.pool_size,
        "retry_attempts": args.retry_attempts,
        "retry_delay_seconds": args.retry_delay_seconds,
        "read_timeout": args.read_timeout,
        "write_timeout": args.write_timeout,
        "connection_timeout": args.connection_timeout,
    }
    for field_name, value in overrides.items():
        if value is not None:
            setattr(config, field_name, value)
    config.__post_init__()
    downloader = MySQLDataDownloader(config)
    downloader.test_connection()

    if args.list_datasets:
        datasets = downloader.list_datasets()
        for dataset in datasets:
            logger.info("dataset: {}", dataset)
        return

    missing = [name for name in ["dataset", "start", "end"] if getattr(args, name) in {None, ""}]
    if missing:
        raise ValueError(f"missing required args: {', '.join(missing)}")

    symbols = _parse_symbols(args.symbols)
    if args.symbol:
        symbols.insert(0, args.symbol)
    symbols = list(dict.fromkeys(symbols))

    if len(symbols) > 1:
        logger.info("downloading {} symbols: {}", len(symbols), ", ".join(symbols))
        request = DownloadRequest(
            dataset=args.dataset,
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
            symbol=None,
            timeframe=args.timeframe,
            fetch_size=args.fetch_size,
            window_days=args.window_days,
            overwrite=args.overwrite,
        )
        concurrency = max(1, int(args.concurrency or 1))
        if concurrency > 1:
            _download_symbols_parallel(downloader, request, symbols, concurrency)
            return

        results = downloader.download_symbols(request, symbols)
        total_rows = sum(result.row_count for result in results)
        output_paths = [path for result in results for path in result.output_paths]
        logger.info(
            "saved dataset={} symbols={} rows={} files={} output_dir={}",
            args.dataset,
            len(symbols),
            total_rows,
            len(output_paths),
            args.output_dir,
        )
        return

    result = downloader.download(
        DownloadRequest(
            dataset=args.dataset,
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
            symbol=symbols[0] if symbols else None,
            timeframe=args.timeframe,
            fetch_size=args.fetch_size,
            window_days=args.window_days,
            overwrite=args.overwrite,
        )
    )
    logger.info(
        "saved dataset={} tables={} rows={} output_dir={}",
        result.dataset,
        len(result.output_paths),
        result.row_count,
        result.output_path,
    )


def _download_symbols_parallel(
    downloader: MySQLDataDownloader,
    request: DownloadRequest,
    symbols: list[str],
    concurrency: int,
) -> None:
    """多 symbol 并行下载：每个 symbol 一个工作线程 + 一个独立连接池。

    manifest.json 的写入由 _MANIFEST_LOCK 串行化（见 _download_symbol_dataset），
    保证并发下各 symbol 的条目都不会丢失。
    """
    start_dt, end_dt_exclusive = _parse_date_range(request.start, request.end)
    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("scanning matched tables once for dataset prefix {}", request.dataset)
    active_specs, matched_table_count = downloader.inspect_download_datasets(
        request.dataset,
        start_dt,
        end_dt_exclusive,
    )
    if matched_table_count <= 0:
        raise ValueError(f"dataset prefix {request.dataset!r} did not match any table")

    started = time.monotonic()
    results = asyncio.run(
        _gather_symbol_downloads(
            base_config=downloader.config,
            request=request,
            symbols=symbols,
            active_specs=active_specs,
            matched_table_count=matched_table_count,
            output_dir=output_dir,
            start_dt=start_dt,
            end_dt_exclusive=end_dt_exclusive,
            concurrency=concurrency,
        )
    )

    ok = failed = 0
    total_rows = 0
    for symbol, result in zip(symbols, results):
        if isinstance(result, Exception):
            failed += 1
            logger.error("symbol={} download failed: {}", symbol, result)
        else:
            ok += 1
            total_rows += result.row_count
            logger.info(
                "saved symbol={} rows={} file={}",
                symbol,
                result.row_count,
                result.output_path,
            )
    logger.info(
        "parallel download finished ok={} failed={} rows={} elapsed={:.1f}s",
        ok,
        failed,
        total_rows,
        time.monotonic() - started,
    )


async def _gather_symbol_downloads(
    base_config: MySQLConfig,
    request: DownloadRequest,
    symbols: list[str],
    active_specs: list[DatasetSpec],
    matched_table_count: int,
    output_dir: Path,
    start_dt: datetime,
    end_dt_exclusive: datetime,
    concurrency: int,
):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="dl") as pool:
        futures = [
            loop.run_in_executor(
                pool,
                _symbol_download_worker,
                base_config,
                request,
                symbol,
                active_specs,
                matched_table_count,
                output_dir,
                start_dt,
                end_dt_exclusive,
            )
            for symbol in symbols
        ]
        return await asyncio.gather(*futures, return_exceptions=True)


def _symbol_download_worker(
    base_config: MySQLConfig,
    request: DownloadRequest,
    symbol: str,
    active_specs: list[DatasetSpec],
    matched_table_count: int,
    output_dir: Path,
    start_dt: datetime,
    end_dt_exclusive: datetime,
) -> DownloadResult:
    """在工作线程中下载单个 symbol；使用独立连接池（池名唯一，避免复用全局同名池）。"""
    config = replace(
        base_config,
        pool_name=f"{base_config.pool_name}_{uuid.uuid4().hex[:8]}",
        pool_size=1,  # 单个任务串行取窗口连接，一个连接足够
    )
    downloader = MySQLDataDownloader(config)
    symbol_request = replace(request, symbol=symbol)
    return downloader._download_symbol_dataset(
        request=symbol_request,
        active_specs=active_specs,
        matched_table_count=matched_table_count,
        output_dir=output_dir,
        start_dt=start_dt,
        end_dt_exclusive=end_dt_exclusive,
        started_at=datetime.now(),
    )


def _parse_date_range(start: str, end: str) -> tuple[datetime, datetime]:
    start_dt = _parse_yyyymmdd(start)
    end_dt = _parse_yyyymmdd(end)
    if end_dt < start_dt:
        raise ValueError("end date must be greater than or equal to start date")
    return start_dt, end_dt + timedelta(days=1)


def _parse_symbols(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _table_date_range_from_name(table_name: str) -> tuple[datetime, datetime] | None:
    dated_range = re.search(r"_(\d{8})_(\d{8})$", table_name)
    if dated_range:
        start = _parse_yyyymmdd(dated_range.group(1))
        end_exclusive = _parse_yyyymmdd(dated_range.group(2)) + timedelta(days=1)
        return start, end_exclusive

    yearly = re.search(r"_(\d{4})$", table_name)
    if yearly:
        year = int(yearly.group(1))
        return datetime(year, 1, 1), datetime(year + 1, 1, 1)

    return None


def _mysql_like_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _quote_identifier(value: str) -> str:
    return "`" + value.replace("`", "``") + "`"


def _dataset_file_label(dataset: str, timeframe: str | None = None) -> str:
    if timeframe:
        return timeframe

    kline_match = re.search(r"_kline_([^_]+)(?:_|$)", dataset)
    if kline_match:
        return kline_match.group(1)

    if "funding_rate_daily" in dataset:
        return "funding_rate_daily"
    if "funding" in dataset:
        return "funding"

    return dataset


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "manifest.json"


def _load_manifest(output_dir: Path) -> dict[str, Any]:
    path = _manifest_path(output_dir)
    if not path.exists():
        return {"version": 1, "updated_at": None, "data": {}}
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.setdefault("version", 1)
    manifest.setdefault("data", {})
    return manifest


def _save_manifest(output_dir: Path, manifest: dict[str, Any]):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    path = _manifest_path(output_dir)
    temp_path = path.with_name(path.name + TEMP_FILE_SUFFIX)
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    temp_path.replace(path)


def _manifest_entry(manifest: dict[str, Any], symbol: str, label: str) -> dict[str, Any]:
    return manifest.get("data", {}).get(symbol, {}).get(label, {})


def _entry_ranges(entry: dict[str, Any]) -> list[tuple[datetime, datetime]]:
    raw_ranges = entry.get("ranges") or []
    ranges = []
    for raw_range in raw_ranges:
        start = _parse_manifest_date(raw_range.get("start"))
        end = _parse_manifest_date(raw_range.get("end"))
        if start is None or end is None:
            continue
        ranges.append((start, end + timedelta(days=1)))

    if ranges:
        return _merge_ranges(ranges)

    start = _parse_manifest_date(entry.get("start"))
    end = _parse_manifest_date(entry.get("end"))
    if start is None or end is None:
        return []
    return [(start, end + timedelta(days=1))]


def _parse_manifest_date(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.replace(hour=0, minute=0, second=0, microsecond=0)
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    parsed = _parse_datetime_text(str(value))
    if parsed is None:
        return None
    return parsed.replace(hour=0, minute=0, second=0, microsecond=0)


def _merge_ranges(ranges: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    valid_ranges = sorted((start, end) for start, end in ranges if start < end)
    if not valid_ranges:
        return []

    merged = [valid_ranges[0]]
    for start, end in valid_ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _ranges_cover(ranges: list[tuple[datetime, datetime]], start_dt: datetime, end_dt_exclusive: datetime) -> bool:
    current = start_dt
    for start, end in _merge_ranges(ranges):
        if end <= current:
            continue
        if start > current:
            return False
        current = max(current, end)
        if current >= end_dt_exclusive:
            return True
    return current >= end_dt_exclusive


def _missing_ranges(
    existing_ranges: list[tuple[datetime, datetime]],
    start_dt: datetime,
    end_dt_exclusive: datetime,
) -> list[tuple[datetime, datetime]]:
    missing = []
    current = start_dt
    for start, end in _merge_ranges(existing_ranges):
        if end <= current:
            continue
        if start > current:
            missing.append((current, min(start, end_dt_exclusive)))
        current = max(current, end)
        if current >= end_dt_exclusive:
            break
    if current < end_dt_exclusive:
        missing.append((current, end_dt_exclusive))
    return [(start, end) for start, end in missing if start < end]


def _ranges_are_after_existing(
    existing_ranges: list[tuple[datetime, datetime]],
    missing_ranges: list[tuple[datetime, datetime]],
) -> bool:
    if not existing_ranges or not missing_ranges:
        return False
    existing_end = max(end for _, end in existing_ranges)
    return all(start >= existing_end for start, _ in missing_ranges)


def _update_manifest_entry(
    manifest: dict[str, Any],
    output_dir: Path,
    symbol: str,
    label: str,
    output_path: Path,
    dataset: str,
    request_start: datetime,
    request_end_exclusive: datetime,
    ranges: list[tuple[datetime, datetime]],
    rows: int,
    spec: DatasetSpec | None,
    source_tables: list[str],
):
    merged_ranges = _merge_ranges(ranges)
    relative_path = output_path
    try:
        relative_path = output_path.relative_to(output_dir)
    except ValueError:
        pass

    symbol_section = manifest.setdefault("data", {}).setdefault(symbol, {})
    entry = {
        "symbol": symbol,
        "dataset": dataset,
        "data_type": label,
        "path": str(relative_path).replace("\\", "/"),
        "start": _format_manifest_start(merged_ranges[0][0]) if merged_ranges else None,
        "end": _format_manifest_end(merged_ranges[-1][1]) if merged_ranges else None,
        "ranges": [
            {
                "start": _format_manifest_start(start),
                "end": _format_manifest_end(end),
            }
            for start, end in merged_ranges
        ],
        "rows": rows,
        "time_column": spec.time_column if spec else None,
        "time_unit": spec.time_unit if spec else None,
        "columns": spec.columns if spec else [],
        "source_tables": source_tables,
        "last_request_start": _format_manifest_start(request_start),
        "last_request_end": _format_manifest_end(request_end_exclusive),
        "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    symbol_section[label] = entry


def _format_manifest_start(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _format_manifest_end(end_dt_exclusive: datetime) -> str:
    return (end_dt_exclusive - timedelta(days=1)).strftime("%Y-%m-%d")


def _format_range_for_log(start: datetime, end_dt_exclusive: datetime) -> str:
    return f"{_format_manifest_start(start)} -> {_format_manifest_end(end_dt_exclusive)}"


def _parse_yyyymmdd(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"invalid date {value!r}, expected YYYYMMDD") from exc


def _parse_datetime_text(value: str) -> datetime | None:
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _sanitize_filename_part(value: str) -> str:
    chars = [char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value)]
    return "".join(chars).strip("_") or "value"


def _row_signature(row: dict[str, Any]) -> str:
    encoded = repr(sorted(row.items(), key=lambda item: item[0])).encode("utf-8", errors="replace")
    return hashlib.sha1(encoded).hexdigest()


def _ensure_mysql_dependency():
    if MySQLConnectionPool is None:
        raise ModuleNotFoundError(
            "mysql-connector-python is not installed. Run `pip install -r requirements.txt` first."
        )


if __name__ == "__main__":
    main()
