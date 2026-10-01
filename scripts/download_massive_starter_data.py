"""Download all bulk-downloadable datasets available in Massive Stocks Starter.

The script keeps each dataset in its own directory under data/massive/stocks:

    flat_files/day_aggregates/YYYY/MM/YYYY-MM-DD.csv.gz
    flat_files/minute_aggregates/YYYY/MM/YYYY-MM-DD.csv.gz
    reference/tickers/tickers.json
    reference/ticker_types/ticker_types.json
    reference/exchanges/exchanges.json
    corporate_actions/splits/splits.json
    corporate_actions/dividends/dividends.json
    corporate_actions/ipos/ipos.json
    market_operations/holidays/holidays.json

The $29 Stocks Starter plan also exposes Second Aggregates and Technical
Indicators, but those are request-shaped products (ticker, date range, and/or
indicator parameters), not one finite historical bulk archive. They are
therefore not silently expanded into an enormous, ambiguous download. Use the
Flat Files and paginated API datasets here for the complete archiveable inputs,
and use --include-snapshot for a current full-market snapshot. Trades, Quotes,
and Financials & Ratios are not included in this plan.
"""

from __future__ import annotations

import argparse
import configparser
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

try:
    import boto3
    from botocore.client import Config as BotoConfig
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError as exc:  # pragma: no cover
    boto3 = None
    BotoConfig = None
    BotoCoreError = Exception
    ClientError = Exception
    _BOTO_IMPORT_ERROR = exc
else:
    _BOTO_IMPORT_ERROR = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "massive_key.config"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "massive" / "stocks"
MASSIVE_ENDPOINT = "https://api.massive.com"
FILES_ENDPOINT = "https://files.massive.com"
FILES_BUCKET = "flatfiles"
LOGGER = logging.getLogger("download_massive_starter_data")

FLAT_DATASETS = {
    "day_aggregates": "us_stocks_sip/day_aggs_v1",
    "minute_aggregates": "us_stocks_sip/minute_aggs_v1",
}
API_DATASETS = {
    "tickers": ("/v3/reference/tickers", "reference/tickers", 1000),
    "ticker_types": ("/v3/reference/tickers/types", "reference/ticker_types", 1000),
    "exchanges": ("/v3/reference/exchanges", "reference/exchanges", 1000),
    "splits": ("/stocks/v1/splits", "corporate_actions/splits", 5000),
    "dividends": ("/stocks/v1/dividends", "corporate_actions/dividends", 5000),
    "ipos": ("/vX/reference/ipos", "corporate_actions/ipos", 1000),
    "holidays": ("/v1/marketstatus/upcoming", "market_operations/holidays", 1000),
    "market_status": ("/v1/marketstatus/now", "market_operations/market_status", 1000),
    "conditions": ("/v3/reference/conditions", "reference/conditions", 1000),
}


def _load_credentials(path: Path) -> tuple[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Massive credentials file not found: {path}")
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(path, encoding="utf-8")
    if not parser.has_section("massive"):
        raise ValueError(f"credentials file must contain [massive]: {path}")
    section = parser["massive"]
    access = os.environ.get("MASSIVE_ACCESS_KEY", section.get("aws_access_key_id", "")).strip()
    secret = os.environ.get("MASSIVE_API_KEY", section.get("aws_secret_access_key", "")).strip()
    if not access or not secret:
        raise ValueError("Massive credentials require aws_access_key_id and aws_secret_access_key")
    return access, secret


def _with_api_key(url: str, key: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["apiKey"] = key
    return urlunparse(parsed._replace(query=urlencode(query)))


def _request_json(url: str, key: str, timeout: float, retries: int = 4) -> Any:
    target = _with_api_key(url, key)
    for attempt in range(retries + 1):
        request = Request(target, headers={"Accept": "application/json", "User-Agent": "stat-arb-massive-downloader/1.0"})
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            if not isinstance(payload, (dict, list)):
                raise RuntimeError("Massive API returned an unsupported JSON response")
            if isinstance(payload, dict) and payload.get("status") not in {None, "OK"}:
                raise RuntimeError(f"Massive API returned status {payload.get('status')!r}")
            return payload
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(min(30.0, 2.0 ** attempt))
                continue
            raise RuntimeError(f"Massive API HTTP {exc.code}: {body[:1000]}") from exc
        except (URLError, TimeoutError) as exc:
            if attempt < retries:
                time.sleep(min(30.0, 2.0 ** attempt))
                continue
            raise RuntimeError(f"Massive API request failed: {exc}") from exc
    raise RuntimeError("Massive API request failed after retries")


def _paginate(endpoint: str, key: str, params: dict[str, Any], timeout: float) -> list[dict[str, Any]]:
    query = urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{MASSIVE_ENDPOINT}{endpoint}?{query}" if query else f"{MASSIVE_ENDPOINT}{endpoint}"
    records: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    pages = 0
    while url:
        if url in seen:
            raise RuntimeError(f"repeated next_url while downloading {endpoint}")
        seen.add(url)
        pages += 1
        payload = _request_json(url, key, timeout)
        if isinstance(payload, list):
            rows = payload
        else:
            rows = payload.get("results")
            if rows is None:
                rows = payload.get("response")
            if rows is None:
                rows = payload.get("tickers")
            if isinstance(rows, dict):
                rows = [rows]
            if not isinstance(rows, list):
                rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            identity = str(row.get("id") or row.get("ticker") or row.get("code") or json.dumps(row, sort_keys=True))
            records[identity] = row
        url = str(payload.get("next_url") or "") if isinstance(payload, dict) else ""
        LOGGER.info("%s page=%d rows=%d total=%d", endpoint, pages, len(rows), len(records))
    return list(records.values())


def _write_json(output: Path, source: str, records: Any, extra: dict[str, Any] | None = None) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records) if isinstance(records, (list, dict)) else None,
        "results": records,
    }
    if extra:
        payload.update(extra)
    temp = output / f"{uuid.uuid4().hex}.tmp"
    target = output / f"{output.name}.json"
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(target)
    LOGGER.info("saved %s", target)
    return target


def _s3_client(access: str, secret: str):
    if boto3 is None:
        raise RuntimeError("boto3 is required for flat-file downloads") from _BOTO_IMPORT_ERROR
    return boto3.client(
        "s3", endpoint_url=FILES_ENDPOINT, region_name="us-east-1",
        aws_access_key_id=access, aws_secret_access_key=secret,
        config=BotoConfig(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}, connect_timeout=30, read_timeout=900),
    )


def _date_range(start: date, end: date) -> Iterable[date]:
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def _download_flat_one(dataset: str, prefix: str, day: date, output: Path, access: str, secret: str, overwrite: bool) -> str:
    target_dir = output / "flat_files" / dataset / f"{day:%Y}" / f"{day:%m}"
    target = target_dir / f"{day:%Y-%m-%d}.csv.gz"
    if target.exists() and target.stat().st_size > 0 and not overwrite:
        return "skipped"
    key = f"{prefix}/{day:%Y}/{day:%m}/{day:%Y-%m-%d}.csv.gz"
    target_dir.mkdir(parents=True, exist_ok=True)
    temp = target_dir / f"{target.name}.{uuid.uuid4().hex}.part"
    client = _s3_client(access, secret)
    try:
        client.download_file(FILES_BUCKET, key, str(temp))
        temp.replace(target)
        return "downloaded"
    except (ClientError, BotoCoreError, OSError) as exc:
        temp.unlink(missing_ok=True)
        code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return "missing"
        raise RuntimeError(f"failed to download {key}: {exc}") from exc


def download_flat_files(args, access: str, secret: str) -> None:
    days = list(_date_range(args.start_date, args.end_date))
    jobs = [(name, prefix, day) for name, prefix in FLAT_DATASETS.items() for day in days]
    counts = {"downloaded": 0, "skipped": 0, "missing": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_download_flat_one, name, prefix, day, args.output, access, secret, args.overwrite) for name, prefix, day in jobs]
        for future in as_completed(futures):
            status = future.result()
            counts[status] += 1
            if sum(counts.values()) % 20 == 0:
                LOGGER.info("flat files progress=%d/%d counts=%s", sum(counts.values()), len(jobs), counts)
    LOGGER.info("flat files complete counts=%s", counts)


def download_api_datasets(args, key: str) -> None:
    selected = args.datasets or list(API_DATASETS)
    manifest: list[dict[str, Any]] = []
    for name in selected:
        endpoint, relative, limit = API_DATASETS[name]
        params: dict[str, Any] = {"limit": limit}
        try:
            if name == "tickers":
                params.update({"market": "stocks", "order": "asc", "sort": "ticker", "active": "true"})
                active = _paginate(endpoint, key, params, args.timeout)
                params["active"] = "false"
                inactive = _paginate(endpoint, key, params, args.timeout)
                merged = {(str(row.get("ticker")), bool(row.get("active"))): row for row in active + inactive}
                records = list(merged.values())
            elif name == "ticker_types":
                params.update({"asset_class": "stocks", "locale": "us"})
                records = _paginate(endpoint, key, params, args.timeout)
            elif name == "ipos":
                params.update({"order": "desc", "sort": "listing_date"})
                records = _paginate(endpoint, key, params, args.timeout)
            elif name == "conditions":
                params.update({"asset_class": "stocks", "order": "asc", "sort": "asset_class"})
                records = _paginate(endpoint, key, params, args.timeout)
            elif name == "market_status":
                records = _request_json(f"{MASSIVE_ENDPOINT}{endpoint}", key, args.timeout)
            else:
                records = _paginate(endpoint, key, params, args.timeout)
            target = _write_json(args.output / relative, f"{MASSIVE_ENDPOINT}{endpoint}", records)
            manifest.append({
                "dataset": name,
                "status": "ok",
                "path": str(target.relative_to(args.output)),
                "record_count": len(records) if isinstance(records, (list, dict)) else None,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            })
        except (RuntimeError, OSError) as exc:
            LOGGER.warning("dataset %s failed: %s", name, exc)
            manifest.append({
                "dataset": name,
                "status": "error",
                "error": str(exc),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            })
    _write_manifest(args.output, manifest)


def _write_manifest(output: Path, datasets: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
    }
    target = output / "manifest.json"
    temp = output / f"{target.name}.{uuid.uuid4().hex}.tmp"
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(target)
    LOGGER.info("saved %s", target)


def download_snapshot(args, key: str) -> None:
    endpoint = "/v2/snapshot/locale/us/markets/stocks/tickers"
    payload = _request_json(f"{MASSIVE_ENDPOINT}{endpoint}?include_otc=false", key, args.timeout)
    output = args.output / "snapshots" / "full_market"
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"snapshot_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("saved snapshot %s", target)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=_parse_date, required=True, help="first UTC date for flat files")
    parser.add_argument("--end-date", type=_parse_date, default=date.today(), help="last UTC date for flat files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=sorted(API_DATASETS), help="API datasets; default: all bulk datasets")
    parser.add_argument("--skip-flat-files", action="store_true")
    parser.add_argument("--include-snapshot", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.end_date < args.start_date:
        parser.error("--end-date cannot be before --start-date")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        access, key = _load_credentials(args.config)
        if not args.skip_flat_files:
            download_flat_files(args, access, key)
        download_api_datasets(args, key)
        if args.include_snapshot:
            download_snapshot(args, key)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
