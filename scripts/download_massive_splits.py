"""Download Massive stock split events and historical adjustment factors."""

from __future__ import annotations

import argparse
import configparser
import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "massive_key.config"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "massive" / "stocks" / "corporate_actions" / "splits" / "splits.json"
API_ENDPOINT = "https://api.massive.com/stocks/v1/splits"
LOGGER = logging.getLogger("download_massive_splits")


def _load_api_key(config_path: Path) -> str:
    if not config_path.exists():
        raise FileNotFoundError(f"Massive credentials file not found: {config_path}")
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(config_path, encoding="utf-8")
    if not parser.has_section("massive"):
        raise ValueError(f"credentials file must contain [massive]: {config_path}")
    key = parser["massive"].get("aws_secret_access_key", "").strip()
    if not key:
        raise ValueError(f"[massive] aws_secret_access_key is empty: {config_path}")
    return key


def _get_api_key(config_path: Path) -> str:
    return os.environ.get("MASSIVE_API_KEY", "").strip() or _load_api_key(config_path)


def _with_api_key(url: str, api_key: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["apiKey"] = api_key
    return urlunparse(parsed._replace(query=urlencode(query)))


def _request_json(url: str, api_key: str, timeout: float) -> dict[str, Any]:
    request = Request(
        _with_api_key(url, api_key),
        headers={"Accept": "application/json", "User-Agent": "stat-arb-splits-downloader/1.0"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Massive API HTTP {exc.code}: {body[:1000]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Massive API request failed: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Massive API returned a non-object JSON response")
    if payload.get("status") not in {None, "OK"}:
        raise RuntimeError(f"Massive API returned status {payload.get('status')!r}")
    return payload


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def _build_first_url(args: argparse.Namespace) -> str:
    params: dict[str, str] = {"limit": str(args.limit), "sort": "execution_date.asc"}
    if args.ticker:
        params["ticker"] = args.ticker
    if args.start_date:
        params["execution_date.gte"] = args.start_date.isoformat()
    if args.end_date:
        params["execution_date.lte"] = args.end_date.isoformat()
    return f"{API_ENDPOINT}?{urlencode(params)}"


def download_splits(args: argparse.Namespace) -> int:
    api_key = _get_api_key(args.config)
    url = _build_first_url(args)
    records: dict[str, dict[str, Any]] = {}
    seen_urls: set[str] = set()
    pages = 0

    while url:
        if url in seen_urls:
            raise RuntimeError("Massive API returned a repeated next_url; stopping to avoid a loop")
        seen_urls.add(url)
        pages += 1
        LOGGER.info("requesting splits page %d", pages)
        payload = _request_json(url, api_key, args.timeout)
        page_results = payload.get("results") or []
        if not isinstance(page_results, list):
            raise RuntimeError("Massive API response field 'results' is not a list")
        for item in page_results:
            if not isinstance(item, dict):
                continue
            event_id = str(item.get("id") or "")
            fallback = "|".join(str(item.get(k, "")) for k in ("ticker", "execution_date", "split_from", "split_to"))
            records[event_id or fallback] = item
        LOGGER.info("page %d: %d records; %d unique total", pages, len(page_results), len(records))
        next_url = payload.get("next_url")
        url = str(next_url) if next_url else ""

    ordered = sorted(records.values(), key=lambda x: (str(x.get("ticker", "")), str(x.get("execution_date", "")), str(x.get("id", ""))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "source": API_ENDPOINT,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "ticker": args.ticker,
            "execution_date_gte": args.start_date.isoformat() if args.start_date else None,
            "execution_date_lte": args.end_date.isoformat() if args.end_date else None,
        },
        "page_count": pages,
        "record_count": len(ordered),
        "results": ordered,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    LOGGER.info("saved %d split events from %d pages to %s", len(ordered), pages, args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ticker", help="optional ticker filter; omit for all tickers")
    parser.add_argument("--start-date", type=_parse_date, help="include events on/after YYYY-MM-DD")
    parser.add_argument("--end-date", type=_parse_date, default=date.today(), help="include events on/before YYYY-MM-DD (default: today)")
    parser.add_argument("--limit", type=int, default=5000, help="records per page (1-5000)")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 5000:
        parser.error("--limit must be between 1 and 5000")
    if args.start_date and args.start_date > args.end_date:
        parser.error("--start-date cannot be after --end-date")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return download_splits(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
