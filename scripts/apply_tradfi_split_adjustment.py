"""Apply Massive split adjustment factors to TradFi minute files in place.

Only files whose date range intersects a known split event are changed.  The
script keeps a raw-vs-adjusted daily-close plot for each changed symbol in
research/split_adjustment_YYYYMMDD/ before replacing the source file.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import tempfile
from bisect import bisect_right
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = PROJECT_ROOT / "data" / "corporate_actions" / "splits.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "tradfi"
DEFAULT_RESEARCH_DIR = PROJECT_ROOT / "research" / f"split_adjustment_{date.today():%Y%m%d}"
LOGGER = logging.getLogger("apply_tradfi_split_adjustment")


def _load_events(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events: dict[str, list[dict[str, Any]]] = {}
    for item in payload.get("results", []):
        ticker = str(item.get("ticker", "")).strip().upper()
        execution_date = str(item.get("execution_date", ""))
        factor = item.get("historical_adjustment_factor")
        if not ticker or not execution_date or factor in (None, 0):
            continue
        events.setdefault(ticker, []).append(
            {"execution_date": execution_date, "factor": float(factor), "item": item}
        )
    for ticker in events:
        events[ticker].sort(key=lambda x: x["execution_date"])
    return events


def _factor_for_day(day: str, events: list[dict[str, Any]], dates: list[str]) -> float:
    index = bisect_right(dates, day)
    return events[index]["factor"] if index < len(events) else 1.0


def _number(value: str, factor: float, divide: bool = False) -> str:
    result = float(value) / factor if divide else float(value) * factor
    return f"{result:.15g}"


def _process_file(path: Path, events: list[dict[str, Any]], plot_path: Path) -> dict[str, Any]:
    event_dates = [x["execution_date"] for x in events]
    daily: dict[str, tuple[float, float]] = {}
    row_count = 0
    adjusted_rows = 0
    first_day: str | None = None
    last_day: str | None = None

    with path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        header = next(reader)
        try:
            ts_i = header.index("ts")
            open_i = header.index("open")
            high_i = header.index("high")
            low_i = header.index("low")
            close_i = header.index("close")
            volume_i = header.index("volume")
        except ValueError as exc:
            raise ValueError(f"{path} does not contain expected OHLCV columns") from exc

        with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as target:
            temp_path = Path(target.name)
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(header)
            for row in reader:
                if not row:
                    continue
                row_count += 1
                timestamp = int(float(row[ts_i]))
                day = datetime.fromtimestamp(timestamp / 1000, timezone.utc).date().isoformat()
                first_day = first_day or day
                last_day = day
                factor = _factor_for_day(day, events, event_dates)
                raw_close = float(row[close_i])
                if factor != 1.0:
                    for index in (open_i, high_i, low_i, close_i):
                        row[index] = _number(row[index], factor)
                    row[volume_i] = _number(row[volume_i], factor, divide=True)
                    adjusted_rows += 1
                daily[day] = (raw_close, float(row[close_i]))
                writer.writerow(row)

    os.replace(temp_path, path)
    days = sorted(daily)
    raw = [daily[d][0] for d in days]
    adjusted = [daily[d][1] for d in days]
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.plot(days, raw, linewidth=0.8, label="Raw close", alpha=0.75)
    ax.plot(days, adjusted, linewidth=0.9, label="Split-adjusted close")
    for event in events:
        if first_day and last_day and first_day <= event["execution_date"] <= last_day:
            ax.axvline(event["execution_date"], color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
            ax.text(event["execution_date"], 0.98, event["execution_date"], rotation=90, transform=ax.get_xaxis_transform(), fontsize=7, va="top")
    ax.set_title(f"{path.parent.name}: raw vs split-adjusted daily close")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return {"file": str(path), "start_date": first_day, "end_date": last_day, "rows": row_count, "adjusted_rows": adjusted_rows, "events": [x["item"] for x in events]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    parser.add_argument("--symbols", help="comma-separated project symbols; default: auto-detect affected files")
    parser.add_argument("--force", action="store_true", help="allow reapplying adjustment to files already processed in this research directory")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    marker = args.research_dir / "adjustment_summary.json"
    if marker.exists() and not args.force:
        raise SystemExit(f"adjustment marker already exists: {marker}; refusing to double-adjust (use --force only after restoring raw files)")
    events_by_ticker = _load_events(args.splits)
    wanted = {x.strip().upper() for x in args.symbols.split(",")} if args.symbols else None
    summaries: list[dict[str, Any]] = []
    for path in sorted(args.data_dir.glob("*/*-1m-massive-engine.csv")):
        symbol = path.parent.name.upper()
        if not symbol.endswith("USDT"):
            continue
        ticker = symbol[:-4]
        if wanted and symbol not in wanted and ticker not in wanted:
            continue
        events = events_by_ticker.get(ticker, [])
        if not events:
            continue
        # Read only the first/last timestamps to avoid processing symbols whose
        # events all predate the available minute window.
        with path.open("r", encoding="utf-8") as fh:
            next(fh)
            first = next(fh).split(",", 1)[0]
            last = first
            for line in fh:
                if line.strip():
                    last = line.split(",", 1)[0]
        first_day = datetime.fromtimestamp(int(float(first)) / 1000, timezone.utc).date().isoformat()
        last_day = datetime.fromtimestamp(int(float(last)) / 1000, timezone.utc).date().isoformat()
        window_events = [x for x in events if x["execution_date"] <= last_day]
        relevant = [x for x in window_events if first_day <= x["execution_date"] <= last_day]
        if not relevant:
            continue
        LOGGER.info("adjusting %s (%s): %d event(s)", symbol, path, len(relevant))
        summaries.append(_process_file(path, window_events, args.research_dir / f"{symbol}_raw_vs_adjusted.png"))

    if not summaries:
        LOGGER.info("no TradFi file intersects a split event")
        return 0
    args.research_dir.mkdir(parents=True, exist_ok=True)
    (args.research_dir / "adjustment_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("adjusted %d files; plots and summary saved under %s", len(summaries), args.research_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
