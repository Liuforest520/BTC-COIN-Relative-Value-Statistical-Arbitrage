"""Plot raw-vs-adjusted prices after in-place TradFi split adjustment."""

from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_right
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = PROJECT_ROOT / "data" / "corporate_actions" / "splits.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "tradfi"
DEFAULT_OUTPUT = PROJECT_ROOT / "research" / f"split_adjustment_{date.today():%Y%m%d}"


def load_events(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, list[dict[str, Any]]] = {}
    for item in payload.get("results", []):
        ticker = str(item.get("ticker", "")).upper()
        if ticker and item.get("execution_date") and item.get("historical_adjustment_factor") not in (None, 0):
            result.setdefault(ticker, []).append({"date": str(item["execution_date"]), "factor": float(item["historical_adjustment_factor"]), "item": item})
    for items in result.values():
        items.sort(key=lambda x: x["date"])
    return result


def plot_file(path: Path, events: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    dates = [x["date"] for x in events]
    daily: dict[str, tuple[float, float]] = {}
    first_day = last_day = None
    close_i = None
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        ts_i = header.index("ts")
        close_i = header.index("close")
        for row in reader:
            if not row:
                continue
            day = datetime.fromtimestamp(int(float(row[ts_i])) / 1000, timezone.utc).date().isoformat()
            first_day = first_day or day
            last_day = day
            index = bisect_right(dates, day)
            factor = events[index]["factor"] if index < len(events) else 1.0
            adjusted_close = float(row[close_i])
            raw_close = adjusted_close / factor
            daily[day] = (raw_close, adjusted_close)
    days = sorted(daily)
    x = [datetime.fromisoformat(d) for d in days]
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.plot(x, [daily[d][0] for d in days], linewidth=0.8, label="Raw close", alpha=0.75)
    ax.plot(x, [daily[d][1] for d in days], linewidth=0.9, label="Split-adjusted close")
    for event in events:
        if first_day and last_day and first_day <= event["date"] <= last_day:
            ax.axvline(datetime.fromisoformat(event["date"]), color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
            ax.text(datetime.fromisoformat(event["date"]), 0.98, event["date"], rotation=90, transform=ax.get_xaxis_transform(), fontsize=7, va="top")
    ax.set_title(f"{path.parent.name}: raw vs split-adjusted daily close")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return {"file": str(path), "start_date": first_day, "end_date": last_day, "days": len(days), "events": [x["item"] for x in events]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    events_by_ticker = load_events(args.splits)
    summaries = []
    for path in sorted(args.data_dir.glob("*/*-1m-massive-engine.csv")):
        symbol = path.parent.name.upper()
        ticker = symbol[:-4] if symbol.endswith("USDT") else symbol
        events = events_by_ticker.get(ticker, [])
        if not events:
            continue
        with path.open("r", encoding="utf-8") as fh:
            next(fh)
            first = next(fh).split(",", 1)[0]
            last = first
            for line in fh:
                if line.strip():
                    last = line.split(",", 1)[0]
        first_day = datetime.fromtimestamp(int(float(first)) / 1000, timezone.utc).date().isoformat()
        last_day = datetime.fromtimestamp(int(float(last)) / 1000, timezone.utc).date().isoformat()
        relevant = [x for x in events if first_day <= x["date"] <= last_day]
        if relevant:
            window_events = [x for x in events if x["date"] <= last_day]
            summaries.append(plot_file(path, window_events, args.output_dir / f"{symbol}_raw_vs_adjusted.png"))
    (args.output_dir / "plot_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(summaries)} plots to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
