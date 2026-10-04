"""Prepend historical stock bars to a later-listed TradFi contract.

For every symbol, the earliest contract ``open_time`` is the hard cutover:

    stock bars with ts < first contract bar
    + all original contract bars from first contract bar onward

The contract feed is authoritative at and after the cutover.  The original
stock and contract files are never modified.  Output uses the contract CSV
schema so it can be inspected or supplied to the existing data loader.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STOCK_DIR = PROJECT_ROOT / "data" / "stocks" / "tradfi"
DEFAULT_CONTRACT_DIR = PROJECT_ROOT / "data" / "crypto"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "crypto_extended"
LOGGER = logging.getLogger("extend_tradfi_contract_history")

CONTRACT_COLUMNS = [
    "date", "symbol", "open_time", "open_time_utc", "open_time_date",
    "open", "high", "low", "close", "volume", "close_time",
    "close_time_utc", "close_time_date", "amount", "number_of_trades",
    "buy_volume", "buy_amount",
]


def _stock_file(root: Path, symbol: str) -> Path | None:
    directory = root / symbol
    if not directory.exists():
        return None
    candidates = []
    for pattern in ("*-1m-massive-engine.csv", "*-1m-massive.csv", "*.csv"):
        candidates.extend(p for p in directory.glob(pattern) if p.is_file())
    candidates = [p for p in candidates if "extended" not in p.stem and "merged" not in p.stem]
    if not candidates:
        return None
    candidates.sort(key=lambda p: ("engine" not in p.name, len(p.name), p.name))
    return candidates[0]


def _contract_file(root: Path, symbol: str) -> Path | None:
    directory = root / symbol
    for candidate in (directory / f"{symbol}-1m.csv", *directory.glob("*-1m.csv")):
        if candidate.is_file() and "extended" not in candidate.stem:
            return candidate
    return None


def _read_stock(path: Path, symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"ts", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing stock columns: {sorted(missing)}")
    frame = frame[["ts", "open", "high", "low", "close", "volume"]].copy()
    frame = frame.rename(columns={"ts": "open_time"})
    frame["symbol"] = symbol
    return frame


def _read_contract(path: Path, symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"open_time", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing contract columns: {sorted(missing)}")
    # Keep all known contract metadata, while tolerating older files with a
    # smaller schema. Missing optional columns are reconstructed below.
    for column in CONTRACT_COLUMNS:
        if column not in frame:
            frame[column] = 0 if column in {
                "amount", "number_of_trades", "buy_volume", "buy_amount"
            } else ""
    frame["symbol"] = symbol
    return frame[CONTRACT_COLUMNS]


def _normalise_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["open_time"] = pd.to_numeric(frame["open_time"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "amount", "number_of_trades", "buy_volume", "buy_amount"):
        # Stock history only has OHLCV; reconstruct optional contract metadata.
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame.dropna(subset=["open_time"])


def _stock_prefix(stock: pd.DataFrame, cutoff: int, symbol: str) -> pd.DataFrame:
    stock = stock[stock["open_time"] < cutoff].copy()
    if stock.empty:
        return pd.DataFrame(columns=CONTRACT_COLUMNS)
    ts = stock["open_time"].astype("int64")
    utc = pd.to_datetime(ts, unit="ms", utc=True)
    local = utc.dt.tz_convert("Asia/Shanghai")
    prefix = pd.DataFrame({
        "date": utc.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "open_time": ts,
        "open_time_utc": utc.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "open_time_date": local.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "open": stock["open"].to_numpy(),
        "high": stock["high"].to_numpy(),
        "low": stock["low"].to_numpy(),
        "close": stock["close"].to_numpy(),
        "volume": stock["volume"].to_numpy(),
        "close_time": ts + 59999,
        "close_time_utc": (utc + pd.Timedelta(milliseconds=59999)).dt.strftime("%Y-%m-%d %H:%M:%S"),
        "close_time_date": (local + pd.Timedelta(milliseconds=59999)).dt.strftime("%Y-%m-%d %H:%M:%S"),
        "amount": 0.0,
        "number_of_trades": 0,
        "buy_volume": 0.0,
        "buy_amount": 0.0,
    })
    return prefix[CONTRACT_COLUMNS]


def extend_symbol(stock_path: Path, contract_path: Path, output_path: Path, symbol: str) -> dict:
    stock = _normalise_numeric(_read_stock(stock_path, symbol))
    contract = _normalise_numeric(_read_contract(contract_path, symbol))
    if contract.empty:
        raise ValueError("contract file contains no valid open_time rows")
    cutoff = int(contract["open_time"].min())
    prefix = _stock_prefix(stock, cutoff, symbol)
    result = pd.concat([prefix, contract], ignore_index=True)
    result = result.sort_values("open_time", kind="stable")
    result = result.drop_duplicates(subset=["open_time"], keep="last")
    result["open_time"] = result["open_time"].astype("int64")
    result["close_time"] = pd.to_numeric(result["close_time"], errors="coerce").fillna(result["open_time"] + 59999).astype("int64")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    result[CONTRACT_COLUMNS].to_csv(temporary, index=False)
    temporary.replace(output_path)
    return {
        "symbol": symbol,
        "stock_rows_before_cutover": int(len(prefix)),
        "contract_rows": int(len(contract)),
        "output_rows": int(len(result)),
        "first_contract_open_time": cutoff,
        "stock_first_ts": int(stock["open_time"].min()) if not stock.empty else None,
        "output_path": str(output_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-dir", type=Path, default=DEFAULT_STOCK_DIR)
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--symbols", help="comma-separated symbols; default: all matching symbols")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    stock_symbols = {p.name for p in args.stock_dir.iterdir() if p.is_dir()} if args.stock_dir.exists() else set()
    contract_symbols = {p.name for p in args.contract_dir.iterdir() if p.is_dir()} if args.contract_dir.exists() else set()
    symbols = (
        {x.strip().upper() for x in args.symbols.split(",") if x.strip()}
        if args.symbols else stock_symbols & contract_symbols
    )
    if not symbols:
        parser.error("no matching symbols found")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stock_dir": str(args.stock_dir),
        "contract_dir": str(args.contract_dir),
        "cutover_rule": "stock rows strictly before earliest contract open_time; contract rows from cutover onward",
        "symbols": [],
    }
    failures = 0
    for symbol in sorted(symbols):
        stock_path = _stock_file(args.stock_dir, symbol)
        contract_path = _contract_file(args.contract_dir, symbol)
        if stock_path is None or contract_path is None:
            LOGGER.warning("%s: missing stock or contract file; skipped", symbol)
            continue
        output_path = args.output_dir / symbol / f"{symbol}-1m-extended.csv"
        if output_path.exists() and not args.overwrite:
            LOGGER.info("%s: output exists, skipped", symbol)
            continue
        try:
            stats = extend_symbol(stock_path, contract_path, output_path, symbol)
            manifest["symbols"].append(stats)
            LOGGER.info(
                "%s: stock_prefix=%d contract=%d output=%d cutover=%s -> %s",
                symbol, stats["stock_rows_before_cutover"], stats["contract_rows"],
                stats["output_rows"],
                datetime.fromtimestamp(stats["first_contract_open_time"] / 1000, tz=timezone.utc).isoformat(),
                output_path,
            )
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            failures += 1
            LOGGER.error("%s: failed: %s", symbol, exc)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("complete: symbols=%d failures=%d", len(manifest["symbols"]), failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

