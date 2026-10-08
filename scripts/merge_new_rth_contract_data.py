"""Merge selected new contract files with Massive adjusted RTH history."""
from __future__ import annotations
import argparse, json, logging
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STOCK_DIR = ROOT / "research" / "\u65b0\u589etradfi\u4ea4\u6613\u5bf9" / "by_stock_adjusted_RTH"
DEFAULT_CONTRACT_DIR = ROOT / "data" / "crypto"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "crypto_extended"
LOG = logging.getLogger("merge_new_rth_contract_data")
SYMBOLS = "AMATUSDT AMDUSDT AMZNUSDT APLDUSDT ARMUSDT AVGOUSDT BTCUSDT BXUSDT CIENUSDT COINUSDT COSTUSDT CRMUSDT CRWDUSDT DDOGUSDT EBAYUSDT GEVUSDT GSUSDT HOODUSDT INTCUSDT IRENUSDT JPMUSDT KLACUSDT LITEUSDT LRCXUSDT MPUSDT MRVLUSDT MSFTUSDT MUUSDT NETUSDT NOWUSDT NVDAUSDT NVDLUSDT PANWUSDT QQQUSDT SHOPUSDT SNDKUSDT SOFIUSDT SPYUSDT TSLAUSDT TSLLUSDT USARUSDT VRTUSDT WMTUSDT XAGUSDT XAUUSDT XLEUSDT XOMUSDT XPTUSDT".split()
COLS = ["date","symbol","open_time","open_time_utc","open_time_date","open","high","low","close","volume","close_time","close_time_utc","close_time_date","amount","number_of_trades","buy_volume","buy_amount"]

def stock_path(root: Path, symbol: str):
    p = root / f"{symbol.removesuffix('USDT')}.csv"
    return p if p.is_file() else None

def contract_path(root: Path, symbol: str):
    d = root / symbol
    p = d / f"{symbol}-1m.csv"
    if p.is_file(): return p
    xs = sorted(d.glob("*-1m.csv"))
    return xs[0] if xs else None

def read_stock(path: Path, symbol: str):
    f = pd.read_csv(path, encoding="utf-8-sig")
    req = {"dt_utc","open","high","low","close","volume"}
    missing = req - set(f.columns)
    if missing: raise ValueError(f"{path} missing columns: {sorted(missing)}")
    f = f[["dt_utc","open","high","low","close","volume"]].copy()
    parsed = pd.to_datetime(f["dt_utc"], utc=True, errors="coerce")
    f["open_time"] = parsed.astype("int64") // 1_000_000
    f["symbol"] = symbol
    return f.dropna(subset=["open_time"])

def read_contract(path: Path | None, symbol: str):
    if path is None:
        return pd.DataFrame(columns=COLS)
    f = pd.read_csv(path, encoding="utf-8-sig")
    req = {"open_time","open","high","low","close","volume"}
    missing = req - set(f.columns)
    if missing: raise ValueError(f"{path} missing columns: {sorted(missing)}")
    for c in COLS:
        if c not in f: f[c] = 0 if c in {"amount","number_of_trades","buy_volume","buy_amount"} else ""
    f["symbol"] = symbol
    f["open_time"] = pd.to_numeric(f["open_time"], errors="coerce")
    return f.dropna(subset=["open_time"])[COLS]

def numeric(f):
    f = f.copy()
    f["open_time"] = pd.to_numeric(f["open_time"], errors="coerce")
    for c in ("open","high","low","close","volume","amount","number_of_trades","buy_volume","buy_amount"):
        if c in f.columns:
            f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0.0)
    return f.dropna(subset=["open_time"])

def stock_prefix(stock, cutoff, symbol):
    stock = stock[stock["open_time"] < cutoff].copy()
    if stock.empty: return pd.DataFrame(columns=COLS)
    ts = stock["open_time"].astype("int64")
    utc = pd.to_datetime(ts, unit="ms", utc=True)
    local = utc.dt.tz_convert("Asia/Shanghai")
    return pd.DataFrame({
        "date": utc.dt.strftime("%Y-%m-%d %H:%M:%S"), "symbol": symbol, "open_time": ts,
        "open_time_utc": utc.dt.strftime("%Y-%m-%d %H:%M:%S"), "open_time_date": local.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "open": stock["open"].to_numpy(), "high": stock["high"].to_numpy(), "low": stock["low"].to_numpy(), "close": stock["close"].to_numpy(), "volume": stock["volume"].to_numpy(),
        "close_time": ts + 59999, "close_time_utc": (utc + pd.Timedelta(milliseconds=59999)).dt.strftime("%Y-%m-%d %H:%M:%S"), "close_time_date": (local + pd.Timedelta(milliseconds=59999)).dt.strftime("%Y-%m-%d %H:%M:%S"),
        "amount": 0.0, "number_of_trades": 0, "buy_volume": 0.0, "buy_amount": 0.0,
    }, columns=COLS)

def merge_one(sp, cp, out, symbol):
    stock, contract = numeric(read_stock(sp, symbol)), numeric(read_contract(cp, symbol))
    if contract.empty:
        # No contract rows yet: export the full Massive history in contract schema.
        cutoff = None
        result = stock_prefix(stock, 2**63 - 1, symbol)
    else:
        cutoff = int(contract["open_time"].min())
        result = pd.concat([stock_prefix(stock, cutoff, symbol), contract], ignore_index=True)
    result = result.sort_values("open_time", kind="stable")
    result = result.drop_duplicates("open_time", keep="last")
    result["open_time"] = result["open_time"].astype("int64")
    result["close_time"] = pd.to_numeric(result["close_time"], errors="coerce").fillna(result["open_time"] + 59999).astype("int64")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    result[COLS].to_csv(tmp, index=False)
    tmp.replace(out)
    return {"symbol": symbol, "massive_rows": int(len(result) - len(contract)), "contract_rows": int(len(contract)), "output_rows": int(len(result)), "contract_start_utc": (datetime.fromtimestamp(cutoff / 1000, tz=timezone.utc).isoformat() if cutoff is not None else None), "output_path": str(out)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stock-dir", type=Path, default=DEFAULT_STOCK_DIR)
    p.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--symbols", default=",".join(SYMBOLS))
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    symbols = list(dict.fromkeys(x.strip().upper() for x in a.symbols.split(",") if x.strip()))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"stock_dir": str(a.stock_dir), "contract_dir": str(a.contract_dir), "output_dir": str(a.output_dir), "results": [], "skipped": []}
    for s in symbols:
        sp, cp = stock_path(a.stock_dir, s), contract_path(a.contract_dir, s)
        out = a.output_dir / s / f"{s}-1m-extended.csv"
        if sp is None: LOG.warning("%s: Massive RTH file missing; skipped", s); summary["skipped"].append({"symbol":s,"reason":"massive_file_missing"}); continue
        if cp is None: LOG.info("%s: contract file missing; using Massive history only", s)
        if out.exists() and not a.overwrite: LOG.warning("%s: output exists, skipped: %s", s, out); summary["skipped"].append({"symbol":s,"reason":"output_exists"}); continue
        try:
            r = merge_one(sp, cp, out, s); summary["results"].append(r); LOG.info("%s: Massive=%d contract=%d output=%d", s, r["massive_rows"], r["contract_rows"], r["output_rows"])
        except (OSError, ValueError, pd.errors.ParserError) as e:
            LOG.error("%s: failed: %s", s, e); summary["skipped"].append({"symbol":s,"reason":str(e)})
    (a.output_dir / "merge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOG.info("complete: generated=%d skipped=%d", len(summary["results"]), len(summary["skipped"]))

if __name__ == "__main__": main()









