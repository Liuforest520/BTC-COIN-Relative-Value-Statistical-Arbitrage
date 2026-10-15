"""Disk-backed, timestamp-aligned data for shared-account optimization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

import numpy as np

from core.backtest.backtest import Backtest
from core.backtest.pair_data_cache import NoAlignedPairDataError, load_pair_data


@dataclass(frozen=True)
class PortfolioPairSource:
    pair: dict
    x_path: Path
    y_path: Path
    x_funding_path: Path | None
    y_funding_path: Path | None


@dataclass
class PreparedPortfolioData:
    """One window of pair data backed by temporary ``.npy`` memmaps."""

    timestamps: np.ndarray
    bar_indices: np.ndarray
    values: np.ndarray
    offsets: np.ndarray
    pair_ids: list[str]
    pairs: list[dict]
    target_capital: np.ndarray
    hedge_beta_lookback_bars: np.ndarray
    hedge_beta_min_samples: np.ndarray
    reversion_min_samples: np.ndarray
    stop_freeze_lookback_multiplier: np.ndarray
    wait_for_model_update_after_stop: np.ndarray
    skipped: list[dict]
    temp_dir: Path

    @property
    def pair_count(self) -> int:
        return len(self.pair_ids)

    def close(self) -> None:
        # Windows cannot remove an open mmap. Drop references before cleanup.
        for name in ("bar_indices", "values"):
            value = getattr(self, name, None)
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()
            setattr(self, name, np.empty(0))
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class PortfolioDataStream:
    """Prepare all active pairs without retaining source CSV frames in RAM."""

    VALUE_COLUMNS = 8

    def __init__(
        self,
        config,
        pairs,
        start_ts: int,
        end_ts: int,
        measurement_start_ts: int | None = None,
        warmup_bars: int = 0,
    ):
        self.config = config
        self.pairs = list(pairs)
        self.start_ts = int(start_ts)
        self.end_ts = int(end_ts)
        self.measurement_start_ts = (
            int(measurement_start_ts) if measurement_start_ts is not None else None
        )
        self.warmup_bars = max(0, int(warmup_bars))
        self.skipped: list[dict] = []

    def sources(self):
        for pair in self.pairs:
            pair_id = pair.get("pair_id")
            x_symbol = pair.get("x_symbol", pair.get("long_symbol"))
            y_symbol = pair.get("y_symbol", pair.get("short_symbol"))
            x_info = self.config.symbols.get(x_symbol) or {}
            y_info = self.config.symbols.get(y_symbol) or {}
            if not x_info.get("path") or not y_info.get("path"):
                self.skipped.append({"pair_id": pair_id, "reason": "missing market path"})
                continue
            yield PortfolioPairSource(
                pair=pair,
                x_path=Path(x_info["path"]),
                y_path=Path(y_info["path"]),
                x_funding_path=Backtest._funding_path(x_symbol, x_info),
                y_funding_path=Backtest._funding_path(y_symbol, y_info),
            )

    def load(self, source: PortfolioPairSource):
        try:
            data = load_pair_data(
                source.x_path,
                source.y_path,
                source.x_funding_path,
                source.y_funding_path,
                start_ts=self.start_ts,
                end_ts=self.end_ts,
                measurement_start_ts=self.measurement_start_ts,
                warmup_bars=self.warmup_bars,
            )
            if self.measurement_start_ts is not None:
                first_train = int(np.searchsorted(
                    data.timestamps, self.measurement_start_ts, side="left"
                ))
                if first_train >= data.size:
                    data.close()
                    raise NoAlignedPairDataError("pair has warmup rows but no training rows")
            return data
        except (NoAlignedPairDataError, FileNotFoundError, ValueError, OSError) as exc:
            self.skipped.append({
                "pair_id": source.pair.get("pair_id"),
                "reason": f"{type(exc).__name__}: {exc}",
            })
            return None

    def prepare(self) -> PreparedPortfolioData:
        """Load each pair once, persist compact arrays, and build a union clock."""
        temp_dir = Path(tempfile.mkdtemp(prefix="wf_portfolio_"))
        staged: list[tuple[PortfolioPairSource, Path, int]] = []
        union_ts = np.empty(0, dtype=np.int64)
        try:
            for source_index, source in enumerate(self.sources()):
                data = self.load(source)
                if data is None:
                    continue
                try:
                    pair_file = temp_dir / f"pair_{source_index:04d}.npy"
                    matrix = np.lib.format.open_memmap(
                        pair_file, mode="w+", dtype=np.float64,
                        shape=(data.size, 9),
                    )
                    matrix[:, 0] = data.timestamps
                    matrix[:, 1] = data.x_open
                    matrix[:, 2] = data.x_close
                    matrix[:, 3] = data.y_open
                    matrix[:, 4] = data.y_close
                    matrix[:, 5] = data.funding_x
                    matrix[:, 6] = data.funding_y
                    matrix[:, 7] = data.funding_x_event
                    matrix[:, 8] = data.funding_y_event
                    matrix.flush()
                    del matrix
                    union_ts = np.union1d(union_ts, data.timestamps)
                    staged.append((source, pair_file, data.size))
                finally:
                    data.close()

            if not staged:
                raise NoAlignedPairDataError("no pair has aligned training data")

            return self._combine_staged(temp_dir, staged, union_ts)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def _combine_staged(self, temp_dir, staged, union_ts):
        total_rows = sum(size for _source, _path, size in staged)
        pair_count = len(staged)
        values = np.lib.format.open_memmap(
            temp_dir / "values.npy", mode="w+", dtype=np.float64,
            shape=(total_rows, self.VALUE_COLUMNS),
        )
        bar_indices = np.lib.format.open_memmap(
            temp_dir / "bar_indices.npy", mode="w+", dtype=np.int32,
            shape=(len(union_ts), pair_count),
        )
        bar_indices[:] = -1
        offsets = np.zeros(pair_count + 1, dtype=np.int64)
        pair_ids: list[str] = []
        active_pairs: list[dict] = []
        targets = np.zeros(pair_count, dtype=np.float64)
        beta_lookbacks = np.zeros(pair_count, dtype=np.int64)
        beta_min_samples = np.zeros(pair_count, dtype=np.int64)
        reversion_min_samples = np.zeros(pair_count, dtype=np.int64)
        stop_freeze_multiplier = np.zeros(pair_count, dtype=np.float64)
        wait_after_stop = np.zeros(pair_count, dtype=np.int8)

        cursor = 0
        for pair_index, (source, pair_file, size) in enumerate(staged):
            matrix = np.load(pair_file, mmap_mode="r")
            offsets[pair_index] = cursor
            values[cursor:cursor + size, :] = matrix[:, 1:]
            timestamps = np.asarray(matrix[:, 0], dtype=np.int64)
            union_positions = np.searchsorted(union_ts, timestamps)
            bar_indices[union_positions, pair_index] = np.arange(size, dtype=np.int32)
            cursor += size
            offsets[pair_index + 1] = cursor
            self._append_pair_config(
                source.pair, pair_index, pair_ids, active_pairs, targets,
                beta_lookbacks, beta_min_samples, reversion_min_samples,
                stop_freeze_multiplier, wait_after_stop,
            )
            mmap = getattr(matrix, "_mmap", None)
            if mmap is not None:
                mmap.close()
            pair_file.unlink(missing_ok=True)

        values.flush()
        bar_indices.flush()
        return PreparedPortfolioData(
            timestamps=np.ascontiguousarray(union_ts, dtype=np.int64),
            bar_indices=bar_indices,
            values=values,
            offsets=offsets,
            pair_ids=pair_ids,
            pairs=active_pairs,
            target_capital=targets,
            hedge_beta_lookback_bars=beta_lookbacks,
            hedge_beta_min_samples=beta_min_samples,
            reversion_min_samples=reversion_min_samples,
            stop_freeze_lookback_multiplier=stop_freeze_multiplier,
            wait_for_model_update_after_stop=wait_after_stop,
            skipped=list(self.skipped),
            temp_dir=temp_dir,
        )

    def _append_pair_config(
        self, pair, index, pair_ids, active_pairs, targets, beta_lookbacks,
        beta_min_samples, reversion_min_samples, stop_freeze_multiplier,
        wait_after_stop,
    ):
        pipeline = _pipeline_for_pair(self.config, pair)
        sizing = dict(pipeline.get("sizing") or {})
        signal = dict(pipeline.get("signal") or {})
        protection = dict(pipeline.get("protection") or {})
        pair_ids.append(str(pair.get("pair_id")))
        active_pairs.append(pair)
        targets[index] = float(pair.get("target_capital") or sizing.get("notional") or 0.0)
        beta_lookbacks[index] = int(sizing.get("hedge_beta_lookback_bars", 1440))
        beta_min_samples[index] = int(sizing.get("hedge_beta_min_samples", 30))
        reversion_min_samples[index] = int(
            signal.get("reversion_min_samples")
            or signal.get("reversion_ma_lookback_bars", 60)
        )
        stop_freeze_multiplier[index] = float(
            protection.get("pair_loss_stop_freeze_model_lookback_multiplier") or 0.0
        )
        wait_after_stop[index] = int(bool(
            protection.get("wait_for_model_update_after_non_z_exit", False)
        ))


def _pipeline_for_pair(config, pair):
    market = str(pair.get("market", "legacy"))
    views = getattr(config.strategy, "market_pipelines", {}) or {}
    if market in views:
        pipeline = dict(views[market].get("pipeline") or {})
    else:
        pipeline = dict(config.strategy.pipeline or {})
    for stage, patch in dict(pair.get("parameters") or {}).items():
        if isinstance(patch, dict):
            pipeline.setdefault(stage, {}).update(patch)
    return pipeline


__all__ = ["PortfolioDataStream", "PortfolioPairSource", "PreparedPortfolioData"]
