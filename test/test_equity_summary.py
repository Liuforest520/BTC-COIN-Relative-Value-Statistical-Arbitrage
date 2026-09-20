from types import SimpleNamespace

import numpy as np

from core.modules.reporting.equity_summary import (
    _downsample_extrema,
    _equity_arrays,
    export_equity_summary_image,
)


def test_export_equity_summary_image_writes_png_and_removes_legacy_html(tmp_path):
    output = tmp_path / "portfolio_summary.png"
    legacy_html = tmp_path / "portfolio_summary.html"
    legacy_html.write_text("legacy", encoding="utf-8")
    result = SimpleNamespace(
        metrics={
            "initial_equity": 100.0,
            "final_equity": 103.0,
            "total_return": 0.03,
            "max_drawdown": -0.01,
        },
        equity_curve=[
            {"ts": 1_735_689_600_000, "equity": 100.0},
            {"ts": 1_735_689_660_000, "equity": 99.0},
            {"ts": 1_735_689_720_000, "equity": 103.0},
        ],
    )

    exported = export_equity_summary_image(result, output)

    assert exported == output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not legacy_html.exists()


def test_downsample_extrema_preserves_global_high_and_low():
    timestamps = np.arange(100, dtype=np.int64)
    equities = np.linspace(100.0, 101.0, 100)
    equities[31] = 80.0
    equities[67] = 130.0

    sampled_ts, sampled_equity = _downsample_extrema(timestamps, equities, 20)

    assert len(sampled_ts) <= 20
    assert 80.0 in sampled_equity
    assert 130.0 in sampled_equity


def test_equity_arrays_accepts_compact_sweep_curve():
    timestamps, equities = _equity_arrays(
        {
            "ts": [1_735_689_600_000, 1_735_689_660_000],
            "equity": [100.0, 103.0],
        }
    )

    assert timestamps.tolist() == [1_735_689_600_000, 1_735_689_660_000]
    assert equities.tolist() == [100.0, 103.0]
