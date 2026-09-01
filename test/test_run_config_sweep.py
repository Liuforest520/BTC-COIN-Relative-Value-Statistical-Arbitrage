import pytest

from scripts.run_config_sweep import _worker_count, run_sweep_rows


def test_worker_count_prefers_cli_and_validates():
    assert _worker_count(3, 2) == 3
    assert _worker_count(None, 2) == 2
    with pytest.raises(ValueError, match="at least 1"):
        _worker_count(0, None)


def test_parallel_sweep_returns_worker_errors_in_input_order(tmp_path):
    rows = [
        {"config_id": "000001", "config_path": str(tmp_path / "missing_1.yaml")},
        {"config_id": "000002", "config_path": str(tmp_path / "missing_2.yaml")},
    ]

    results = run_sweep_rows(rows, fast_summary=True, workers=2)

    assert [row["config_id"] for row in results] == ["000001", "000002"]
    assert all(row["error"] for row in results)
    assert all("traceback" in row for row in results)
