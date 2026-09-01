import json
import os
from math import log
from types import SimpleNamespace

import polars as pl
import pytest

from core.modules.reporting import export as report_export
from core.modules.reporting.review_assets import OVERVIEW_TEMPLATE, PAIR_TEMPLATE, REVIEW_JS
from core.modules.reporting.pair_summary import build_pair_summary
from core.modules.reporting.trade_review import (
    _complete_trade_rows,
    _inject_report_center_link,
    _pair_contribution_points,
    _raw_pair_prices,
    _report_center_html,
    _resolved_run_dir,
    discover_report_runs,
    trade_rows,
)


def _trades_frame():
    return pl.DataFrame(
        {
            "group_id": ["open-group", "open-group", "close-group", "close-group"],
            "action": ["open", "open", "close", "close"],
            "ts": [1_000, 1_000, 2_000, 2_000],
            "position_id": ["position-1"] * 4,
            "pair_id": ["pair-1"] * 4,
            "symbol": ["X", "Y", "X", "Y"],
            "side": ["buy", "sell", "sell", "buy"],
            "quantity": [2.0, 1.0, 2.0, 1.0],
            "price": [10.01, 20.02, 10.99, 18.98],
            "notional": [20.02, 20.02, 21.98, 18.98],
            "fee": [0.1, 0.1, 0.11, 0.1],
            "slippage": [0.02, 0.02, 0.022, 0.019],
            "funding_fee": [0.0, 0.0, 0.0, 0.0],
            "target_hedge_ratio": [1.0] * 4,
        }
    )


def test_trade_review_combines_open_and_close_into_one_position():
    events = trade_rows(_trades_frame())
    complete = _complete_trade_rows(events)

    assert len(events) == 2
    assert len(complete) == 1
    row = complete[0]
    assert row["open_ts"] == 1_000
    assert row["close_ts"] == 2_000
    assert row["close_legs"]
    assert row["gross_pnl"] is not None
    assert row["net_pnl"] == row["gross_pnl"] - row["fee"] - row["slippage"]


def test_trade_review_keeps_unclosed_position_without_realized_pnl():
    frame = _trades_frame().filter(pl.col("action") == "open")
    complete = _complete_trade_rows(trade_rows(frame))

    assert len(complete) == 1
    assert complete[0]["close_ts"] is None
    assert complete[0]["gross_pnl"] is None
    assert complete[0]["net_pnl"] is None


def test_review_shortcuts_share_the_minute_range_control():
    assets = REVIEW_JS + OVERVIEW_TEMPLATE + PAIR_TEMPLATE
    assert "const sliderMax = 1000;" not in assets
    assert "state.setView(start, end);" in OVERVIEW_TEMPLATE
    assert "state.setView(start, end);" in PAIR_TEMPLATE


def test_overview_exposes_pair_jump_and_pair_page_back_link():
    assert 'id="pairSelect"' in OVERVIEW_TEMPLATE
    assert 'id="openPairBtn"' in OVERVIEW_TEMPLATE
    assert 'trade_review_pairs/" + encodeURIComponent(pairId) + ".html"' in OVERVIEW_TEMPLATE
    assert 'href="../trade_review.html"' in PAIR_TEMPLATE


def test_report_center_discovers_and_sorts_backtest_folders(tmp_path):
    older = tmp_path / "backtest_older"
    newer = tmp_path / "backtest_newer"
    ignored = tmp_path / "unrelated"
    for path in (older, newer, ignored):
        path.mkdir()

    (older / "config.yaml").write_text(
        """
active_setup: demo
setups:
  demo:
    pipeline:
      estimator:
        method: tls
        regression_method: price
        model_lookback_bars: 28800
        model_update_interval_bars: 1440
      signal:
        entry_z: 2.5
        exit_z: 0.5
""",
        encoding="utf-8",
    )
    (older / "metrics.json").write_text(
        json.dumps({"final_equity": 120000, "total_return": 0.2, "sharpe": 1.1}),
        encoding="utf-8",
    )
    (older / "trade_review.html").write_text("ready", encoding="utf-8")
    (newer / "metrics.json").write_text(json.dumps({"final_equity": 90000}), encoding="utf-8")
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    runs = discover_report_runs(tmp_path)

    assert [run["run_id"] for run in runs] == ["backtest_newer", "backtest_older"]
    assert runs[0]["report_ready"] is False
    assert runs[1]["report_ready"] is True
    assert runs[1]["method"] == "tls"
    assert runs[1]["regression_method"] == "price"
    assert runs[1]["model_lookback_bars"] == 28800
    assert runs[1]["entry_z"] == 2.5


def test_report_center_html_exposes_folder_selector(tmp_path):
    html = _report_center_html(
        [{"run_id": "backtest_demo", "report_ready": True, "modified_at": 1}],
        tmp_path,
    )

    assert 'id="runSelect"' in html
    assert 'id="openRun"' in html
    assert '"run_id":"backtest_demo"' in html
    assert '"report_ready":true' in html


def test_report_center_rejects_paths_outside_results_root(tmp_path):
    valid = tmp_path / "valid_run"
    valid.mkdir()

    assert _resolved_run_dir(tmp_path, "valid_run") == valid.resolve()
    assert _resolved_run_dir(tmp_path, "../outside") is None
    assert _resolved_run_dir(tmp_path, "valid_run/nested") is None


def test_generated_reports_expose_report_center_link():
    assert 'id="runIndexLink"' in OVERVIEW_TEMPLATE
    assert 'id="runIndexLink"' in PAIR_TEMPLATE
    assert 'window.location.pathname.startsWith("/runs/")' in OVERVIEW_TEMPLATE
    assert 'window.location.pathname.startsWith("/runs/")' in PAIR_TEMPLATE


def test_pair_template_keeps_all_cards_and_uses_training_window_context():
    assert 'const trades = config.trades || [];' in PAIR_TEMPLATE
    assert 'renderTradeList();' in PAIR_TEMPLATE
    assert '原始价差（Y-X）' in PAIR_TEMPLATE
    assert 'config.model_lookback_bars' in PAIR_TEMPLATE
    assert 'selectedTradeId' in PAIR_TEMPLATE


def test_pair_template_places_theoretical_y_chart_before_zscore():
    theoretical = PAIR_TEMPLATE.index('id="theoreticalYChart"')
    zscore = PAIR_TEMPLATE.index('id="zscoreChart"')

    assert theoretical < zscore
    assert 'key: "y_theoretical"' in PAIR_TEMPLATE
    assert 'config.regression_method === "log_price"' in PAIR_TEMPLATE
    assert 'config.regression_method === "price"' in PAIR_TEMPLATE


def test_report_center_adds_switch_link_to_legacy_report_html():
    legacy = b'<html><body><div class="buttons"></div></body></html>'

    upgraded = _inject_report_center_link(legacy)

    assert b'id = "runIndexLink"' in upgraded
    assert b'href = "/"' in upgraded
    assert upgraded.endswith(b"</html>")


def test_full_backtest_report_always_exports_trade_review(tmp_path, monkeypatch):
    result = SimpleNamespace(
        metrics={"initial_equity": 100.0, "final_equity": 101.0},
        equity_curve=[],
        position_curve=[],
        signal_curve=[],
        orders=[],
        trades=[],
        funding_payments=[],
        risk_history=[],
        final_position_valuation={},
        pair_curve=[],
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("active_setup: demo\nsetups: {demo: {}}\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(report_export, "_per_pair_metrics", lambda *_: [])
    monkeypatch.setattr(report_export, "export_equity_summary_html", lambda *_: None)
    monkeypatch.setattr(
        report_export,
        "_export_trade_review",
        lambda output_dir, max_points, max_pairs: calls.append(
            (output_dir, max_points, max_pairs)
        ),
    )

    output_dir = report_export.export_backtest_report(
        result,
        config_path,
        output_root=tmp_path / "reports",
        run_name="demo",
    )

    assert calls == [(output_dir, 2000, 20)]
    assert "trade_review.html" in (output_dir / "summary.md").read_text(encoding="utf-8")


def test_report_export_survives_trade_review_failure(tmp_path, monkeypatch):
    result = SimpleNamespace(
        metrics={"initial_equity": 100.0, "final_equity": 101.0},
        equity_curve=[], position_curve=[], signal_curve=[], orders=[], trades=[],
        funding_payments=[], risk_history=[], final_position_valuation={}, pair_curve=[],
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("active_setup: demo\nsetups: {demo: {}}\n", encoding="utf-8")
    monkeypatch.setattr(report_export, "_per_pair_metrics", lambda *_: [])
    monkeypatch.setattr(report_export, "export_equity_summary_html", lambda *_: None)
    monkeypatch.setattr(
        report_export, "_export_trade_review",
        lambda *_: (_ for _ in ()).throw(RuntimeError("broken html")),
    )

    output_dir = report_export.export_backtest_report(
        result, config_path, output_root=tmp_path / "reports", run_name="demo"
    )

    assert (output_dir / "metrics.json").exists()
    assert "broken html" in (output_dir / "trade_review_error.txt").read_text(encoding="utf-8")


def test_pair_summary_uses_position_funding_for_pnl_and_win_rate():
    trades = _trades_frame().with_columns(
        pl.when(pl.col("action") == "close")
        .then(pl.when(pl.col("symbol") == "X").then(10.11).otherwise(20.02))
        .otherwise(pl.col("price"))
        .alias("price"),
    ).with_columns((pl.col("price") * pl.col("quantity")).abs().alias("notional"))
    funding = pl.DataFrame(
        {
            "exchange": [None], "symbol": ["X"], "ts": [1_500],
            "funding_rate": [0.02], "mark_price": [10.0], "payment": [0.4],
        }
    )
    summary = build_pair_summary(
        trades=trades,
        position_curve=pl.DataFrame({"ts": [2_000], "X_position_value": [0.0], "Y_position_value": [0.0]}),
        funding_payments=funding,
        pair_defs=[{"pair_id": "pair-1", "x_symbol": "X", "y_symbol": "Y"}],
        initial_equity=100.0,
    )[0]

    assert summary["funding_fee"] == 0.4
    assert summary["total_pnl"] < 0
    assert summary["wins"] == 0
    assert summary["losses"] == 1


def test_pair_contribution_carries_last_mark_across_missing_quote():
    trades = [
        {"pair_id": "p", "group_id": "g", "ts": 1, "symbol": "X", "side": "buy", "quantity": 1.0, "notional": 10.0, "fee": 0.0},
        {"pair_id": "p", "group_id": "g", "ts": 1, "symbol": "Y", "side": "sell", "quantity": 1.0, "notional": 10.0, "fee": 0.0},
    ]
    points = _pair_contribution_points(
        "p", "X", "Y", trades, [], [1, 2, 3],
        {1: 10.0, 2: 11.0, 3: 12.0}, {1: 10.0, 3: 10.0}, [],
    )

    assert points[1]["y_close"] is None
    assert points[1]["pnl"] == 1.0


def test_pair_contribution_builds_log_price_theoretical_y_from_active_parameters():
    alpha = log(20.0) - 2.0 * log(10.0)
    z_rows = [{"ts": 1, "zscore": 0.0, "alpha": alpha, "spread_beta": 2.0}]

    points = _pair_contribution_points(
        "p", "X", "Y", [], [], [1, 2],
        {1: 10.0, 2: 11.0}, {1: 19.0, 2: 23.0}, z_rows,
        regression_method="log_price",
    )

    assert points[0]["y_theoretical"] == pytest.approx(20.0)
    assert points[1]["y_theoretical"] == pytest.approx(24.2)


def test_pair_contribution_does_not_invent_theoretical_price_for_return_model():
    z_rows = [{"ts": 1, "zscore": 0.0, "alpha": 0.1, "spread_beta": 2.0}]

    points = _pair_contribution_points(
        "p", "X", "Y", [], [], [1], {1: 10.0}, {1: 20.0}, z_rows,
        regression_method="log_return",
    )

    assert points[0]["y_theoretical"] is None


def test_raw_trade_spread_prefers_same_event_reference_prices():
    events = [{
        "ts": 1000,
        "leg_details": [
            {"symbol": "X", "reference_price": 10.0, "price": 10.1},
            {"symbol": "Y", "reference_price": 20.0, "price": 19.9},
        ],
    }]

    x, y, spread = _raw_pair_prices(
        1000, "X", "Y", {1000: 99.0}, {1000: 199.0}, events
    )

    assert (x, y, spread) == (10.0, 20.0, 10.0)
