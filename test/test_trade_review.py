import json
import os
from math import log
from types import SimpleNamespace

import polars as pl
import pytest

from core.modules.reporting import export as report_export
from core.modules.reporting.review_assets import OVERVIEW_TEMPLATE, PAIR_TEMPLATE, REVIEW_JS
from core.modules.reporting.pair_summary import build_pair_summary
from core.modules.reporting.pair_summary import _to_frame
from core.modules.reporting.trade_review import (
    _complete_trade_rows,
    _inject_report_center_link,
    _pair_contribution_points,
    _pair_scenario_summary,
    _raw_pair_prices,
    _report_center_html,
    _resolved_run_dir,
    _zscore_forward_rows,
    export_trade_review_html,
    _trade_scenario_analysis,
    threshold_rows,
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


def test_report_rows_allow_late_protective_exit_strings(tmp_path):
    """Sparse null exit metadata must not be inferred as a Null column."""
    rows = [
        {
            "group_id": f"g-{index}",
            "exit_reason": None,
            "protection_trigger": None,
            "exit_class": None,
            "reopen_lock_pending": None,
            "protection_max_holding_bars": None,
        }
        for index in range(101)
    ]
    rows.append(
        {
            "group_id": "g-close",
            "exit_reason": "protective_max_holding_time",
            "protection_trigger": "max_holding_time",
            "exit_class": "stop_loss",
            "reopen_lock_pending": True,
            "protection_max_holding_bars": 5760,
        }
    )

    frame = _to_frame(rows)
    assert frame.schema["exit_reason"] == pl.Utf8
    assert frame.schema["reopen_lock_pending"] == pl.Boolean
    assert frame.tail(1)["exit_reason"].item() == "protective_max_holding_time"

    output = tmp_path / "trades.csv"
    report_export._write_csv(output, rows)
    written = pl.read_csv(output, schema_overrides={"exit_reason": pl.Utf8})
    assert written.tail(1)["exit_reason"].item() == "protective_max_holding_time"


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


def test_trade_signal_context_uses_previous_signal():
    signal = pl.DataFrame({
        "ts": [0, 1_000, 2_000],
        "pair-1_zscore": [3.0, 0.4, 0.2],
        "pair-1_action": ["open", "close", "none"],
        "pair-1_side": ["long_x", None, None],
        "pair-1_alpha": [0.1, 0.1, 0.1],
        "pair-1_spread_beta": [1.5, 1.5, 1.5],
        "pair-1_spread_mean": [0.0, 0.0, 0.0],
        "pair-1_spread_std": [0.01, 0.01, 0.01],
        "pair-1_last_model_update_index": [900, 900, 900],
    })

    enriched = trade_rows(_trades_frame(), signal)

    assert enriched[0]["signal_ts"] == 0
    assert enriched[0]["signal_last_model_update_index"] == pytest.approx(900)
    assert enriched[1]["signal_ts"] == 1_000


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
    monkeypatch.setattr(report_export, "export_equity_summary_image", lambda *_: None)
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
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "portfolio_summary.png" in summary
    assert "portfolio_summary.html" not in summary
    assert "trade_review.html" in summary


def test_report_export_survives_trade_review_failure(tmp_path, monkeypatch):
    result = SimpleNamespace(
        metrics={"initial_equity": 100.0, "final_equity": 101.0},
        equity_curve=[], position_curve=[], signal_curve=[], orders=[], trades=[],
        funding_payments=[], risk_history=[], final_position_valuation={}, pair_curve=[],
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("active_setup: demo\nsetups: {demo: {}}\n", encoding="utf-8")
    monkeypatch.setattr(report_export, "_per_pair_metrics", lambda *_: [])
    monkeypatch.setattr(report_export, "export_equity_summary_image", lambda *_: None)
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


def test_zscore_rows_forward_fill_model_parameters_from_sparse_pair_curve():
    signal = pl.DataFrame({
        "ts": [1, 2, 3],
        "p_zscore": [None, 2.0, 1.0],
    })
    pair_curve = pl.DataFrame({
        "ts": [1, 3],
        "pair_id": ["p", "p"],
        "alpha": [0.1, 0.2],
        "beta": [1.5, 1.6],
    })

    rows = _zscore_forward_rows(signal, "p", pair_curve)

    assert [row["alpha"] for row in rows] == [0.1, 0.1, 0.2]
    assert [row["spread_beta"] for row in rows] == [1.5, 1.5, 1.6]


def test_complete_trade_uses_signal_time_model_state_not_fill_time_update():
    signal = pl.DataFrame({"ts": [0, 1_000], "pair-1_zscore": [3.2, 3.1]})
    events = trade_rows(_trades_frame().filter(pl.col("action") == "open"), signal)
    pair_curve = pl.DataFrame({
        "ts": [0, 1_000],
        "pair_id": ["pair-1", "pair-1"],
        "alpha": [0.1, 9.9],
        "beta": [1.5, 8.8],
        "spread_mean": [0.0, 7.7],
        "spread_std": [0.2, 6.6],
    })

    complete = _complete_trade_rows(events, pair_curve)

    assert complete[0]["open_alpha"] == pytest.approx(0.1)
    assert complete[0]["open_beta"] == pytest.approx(1.5)


def test_pair_contribution_hides_nonpositive_price_target():
    points = _pair_contribution_points(
        "p", "X", "Y", [], [], [1], {1: 10.0}, {1: 20.0},
        [{"ts": 1, "zscore": 0.0, "alpha": -30.0, "spread_beta": 1.0}],
        regression_method="price",
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


def test_trade_scenario_keeps_valid_rows_when_one_price_target_is_invalid():
    trade = {
        "open_alpha": -0.11368982592970345,
        "open_beta": 5.075325520474826,
        "open_spread_mean": -7.927814783249266e-17,
        "open_spread_std": 0.02368658715682013,
        "open_x_price": 0.02387,
        "open_y_price": 0.092,
        "open_zscore": 3.6420365611046552,
        "close_x_price": 0.06672,
        "close_y_price": 0.237,
        "gross_return": -0.40237301548566695,
        "net_return": -0.39460211257559846,
        "gross_pnl": -3992.0949125634943,
        "net_pnl": -3914.9968449014136,
    }
    open_events = [{
        "leg_details": [
            {"symbol": "BICOUSDT", "side": "buy", "quantity": 144714.32706297422},
            {"symbol": "DYDXUSDT", "side": "sell", "quantity": 70297.26777387544},
        ],
    }]

    analysis = _trade_scenario_analysis(
        trade,
        open_events,
        "BICOUSDT",
        "DYDXUSDT",
        "price",
        0.5,
    )

    assert analysis["available"] is True
    assert analysis["invalid_x_moves"] == [-0.2]
    assert [row["x_move"] for row in analysis["scenarios"]] == [-0.1, 0.0, 0.1, 0.2]
    assert analysis["summary"]["range_fully_valid"] is False
    assert analysis["summary"]["all_profitable"] is None
    assert analysis["actual"]["theoretical_at_actual_x"]["gross_return"] == pytest.approx(
        -0.400796283436749
    )


@pytest.mark.parametrize(
    ("exit_z", "expected"),
    [
        (0.0, [(0.0, "反转平仓 0.00")]),
        (-0.5, [(0.5, "反向平仓 +0.50"), (-0.5, "反向平仓 -0.50")]),
    ],
)
def test_trade_review_thresholds_show_signed_exit_semantics(tmp_path, exit_z, expected):
    (tmp_path / "config.yaml").write_text(
        "\n".join([
            "active_setup: demo",
            "setups:",
            "  demo:",
            "    pipeline:",
            "      signal:",
            "        entry_z: 3.0",
            f"        exit_z: {exit_z}",
        ]),
        encoding="utf-8",
    )

    exits = [row for row in threshold_rows(tmp_path) if row["kind"] == "exit"]

    assert [(row["value"], row["label"]) for row in exits] == expected


def test_trade_scenario_negative_exit_targets_opposite_z_side():
    trade = {
        "open_alpha": 0.0,
        "open_beta": 1.0,
        "open_spread_mean": 0.0,
        "open_spread_std": 0.1,
        "open_x_price": 10.0,
        "open_y_price": 11.0,
        "open_zscore": 3.0,
    }
    open_events = [{"leg_details": [
        {"symbol": "X", "side": "buy", "quantity": 1.0},
        {"symbol": "Y", "side": "sell", "quantity": 1.0},
    ]}]

    analysis = _trade_scenario_analysis(
        trade, open_events, "X", "Y", "price", -0.5
    )

    assert analysis["available"] is True
    assert analysis["target_z"] == -0.5
    assert analysis["target_residual"] == pytest.approx(-0.05)


def test_log_price_scenario_scans_internal_extremum():
    trade = {
        "open_alpha": log(0.005),
        "open_beta": 2.0,
        "open_spread_mean": 0.0,
        "open_spread_std": 0.1,
        "open_x_price": 100.0,
        "open_y_price": 100.0,
        "open_zscore": 3.0,
    }
    open_events = [{"leg_details": [
        {"symbol": "X", "side": "buy", "quantity": 1.0},
        {"symbol": "Y", "side": "sell", "quantity": 1.0},
    ]}]

    analysis = _trade_scenario_analysis(
        trade, open_events, "X", "Y", "log_price", 0.0
    )

    assert analysis["summary"]["range_max"]["x_move"] == pytest.approx(0.0)
    assert analysis["summary"]["range_max"]["gross_return"] == pytest.approx(0.25)
    assert analysis["summary"]["range_min"]["gross_return"] == pytest.approx(0.24)


def test_pair_scenario_summary_aggregates_distributions_and_rates():
    def scenario(zero, low, high):
        return {
            "available": True,
            "summary": {
                "zero_x": {"gross_return": zero},
                "range_min": {"gross_return": low},
                "range_max": {"gross_return": high},
            },
        }

    summary = _pair_scenario_summary([
        {"scenario_analysis": scenario(0.10, 0.05, 0.20), "gross_return": 0.08, "net_return": 0.06},
        {"scenario_analysis": scenario(-0.10, -0.20, 0.05), "gross_return": -0.02, "net_return": -0.03},
    ])

    assert summary["scenario_count"] == 2
    assert summary["closed_count"] == 2
    assert summary["zero_x"]["median"] == pytest.approx(0.0)
    assert summary["range_min"]["median"] == pytest.approx(-0.075)
    assert summary["range_max"]["median"] == pytest.approx(0.125)
    assert summary["theoretical_positive_rate"] == pytest.approx(0.5)
    assert summary["cost_coverage_rate"] == pytest.approx(0.5)


def test_templates_include_trade_path_and_pair_scenario_charts():
    assert 'id="tradeReturnChart"' in PAIR_TEMPLATE
    assert "buildSelectedTradePath" in PAIR_TEMPLATE
    assert "realizedCosts" in PAIR_TEMPLATE
    assert "investedNotional" in PAIR_TEMPLATE
    assert 'id="pairScenarioChart"' in OVERVIEW_TEMPLATE
    assert "drawPairScenarioChart" in OVERVIEW_TEMPLATE
    assert "X不变时理论收益" in PAIR_TEMPLATE
