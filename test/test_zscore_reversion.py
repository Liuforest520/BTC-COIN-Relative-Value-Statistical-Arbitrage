from core.modules.models.pipeline_types import SignalState
from core.modules.models.pipeline_types import EstimatorOutput
from core.modules.signals.zscore_reversion import ZScoreReversionSignal


def test_two_stage_resets_when_reversion_crosses_zero():
    signal = ZScoreReversionSignal(
        pair_id="test_pair",
        entry_z=2.0,
        exit_z=0.8,
        two_stage_enabled=True,
        two_stage_trigger_z=2.0,
        reversion_filter_enabled=False,
    )
    state = SignalState(
        entry_z_upper=2.0,
        entry_z_lower=-2.0,
        armed_side="long_x",
        armed_bar_index=10,
        armed_zscore=3.0,
        armed_trigger_z=2.0,
        armed_entry_z=2.0,
    )

    out = signal._two_stage_fire_check(state, zscore=-1.2, bar_index=11)

    assert out.action == "none"
    assert out.side is None
    assert "crossed zero" in out.reason
    assert state.armed_side is None


def test_two_stage_fires_before_zero_for_armed_long_x():
    signal = ZScoreReversionSignal(
        pair_id="test_pair",
        entry_z=2.0,
        exit_z=0.8,
        two_stage_enabled=True,
        two_stage_trigger_z=2.0,
        reversion_filter_enabled=False,
    )
    state = SignalState(
        entry_z_upper=2.0,
        entry_z_lower=-2.0,
        armed_side="long_x",
        armed_bar_index=10,
        armed_zscore=3.0,
        armed_trigger_z=2.0,
        armed_entry_z=2.0,
    )

    out = signal._two_stage_fire_check(state, zscore=1.5, bar_index=11)

    assert out.action == "open"
    assert out.side == "long_x"


def test_two_stage_fires_inside_entry_window():
    signal = ZScoreReversionSignal(
        pair_id="test_pair",
        entry_z=2.0,
        exit_z=0.5,
        two_stage_enabled=True,
        two_stage_trigger_z=3.0,
        two_stage_entry_z=2.0,
        two_stage_entry_window_z=0.2,
        reversion_filter_enabled=False,
    )
    state = SignalState(
        entry_z_upper=2.0,
        entry_z_lower=-2.0,
        armed_side="long_x",
        armed_bar_index=10,
        armed_zscore=3.1,
        armed_trigger_z=3.0,
        armed_entry_z=2.0,
    )

    out = signal._two_stage_fire_check(state, zscore=1.9, bar_index=11)

    assert out.action == "open"
    assert out.side == "long_x"


def test_two_stage_resets_after_missing_entry_window():
    signal = ZScoreReversionSignal(
        pair_id="test_pair",
        entry_z=2.0,
        exit_z=0.5,
        two_stage_enabled=True,
        two_stage_trigger_z=3.0,
        two_stage_entry_z=2.0,
        two_stage_entry_window_z=0.2,
        reversion_filter_enabled=False,
    )
    state = SignalState(
        entry_z_upper=2.0,
        entry_z_lower=-2.0,
        armed_side="long_x",
        armed_bar_index=10,
        armed_zscore=3.1,
        armed_trigger_z=3.0,
        armed_entry_z=2.0,
    )

    out = signal._two_stage_fire_check(state, zscore=1.2, bar_index=11)

    assert out.action == "none"
    assert out.side is None
    assert "missed entry window" in out.reason
    assert state.armed_side is None


def test_two_stage_resets_after_expiry():
    signal = ZScoreReversionSignal(
        pair_id="test_pair",
        entry_z=2.0,
        exit_z=0.5,
        two_stage_enabled=True,
        two_stage_trigger_z=3.0,
        two_stage_entry_z=2.0,
        two_stage_max_wait_bars=5,
        reversion_filter_enabled=False,
    )
    state = SignalState(
        entry_z_upper=2.0,
        entry_z_lower=-2.0,
        armed_side="long_x",
        armed_bar_index=10,
        armed_zscore=3.2,
        armed_trigger_z=3.0,
        armed_entry_z=2.0,
    )

    out = signal._two_stage_fire_check(state, zscore=1.9, bar_index=16)

    assert out.action == "none"
    assert out.side is None
    assert "expired" in out.reason
    assert state.armed_side is None


def test_pair_quality_filter_blocks_low_correlation_entry():
    signal = ZScoreReversionSignal(
        pair_id="test_pair",
        entry_z=2.0,
        exit_z=0.5,
        pair_quality_filter_enabled=True,
        pair_quality_min_samples=10,
        pair_quality_min_corr=0.5,
        cost_filter_enabled=False,
        reversion_filter_enabled=False,
    )
    state = SignalState(entry_z_upper=2.0, entry_z_lower=-2.0)
    estimator = EstimatorOutput(
        pair_id="test_pair",
        ready=True,
        spread_mean=0.0,
        spread_std=1.0,
        latest_spread=2.2,
    )

    out = signal.evaluate(
        state,
        estimator,
        bar_index=1,
        para={"pair_quality": {"ready": True, "samples": 100, "return_corr": 0.2}},
    )

    assert out.action == "none"
    assert "pair quality corr" in out.reason


def test_cost_filter_blocks_small_expected_move():
    signal = ZScoreReversionSignal(
        pair_id="test_pair",
        entry_z=2.0,
        exit_z=1.9,
        cost_filter_enabled=True,
        cost_fee_rate=0.0005,
        cost_slippage_bps=1,
        cost_buffer_multiplier=2.0,
        reversion_filter_enabled=False,
    )
    state = SignalState(entry_z_upper=2.0, entry_z_lower=-2.0)
    estimator = EstimatorOutput(
        pair_id="test_pair",
        ready=True,
        spread_mean=0.0,
        spread_std=0.01,
        latest_spread=0.021,
    )

    out = signal.evaluate(state, estimator, bar_index=1)

    assert out.action == "none"
    assert "expected move" in out.reason
