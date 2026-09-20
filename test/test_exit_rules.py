import math

import pytest

from core.modules.signals.exit_rules import resolve_position_exit_decision


def _position(side: str | None = None, has_position: bool = True) -> dict:
    return {"position": {"has_position": has_position, "side": side}}


@pytest.mark.parametrize("zscore", [-0.5, 0.0, 0.5])
def test_positive_exit_threshold_closes_inside_symmetric_band(zscore):
    should_exit, reason = resolve_position_exit_decision(
        _position("long_x"), zscore, 0.5
    )

    assert should_exit is True
    assert reason == "within_exit_band"


@pytest.mark.parametrize(
    ("side", "zscore"),
    [("long_x", 0.1), ("short_x", -0.1)],
)
def test_zero_exit_threshold_waits_for_direction_reversal(side, zscore):
    assert resolve_position_exit_decision(_position(side), zscore, 0.0) == (
        False,
        None,
    )


@pytest.mark.parametrize(
    ("side", "zscore"),
    [("long_x", 0.0), ("long_x", -0.1), ("short_x", 0.0), ("short_x", 0.1)],
)
def test_zero_exit_threshold_closes_after_direction_reversal(side, zscore):
    should_exit, reason = resolve_position_exit_decision(
        _position(side), zscore, 0.0
    )

    assert should_exit is True
    assert reason == "direction_reversed"


@pytest.mark.parametrize(
    ("side", "zscore", "expected"),
    [
        ("long_x", -0.49, False),
        ("long_x", -0.5, True),
        ("short_x", 0.49, False),
        ("short_x", 0.5, True),
    ],
)
def test_negative_exit_threshold_requires_opposite_side_distance(side, zscore, expected):
    should_exit, reason = resolve_position_exit_decision(
        _position(side), zscore, -0.5
    )

    assert should_exit is expected
    assert reason == ("reversal_threshold_reached" if expected else None)


@pytest.mark.parametrize("zscore", [None, math.nan, math.inf, -math.inf])
def test_invalid_zscore_never_closes(zscore):
    assert resolve_position_exit_decision(_position("long_x"), zscore, 0.5) == (
        False,
        None,
    )


def test_directional_exit_requires_an_open_position():
    assert resolve_position_exit_decision(
        _position("long_x", has_position=False), -1.0, 0.0
    ) == (False, None)
