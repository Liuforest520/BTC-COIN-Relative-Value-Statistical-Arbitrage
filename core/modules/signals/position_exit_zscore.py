"""Position-period exit z-score calculations for return regressions."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt


VALID_POSITION_EXIT_ZSCORE_METHODS = {
    "standard",
    "sum_zscore",
    "cumulative_return",
}


def resolve_position_exit_decision(
    para: dict | None,
    exit_zscore: float,
    exit_z: float,
) -> tuple[bool, str | None]:
    """Close inside the exit band or after crossing to the opposite Z side."""
    position = ((para or {}).get("position") or {})
    value = float(exit_zscore)
    threshold = abs(float(exit_z))

    if abs(value) <= threshold:
        return True, "within_exit_band"

    if not bool(position.get("has_position", False)):
        return False, None

    side = position.get("side")
    if side == "long_x" and value <= 0.0:
        return True, "direction_reversed"
    if side == "short_x" and value >= 0.0:
        return True, "direction_reversed"
    return False, None


def resolve_position_exit_zscore(para: dict | None, standard_zscore: float):
    position = ((para or {}).get("position") or {})
    has_position = bool(position.get("has_position", False))
    method = str(position.get("exit_zscore_method", "standard"))
    value = position.get("exit_zscore")
    if (
        has_position
        and method != "standard"
        and value is not None
        and isfinite(float(value))
    ):
        return float(value), method, True
    return float(standard_zscore), "standard", has_position


@dataclass(frozen=True)
class ReturnObservation:
    bar_index: int
    x_return: float
    y_return: float
    alpha: float
    beta: float
    residual_mean: float
    residual_std: float

    @property
    def zscore(self) -> float:
        residual = self.y_return - self.alpha - self.beta * self.x_return
        return (residual - self.residual_mean) / self.residual_std


class PositionExitZScoreTracker:
    """Track an exit statistic from the return that triggered the open."""

    def __init__(self, method: str = "standard"):
        method = str(method or "standard").lower()
        if method not in VALID_POSITION_EXIT_ZSCORE_METHODS:
            choices = ", ".join(sorted(VALID_POSITION_EXIT_ZSCORE_METHODS))
            raise ValueError(f"position_exit_zscore_method must be one of: {choices}")
        self.method = method
        self.latest_observation: ReturnObservation | None = None
        self._state: dict | None = None

    @property
    def active(self) -> bool:
        return self._state is not None

    def on_bar(
        self,
        observation: ReturnObservation | None,
        has_position: bool,
    ) -> float | None:
        if observation is not None:
            self.latest_observation = observation

        if self.method == "standard":
            return None
        if not has_position:
            if self._state is not None:
                self._state = None
            return None
        if observation is None:
            return self.value

        if self._state is None:
            # Recovery fallback. The normal path initializes on the fill event.
            self._initialize(observation)
        elif observation.bar_index != self._state["last_bar_index"]:
            self._append(observation)
        return self.value

    def on_position_opened(self) -> None:
        if self.method == "standard" or self._state is not None:
            return
        if self.latest_observation is not None:
            self._initialize(self.latest_observation)

    def on_position_closed(self) -> None:
        self._state = None

    @property
    def value(self) -> float | None:
        if self._state is None:
            return None
        if self.method == "sum_zscore":
            return float(self._state["sum_zscore"])

        count = int(self._state["count"])
        cumulative_residual = (
            self._state["sum_y_return"]
            - count * self._state["alpha"]
            - self._state["beta"] * self._state["sum_x_return"]
        )
        centered = cumulative_residual - count * self._state["residual_mean"]
        return float(centered / (self._state["residual_std"] * sqrt(count)))

    @property
    def details(self) -> dict:
        if self._state is None:
            return {"method": self.method, "active": False, "zscore": None}
        return {
            "method": self.method,
            "active": True,
            "zscore": self.value,
            "entry_bar_index": self._state["entry_bar_index"],
            "last_bar_index": self._state["last_bar_index"],
            "return_count": self._state["count"],
            "entry_zscore": self._state["entry_zscore"],
            "sum_x_return": self._state["sum_x_return"],
            "sum_y_return": self._state["sum_y_return"],
        }

    def _initialize(self, observation: ReturnObservation) -> None:
        self._validate_observation(observation)
        entry_zscore = observation.zscore
        self._state = {
            "entry_bar_index": observation.bar_index,
            "last_bar_index": observation.bar_index,
            "count": 1,
            "alpha": observation.alpha,
            "beta": observation.beta,
            "residual_mean": observation.residual_mean,
            "residual_std": observation.residual_std,
            "entry_zscore": entry_zscore,
            "sum_zscore": entry_zscore,
            "sum_x_return": observation.x_return,
            "sum_y_return": observation.y_return,
        }

    def _append(self, observation: ReturnObservation) -> None:
        self._validate_observation(observation)
        state = self._state
        frozen_residual = (
            observation.y_return
            - state["alpha"]
            - state["beta"] * observation.x_return
        )
        frozen_zscore = (
            frozen_residual - state["residual_mean"]
        ) / state["residual_std"]
        state["sum_zscore"] += frozen_zscore
        state["sum_x_return"] += observation.x_return
        state["sum_y_return"] += observation.y_return
        state["count"] += 1
        state["last_bar_index"] = observation.bar_index

    @staticmethod
    def _validate_observation(observation: ReturnObservation) -> None:
        values = (
            observation.x_return,
            observation.y_return,
            observation.alpha,
            observation.beta,
            observation.residual_mean,
            observation.residual_std,
        )
        if not all(isfinite(value) for value in values) or observation.residual_std <= 0:
            raise ValueError("return observation requires finite values and positive residual_std")
