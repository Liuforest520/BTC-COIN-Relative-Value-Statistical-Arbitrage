from dataclasses import dataclass


@dataclass
class PositionLot:
    position_id: str
    group_id: str
    long_value: float
    short_value: float
    target_ratio: float
