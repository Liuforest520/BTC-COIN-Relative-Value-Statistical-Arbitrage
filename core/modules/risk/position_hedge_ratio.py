from core.modules.models import OrderAction, OrderSide, PositionLot
from core.modules.risk.base import BaseRisk, RiskResult
from core.modules.risk.utils import bar_price, group_by_id, split_long_short, trade_notional, value_of


class PositionHedgeRatioRisk(BaseRisk):
    def __init__(self, hedge_method="dollar_neutral", max_deviation=0.05):
        super().__init__()
        self.hedge_method = hedge_method
        self.max_deviation = max_deviation
        self.position_lots = {}
        self.target_long_value = 0.0
        self.target_short_value = 0.0

    def check(self, trades=None, positions=None, bars=None, **kwargs):
        if trades:
            self._update_lots(trades)
            if self._has_close_trades(trades) and self._positions_flat(positions):
                self._clear_lots()

        if not positions:
            return RiskResult(True, self.name, "no positions")

        expected = self._expected_ratio()
        if expected is None:
            return RiskResult(True, self.name, "no open lots")

        actual = self._actual_ratio(positions, bars)
        if actual is None:
            return RiskResult(True, self.name, "flat")

        deviation = abs(actual - expected) / expected if expected else 0.0
        if deviation > self.max_deviation:
            message = (
                f"position hedge ratio deviation {deviation:.4f} > {self.max_deviation:.4f} "
                f"actual={actual:.4f} target={expected:.4f}"
            )
            return RiskResult(False, self.name, message, True)

        return RiskResult(True, self.name, f"{self.hedge_method} position hedge ratio")

    def _update_lots(self, trades):
        for group_id, group in group_by_id(trades).items():
            actions = {value_of(getattr(trade, "action", None)) for trade in group}
            if OrderAction.CLOSE.value in actions:
                self._close_lots(group)
            if OrderAction.OPEN.value in actions:
                lot = self._open_lot(group_id, group)
                if lot is not None:
                    self.position_lots.setdefault(lot.position_id, []).append(lot)
                    self.target_long_value += lot.target_ratio * lot.short_value
                    self.target_short_value += lot.short_value

    def _open_lot(self, group_id, group):
        position_id = self._position_id(group)
        if position_id is None:
            return None

        long_value, short_value = split_long_short(group, trade_notional)
        if long_value is None or short_value is None or long_value <= 0 or short_value <= 0:
            return None

        target_ratio = self._target_ratio(group, long_value, short_value)
        return PositionLot(
            position_id=position_id,
            group_id=group_id,
            long_value=float(long_value),
            short_value=float(short_value),
            target_ratio=float(target_ratio),
        )

    def _close_lots(self, group):
        position_id = self._position_id(group)
        if position_id is None:
            return

        lots = self.position_lots.get(position_id, [])
        if not lots:
            return

        close_short_value, close_long_value = split_long_short(group, trade_notional)
        if close_long_value is None or close_short_value is None:
            return

        remaining_long_to_close = float(close_long_value)
        remaining_short_to_close = float(close_short_value)
        remaining_lots = []

        for lot in lots:
            long_fraction = remaining_long_to_close / lot.long_value if lot.long_value > 0 else 0.0
            short_fraction = remaining_short_to_close / lot.short_value if lot.short_value > 0 else 0.0
            close_fraction = min(1.0, max(long_fraction, short_fraction))

            if close_fraction <= 0:
                remaining_lots.append(lot)
                continue

            closed_long_value = lot.long_value * close_fraction
            closed_short_value = lot.short_value * close_fraction
            self.target_long_value -= lot.target_ratio * closed_short_value
            self.target_short_value -= closed_short_value

            remaining_long_to_close = max(remaining_long_to_close - closed_long_value, 0.0)
            remaining_short_to_close = max(remaining_short_to_close - closed_short_value, 0.0)
            lot.long_value -= closed_long_value
            lot.short_value -= closed_short_value

            if lot.long_value > 1e-12 and lot.short_value > 1e-12:
                remaining_lots.append(lot)

        if remaining_lots:
            self.position_lots[position_id] = remaining_lots
        else:
            self.position_lots.pop(position_id, None)

        self.target_long_value = max(self.target_long_value, 0.0)
        self.target_short_value = max(self.target_short_value, 0.0)

    def _target_ratio(self, group, long_value, short_value):
        ratios = [
            float(getattr(trade, "target_hedge_ratio"))
            for trade in group
            if getattr(trade, "target_hedge_ratio", None) is not None
        ]
        if ratios and ratios[0] > 0:
            return ratios[0]
        return long_value / short_value

    def _expected_ratio(self):
        if self.target_long_value <= 0 or self.target_short_value <= 0:
            return None
        return self.target_long_value / self.target_short_value

    def _actual_ratio(self, positions, bars):
        long_value = 0.0
        short_value = 0.0

        for exchange, exchange_positions in positions.items():
            for symbol, quantity in exchange_positions.items():
                price = bar_price(bars, exchange, symbol, "close")
                if price is None:
                    continue

                value = float(quantity) * float(price)
                if value > 0:
                    long_value += value
                elif value < 0:
                    short_value += abs(value)

        if long_value <= 0 and short_value <= 0:
            return None
        if long_value <= 0 or short_value <= 0:
            return float("inf")
        return long_value / short_value

    def _position_id(self, group):
        for item in group:
            position_id = getattr(item, "position_id", None)
            if position_id is not None:
                return position_id
        return None

    def _has_close_trades(self, trades):
        return any(value_of(getattr(trade, "action", None)) == OrderAction.CLOSE.value for trade in trades or [])

    def _positions_flat(self, positions):
        for exchange_positions in (positions or {}).values():
            for quantity in exchange_positions.values():
                if abs(float(quantity)) > 1e-12:
                    return False
        return True

    def _clear_lots(self):
        self.position_lots.clear()
        self.target_long_value = 0.0
        self.target_short_value = 0.0
