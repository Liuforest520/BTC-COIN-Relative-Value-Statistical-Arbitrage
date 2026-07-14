from core.modules.risk.base import RiskResult
from core.modules.risk.dollar_neutral import DollarNeutralRisk
from core.modules.risk.hedge_ratio import HedgeRatioRisk
from core.modules.risk.pair_completeness import PairCompletenessRisk
from core.modules.risk.position_hedge_ratio import PositionHedgeRatioRisk


class RiskManager:
    def __init__(self, pre_trade_risks=None, post_trade_risks=None):
        self.pre_trade_risks = pre_trade_risks or [PairCompletenessRisk(), DollarNeutralRisk()]
        self.post_trade_risks = post_trade_risks or [PairCompletenessRisk(), PositionHedgeRatioRisk()]
        self.history = []
        self.block_open_orders = False

    @property
    def block_new_orders(self):
        return self.block_open_orders

    def check_orders(self, orders, bars):
        if self.block_open_orders and self._has_open_orders(orders):
            result = RiskResult(False, "RiskManager", "open orders blocked", True, stage="pre_trade")
            self.history.append(result)
            return [result]

        results = []
        for risk in self.pre_trade_risks:
            results.append(risk(orders=orders, bars=bars))

        self._set_stage(results, "pre_trade")
        self._update_state(results)
        return results

    def check_trades(self, trades, positions, bars):
        results = []
        for risk in self.post_trade_risks:
            results.append(risk(trades=trades, positions=positions, bars=bars))

        self._set_stage(results, "post_trade")
        self._update_state(results)
        self._recover_if_flat(positions)
        return results

    def passed(self, results):
        return all(result.passed for result in results)

    def _update_state(self, results):
        self.history.extend(results)
        if any(result.block_new_orders and not result.passed for result in results):
            self.block_open_orders = True

    def _set_stage(self, results, stage):
        for result in results:
            result.stage = stage

    def _has_open_orders(self, orders):
        for order in orders or []:
            action = getattr(order.action, "value", order.action)
            if action == "open":
                return True
        return False

    def _recover_if_flat(self, positions):
        if self.block_open_orders and self._is_flat(positions):
            self.block_open_orders = False

    def _is_flat(self, positions):
        for exchange_positions in (positions or {}).values():
            for quantity in exchange_positions.values():
                if abs(float(quantity)) > 1e-12:
                    return False
        return True


def build_risk_manager(strategy_config, risk_config=None):
    risk_config = risk_config or {}
    hedge_method = strategy_config.hedge_method
    order_tolerance = float(risk_config.get("order_hedge_ratio_tolerance", 0.02))
    position_tolerance = float(risk_config.get("position_hedge_ratio_tolerance", 0.05))

    if hedge_method in ["fixed_notional", "dollar_neutral"]:
        hedge_risk = DollarNeutralRisk(max_deviation=order_tolerance)
    else:
        hedge_risk = HedgeRatioRisk(hedge_method=hedge_method, max_deviation=order_tolerance)

    return RiskManager(
        pre_trade_risks=[
            PairCompletenessRisk(),
            hedge_risk,
        ],
        post_trade_risks=[
            PairCompletenessRisk(),
            PositionHedgeRatioRisk(hedge_method=hedge_method, max_deviation=position_tolerance),
        ],
    )
