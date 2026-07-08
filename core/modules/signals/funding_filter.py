from core.modules.signals.cointegration import CointegrationZScoreSignal
from core.modules.logger import logger


class FundingFilterZScoreSignal(CointegrationZScoreSignal):
    def __init__(self, funding_spread_threshold=0.0001, **kwargs):
        super().__init__(**kwargs)
        self.funding_spread_threshold = funding_spread_threshold
        self.last_funding_rates = {}
        self._logged_waiting_funding = False

    def on_funding_rates(self, funding_rates):
        if funding_rates:
            self.last_funding_rates.update(funding_rates)

    def _entry_filter(self, side, zscore):
        x_rate = self.last_funding_rates.get(self.x_symbol)
        y_rate = self.last_funding_rates.get(self.y_symbol)
        if x_rate is None or y_rate is None:
            self._funding_filter_reason = "waiting for first funding rate data"
            if not self._logged_waiting_funding:
                logger.info("funding filter waiting for first funding rate data")
                self._logged_waiting_funding = True
            return False

        funding_spread = y_rate - x_rate
        self._funding_filter_reason = f"funding_spread={funding_spread:.8f}"
        if side == "short_spread":
            return funding_spread >= self.funding_spread_threshold
        if side == "long_spread":
            return funding_spread <= -self.funding_spread_threshold
        return False

    def _record_decision(self, state, decision):
        reason = getattr(self, "_funding_filter_reason", None)
        if reason and decision.action == "none":
            state["funding_filter_reason"] = reason
        return super()._record_decision(state, decision)
