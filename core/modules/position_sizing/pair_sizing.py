class PairSizing:
    def __init__(
        self,
        method="fixed_notional",
        notional=10000.0,
        equity=100000.0,
        gross_exposure_ratio=0.2,
        max_gross_exposure_ratio=1.0,
        min_notional=0.0,
    ):
        self.method = method
        self.notional = float(notional)
        self.equity = float(equity)
        self.gross_exposure_ratio = float(gross_exposure_ratio)
        self.max_gross_exposure_ratio = float(max_gross_exposure_ratio)
        self.min_notional = float(min_notional)

    def notionals(self, bars, x_leg, y_leg, hedge_ratio=1.0, x_vol=None, y_vol=None):
        if self.method == "fixed_notional":
            x_notional = self.notional
            y_notional = self.notional
        elif self.method == "equity_ratio":
            gross_notional = self._gross_notional()
            x_notional = gross_notional / 2
            y_notional = gross_notional / 2
        elif self.method == "beta_neutral":
            gross_notional = self._gross_notional()
            beta = abs(float(hedge_ratio)) if hedge_ratio else 1.0
            x_notional = gross_notional * beta / (1 + beta)
            y_notional = gross_notional / (1 + beta)
        elif self.method == "volatility_neutral":
            gross_notional = self._gross_notional()
            x_risk = 1 / x_vol if x_vol and x_vol > 0 else 1.0
            y_risk = 1 / y_vol if y_vol and y_vol > 0 else 1.0
            base = x_risk + y_risk
            x_notional = gross_notional * x_risk / base
            y_notional = gross_notional * y_risk / base
        else:
            raise ValueError(f"unsupported sizing method: {self.method}")

        x_notional = max(float(x_notional), self.min_notional)
        y_notional = max(float(y_notional), self.min_notional)
        return {
            "x_notional": x_notional,
            "y_notional": y_notional,
            "x_quantity": x_notional / self._open_price(bars, x_leg),
            "y_quantity": y_notional / self._open_price(bars, y_leg),
        }

    def _gross_notional(self):
        gross_notional = self.equity * self.gross_exposure_ratio
        max_gross_notional = self.equity * self.max_gross_exposure_ratio
        return min(gross_notional, max_gross_notional)

    def _open_price(self, bars, leg):
        exchange, symbol = leg
        return float(bars[exchange][symbol][1])
