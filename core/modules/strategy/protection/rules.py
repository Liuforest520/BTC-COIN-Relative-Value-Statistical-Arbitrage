from __future__ import annotations
from core.modules.strategy.position_protection import mark_net_pnl

class PairLossStopRule:
    name, priority = "pair_loss_stop", 20

    def evaluate(self, c):
        state, sizing, ledger, cfg = c.state, c.sizing_state, c.ledger, c.config
        bundle = c.bundle
        accounting = ledger if ledger.active else state
        if not (cfg.enabled and cfg.pair_loss_stop_enabled and accounting.active and bundle and bundle.x_bar and bundle.y_bar and accounting.entry_x_price and accounting.entry_y_price and state.pair_loss_stop_return is not None):
            return None
        qx = accounting.entry_x_quantity or sizing.x_quantity
        qy = accounting.entry_y_quantity or sizing.y_quantity
        gross = accounting.entry_gross_notional or (float(accounting.entry_x_price) * qx + float(accounting.entry_y_price) * qy)
        pnl = mark_net_pnl(accounting.side, float(bundle.x_bar.close), float(bundle.y_bar.close), accounting.entry_x_price, accounting.entry_y_price, qx, qy, accounting.entry_fee, accounting.funding_cost, c.fee_rate, c.slippage_rate)
        net_return = pnl / max(gross, 1e-12)
        if net_return > -float(state.pair_loss_stop_return):
            return None
        return ProtectionDecision(True, self.name, "protective_pair_loss_stop", "stop_loss", self.name, int(getattr(state, "pair_loss_stop_freeze_bars", 0)), _wait(cfg), {"net_pnl": pnl, "net_return": net_return, "threshold": -float(state.pair_loss_stop_return)})

from core.modules.strategy.protection.base import ProtectionDecision


def _wait(cfg):
    value = getattr(cfg, "wait_for_model_update_after_non_z_exit", None)
    return True if value is None else bool(value)
