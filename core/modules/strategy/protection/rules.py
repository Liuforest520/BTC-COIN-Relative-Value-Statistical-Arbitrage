from __future__ import annotations
from .base import ProtectionContext, ProtectionDecision, ProtectionRule
from core.modules.strategy.position_protection import mark_net_pnl

class TheoreticalXStopRule(ProtectionRule):
    name, priority = "theoretical_x_stop", 10
    def evaluate(self, c):
        s, cfg = c.state, c.config
        b = c.bundle
        if not (cfg.enabled and cfg.stop_loss_enabled and s.stop_loss_x_price is not None and b and b.x_bar): return None
        x = float(b.x_bar.close); direction = s.stop_loss_direction or ("lower" if s.side == "long_x" else "upper")
        hit = x <= s.stop_loss_x_price if direction == "lower" else x >= s.stop_loss_x_price
        if not hit: return None
        return ProtectionDecision(True, self.name, "protective_stop_loss", "stop_loss", "stop_loss", int(getattr(cfg, "stop_loss_freeze_bars", 0)), _wait(cfg), {"x_price": x, "stop_loss_x_price": s.stop_loss_x_price, "direction": direction})

class PairLossStopRule(ProtectionRule):
    name, priority = "pair_loss_stop", 20
    def evaluate(self, c):
        s, z, ledger, cfg = c.state, c.sizing_state, c.ledger, c.config
        b = c.bundle
        accounting = ledger if ledger.active else s
        if not (cfg.enabled and cfg.pair_loss_stop_enabled and accounting.active and b and b.x_bar and b.y_bar and accounting.entry_x_price and accounting.entry_y_price and s.pair_loss_stop_return is not None): return None
        qx = accounting.entry_x_quantity or z.x_quantity
        qy = accounting.entry_y_quantity or z.y_quantity
        gross = accounting.entry_gross_notional or (float(accounting.entry_x_price) * qx + float(accounting.entry_y_price) * qy)
        pnl = mark_net_pnl(accounting.side, float(b.x_bar.close), float(b.y_bar.close), accounting.entry_x_price, accounting.entry_y_price, qx, qy, accounting.entry_fee, accounting.funding_cost, c.fee_rate, c.slippage_rate)
        ret = pnl / max(gross, 1e-12)
        if ret > -float(s.pair_loss_stop_return): return None
        return ProtectionDecision(True, self.name, "protective_pair_loss_stop", "stop_loss", self.name, int(getattr(s, "pair_loss_stop_freeze_bars", 0)), _wait(cfg), {"net_pnl": pnl, "net_return": ret, "threshold": -float(s.pair_loss_stop_return)})

class TakeProfitRule(ProtectionRule):
    name, priority = "take_profit", 30
    def evaluate(self, c):
        s, z, ledger, cfg = c.state, c.sizing_state, c.ledger, c.config
        b = c.bundle
        accounting = ledger if ledger.active else s
        if not (cfg.enabled and cfg.take_profit_enabled and accounting.active and b and b.x_bar and b.y_bar and accounting.entry_x_price and accounting.entry_y_price and s.take_profit_return is not None): return None
        qx = accounting.entry_x_quantity or z.x_quantity
        qy = accounting.entry_y_quantity or z.y_quantity
        gross = accounting.entry_gross_notional or (float(accounting.entry_x_price) * qx + float(accounting.entry_y_price) * qy)
        pnl = mark_net_pnl(accounting.side, float(b.x_bar.close), float(b.y_bar.close), accounting.entry_x_price, accounting.entry_y_price, qx, qy, accounting.entry_fee, accounting.funding_cost, c.fee_rate, c.slippage_rate)
        ret = pnl / max(gross, 1e-12)
        if ret < float(s.take_profit_return): return None
        return ProtectionDecision(True, self.name, "protective_take_profit", "take_profit", self.name, int(getattr(cfg, "take_profit_freeze_bars", 0)), False, {"net_pnl": pnl, "net_return": ret, "threshold": float(s.take_profit_return)})

class MaxHoldingTimeRule(ProtectionRule):
    name, priority = "max_holding_time", 40
    def evaluate(self, c):
        s, cfg = c.state, c.config
        if not (cfg.max_holding_time_enabled and s.max_holding_deadline_bar is not None): return None
        if int(getattr(c.pipeline.state, "last_bar_index", 0)) < int(s.max_holding_deadline_bar): return None
        return ProtectionDecision(True, self.name, "protective_max_holding_time", "stop_loss", self.name, int(getattr(cfg, "max_holding_time_freeze_bars", 0)), _wait(cfg), {"deadline_bar": s.max_holding_deadline_bar})

def _wait(cfg):
    value = getattr(cfg, "wait_for_model_update_after_non_z_exit", None)
    return True if value is None else bool(value)
