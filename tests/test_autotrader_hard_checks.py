import unittest
from decimal import Decimal
from src.autotrader.hard_checks import (
    AccountState, HardCheckRejected, HardLimits, OrderRequest,
    validate_order, validate_watchlist, reject_cash_transfer,
)

def state(**kw):
    base = dict(confirmed_positions={}, pending_sell_qty={}, pending_buy_notional=0,
                open_position_cost=0, daily_entries=0, available_cash=10000, reconciled=True)
    base.update(kw)
    return AccountState(**base)

def order(side="BUY", symbol="SBIN", qty=1, price=800, product="LONG"):
    return OrderRequest(symbol, side, qty, Decimal(str(price)), product)

class HardChecksTests(unittest.TestCase):
    def blocked(self, request, account=None, limits=HardLimits()):
        with self.assertRaises(HardCheckRejected):
            validate_order(request, account or state(), limits)

    def test_no_sell_first(self):
        self.blocked(order("SELL"))
    def test_sell_only_owned_qty(self):
        validate_order(order("SELL", qty=3), state(confirmed_positions={"SBIN":3}))
        self.blocked(order("SELL", qty=4), state(confirmed_positions={"SBIN":3}))
        self.blocked(order("SELL", qty=2), state(confirmed_positions={"SBIN":3},pending_sell_qty={"SBIN":2}))
    def test_no_over_budget(self):
        self.blocked(order(qty=13, price=800))
        self.blocked(order(qty=2,price=800), state(open_position_cost=9000))
        self.blocked(order(qty=2,price=800), state(pending_buy_notional=9000))
    def test_no_new_trade_when_fully_allocated(self):
        self.blocked(order(), state(open_position_cost=10000,available_cash=0))
    def test_no_more_than_five_entries(self):
        self.blocked(order(), state(daily_entries=5))
    def test_three_positions_max(self):
        self.blocked(order(), state(confirmed_positions={"A":1,"B":1,"C":1}))
    def test_intraday_not_enabled(self):
        self.blocked(order(product="INTRADAY"))
        self.blocked(order(), limits=HardLimits(product="INTRADAY"))
    def test_reconciliation_required(self):
        self.blocked(order(), state(reconciled=False))
    def test_no_margin(self):
        self.blocked(order(qty=2), state(available_cash=1000))
    def test_watchlist(self):
        self.assertEqual(validate_watchlist(["SBIN","BEL","ITC"]),("SBIN","BEL","ITC"))
        with self.assertRaises(HardCheckRejected):
            validate_watchlist(["SBIN","BEL","ITC","INFY"])
    def test_no_transfers(self):
        with self.assertRaises(HardCheckRejected):
            reject_cash_transfer()
    def test_halt(self):
        self.blocked(order(),state(trading_enabled=False))
    def test_stop_loss_exit_blocks_buy_but_allows_exit(self):
        self.blocked(order(), state(stop_loss_exit_pending=True))
        validate_order(order("SELL"), state(stop_loss_exit_pending=True, confirmed_positions={"SBIN":1}))
    def test_daily_loss_halt_blocks_buy_but_allows_exit(self):
        self.blocked(order(), state(daily_loss_limit_reached=True))
        validate_order(order("SELL"), state(daily_loss_limit_reached=True, confirmed_positions={"SBIN":1}))
    def test_valid_buy(self):
        validate_order(order(),state())

if __name__ == "__main__":
    unittest.main()
