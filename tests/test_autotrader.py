import unittest
from decimal import Decimal
from src.autotrader.core import make_trade_plan
from src.autotrader.paper import PaperBroker
from src.autotrader.hard_checks import AccountState

class AutoTraderTests(unittest.TestCase):
    def test_fixed_two_to_one(self):
        p = make_trade_plan("SBIN", "800", "790")
        self.assertEqual(p.take_profit, Decimal("820"))
        self.assertEqual(p.quantity, 10)
        self.assertEqual(p.risk_rupees, Decimal("100"))
        self.assertEqual(PaperBroker().enter(p, AccountState(reconciled=True,available_cash=10000)).mode, "PAPER")
    def test_invalid_stop(self):
        with self.assertRaises(ValueError):
            make_trade_plan("SBIN", "800", "810")
    def test_unaffordable(self):
        with self.assertRaises(ValueError):
            make_trade_plan("SBIN", "20000", "19990")
    def test_paper_requires_reconciliation(self):
        with self.assertRaises(ValueError):
            PaperBroker().enter(make_trade_plan("SBIN","800","790"),AccountState())

if __name__ == "__main__":
    unittest.main()
