import unittest
from decimal import Decimal
from src.autotrader.core import make_trade_plan
from src.autotrader.paper import PaperBroker

class AutoTraderTests(unittest.TestCase):
    def test_fixed_two_to_one(self):
        p = make_trade_plan("SBIN", "800", "790")
        self.assertEqual(p.take_profit, Decimal("820"))
        self.assertEqual(p.quantity, 10)
        self.assertEqual(p.risk_rupees, Decimal("100"))
        self.assertEqual(PaperBroker().enter(p).mode, "PAPER")

    def test_invalid_stop(self):
        with self.assertRaises(ValueError):
            make_trade_plan("SBIN", "800", "810")

    def test_unaffordable(self):
        with self.assertRaises(ValueError):
            make_trade_plan("SBIN", "20000", "19990")

if __name__ == "__main__":
    unittest.main()
