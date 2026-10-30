import unittest
from decimal import Decimal
from unittest.mock import patch
from src.autotrader.learning import Bar,evaluate_setup,summarize
from src.autotrader.dhan_live_data import DhanReadOnlyFeed
from src.autotrader.instruments import InstrumentCatalog

def bar(day,o,h,l,c):
    return Bar("2026-10-%02dT09:15:00+05:30"%day,*map(Decimal,map(str,(o,h,l,c))))

class LearningTests(unittest.TestCase):
    def test_stop_wins_ambiguous_bar(self):
        bars=[bar(1,100,101,99,100),bar(2,100,105,95,101)]
        r=evaluate_setup("SBIN","TEST",0,bars,100,98,104)
        self.assertEqual(r["outcome"],"STOP")
        self.assertEqual(r["r_multiple"],"-1")
    def test_gap_slippage(self):
        bars=[bar(1,100,101,99,100),bar(2,90,94,89,92)]
        r=evaluate_setup("SBIN","TEST",0,bars,100,98,104)
        self.assertEqual(r["outcome"],"STOP_GAP")
        self.assertEqual(r["r_multiple"],"-5")
    def test_incomplete_not_counted(self):
        bars=[bar(1,100,101,99,100),bar(2,100,101,99,100)]
        r=evaluate_setup("SBIN","TEST",0,bars,100,98,104,horizon=5)
        self.assertFalse(r["complete"])
        self.assertEqual(summarize([r])["complete"],0)
    def test_no_lookahead(self):
        bars=[bar(1,100,101,99,100)]
        with self.assertRaises(ValueError):
            evaluate_setup("SBIN","TEST",0,bars,100,98,104)
    def test_live_requires_verified_master(self):
        feed=DhanReadOnlyFeed(InstrumentCatalog(),client_id="test",access_token="test")
        with self.assertRaises(ValueError):
            feed.snapshot(["SBIN"])
