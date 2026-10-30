import unittest
from decimal import Decimal
from src.autotrader.learning import Bar
from src.autotrader.historical import detect_setups,walkforward

def b(i,price,high=None,low=None):
    p=Decimal(str(price))
    return Bar(f"2026-01-{i:02d}T09:15:00+05:30",p,Decimal(str(high if high is not None else price+1)),Decimal(str(low if low is not None else price-1)),p,100)

class HistoricalTests(unittest.TestCase):
    def test_breakout_only_after_prior_high(self):
        bars=[b(1,100),b(2,101),b(3,100),b(4,105),b(5,106,107,104),b(6,108,109,107)]
        result=detect_setups("SBIN",bars,lookback=2,horizon=2)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]["signal_at"],bars[3].timestamp)
        self.assertEqual(result[0]["status"],"COMPLETE")
    def test_no_future_bar_does_not_enter(self):
        bars=[b(1,100),b(2,101),b(3,100),b(4,105)]
        self.assertEqual(detect_setups("SBIN",bars,lookback=2),[])
    def test_walkforward_excludes_crossing_outcome(self):
        records=[dict(signal_at="2026-01-01",exit_at="2026-02-01",status="COMPLETE",complete=True,r_multiple="2"),
                 dict(signal_at="2026-02-02",exit_at="2026-02-03",status="COMPLETE",complete=True,r_multiple="-1")]
        result=walkforward(records,"2026-02-01")
        self.assertEqual(result["train"]["complete"],0)
        self.assertEqual(result["unseen"]["complete"],1)
