import unittest
from decimal import Decimal
from src.autotrader.learning import Bar
from src.autotrader.batch_research import filter_records,group_report,run
from pathlib import Path
from unittest.mock import patch

def bar(day,close,volume):
    c=Decimal(str(close))
    return Bar(f"2026-01-{day:02d}T09:15:00+05:30",c,c+1,c-1,c,volume)

class BatchResearchTests(unittest.TestCase):
    def test_filters_use_signal_and_past_only(self):
        bars=[bar(i,100+i,100) for i in range(1,8)]
        records=[dict(signal_at=bars[5].timestamp,status="COMPLETE",complete=True,
                      entry="106",stop="105",exit_price="108",exit_at=bars[6].timestamp,
                      r_multiple="2")]
        a=filter_records(bars,records,ema_period=3,volume_period=3)
        self.assertTrue(a[0]["ema_trend_pass"])
        self.assertFalse(a[0]["volume_breakout_pass"])
        bars[-1]=bar(7,100000,999999)
        self.assertEqual(a,filter_records(bars,records,ema_period=3,volume_period=3))
    def test_symbol_from_market_1d_folder(self):
        with patch("src.autotrader.batch_research.load_parquet",return_value=[]):
            result,_=run([Path("data/market/GRASIM/1D/ohlc.parquet")],"2026-04-01")
        self.assertIn("GRASIM",result["symbols"])
        self.assertNotIn("1D",result["symbols"])
    def test_cost_reduces_r(self):
        records=[dict(signal_at="2026-02-02",exit_at="2026-02-03",status="COMPLETE",
                      complete=True,entry="100",stop="98",exit_price="104",
                      r_multiple="2",ema_trend_pass=True,volume_breakout_pass=True)]
        raw=group_report(records,"2026-02-01",fee_bps="0")["BASE"]["unseen"]["mean_r"]
        net=group_report(records,"2026-02-01",fee_bps="10")["BASE"]["unseen"]["mean_r"]
        self.assertLess(Decimal(net),Decimal(raw))
