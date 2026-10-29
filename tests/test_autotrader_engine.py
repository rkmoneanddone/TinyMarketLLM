import tempfile
import unittest
from pathlib import Path
from src.autotrader.engine import AutoTrader
from src.autotrader.hard_checks import HardCheckRejected

class EndToEndTests(unittest.TestCase):
    def test_buy_target_report_restart(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"ledger.json"
            e=AutoTrader(journal=path)
            with self.assertRaises(HardCheckRejected):e.sell("SBIN",800)
            p=e.buy("SBIN",800,790)
            self.assertEqual(p["target"],"820")
            self.assertEqual(p["quantity"],10)
            with self.assertRaises((ValueError,HardCheckRejected)):e.buy("BEL",2500,2475)
            restored=AutoTrader(journal=path)
            self.assertEqual(restored.positions["SBIN"]["quantity"],10)
            t=restored.on_quote("SBIN",820)
            self.assertEqual(t["reason"],"TAKE_PROFIT")
            self.assertEqual(restored.report()["realized_pnl"],"200")
            self.assertEqual(restored.report("daily")["trades"],1)
            self.assertEqual(restored.report("weekly")["learning_records"][0]["outcome"],"WIN")

    def test_stop_loss_no_short(self):
        e=AutoTrader()
        e.buy("SBIN",800,790)
        self.assertEqual(e.on_quote("SBIN",789)["reason"],"STOP_LOSS")
        with self.assertRaises(HardCheckRejected):e.sell("SBIN",789)

    def test_five_entries(self):
        e=AutoTrader()
        for _ in range(5):
            e.buy("SBIN",100,99)
            e.sell("SBIN",100)
        with self.assertRaises(HardCheckRejected):e.buy("SBIN",100,99)

    def test_daily_loss_halt(self):
        e=AutoTrader()
        for _ in range(3):
            e.buy("SBIN",100,99)
            e.on_quote("SBIN",99)
        with self.assertRaises(HardCheckRejected):e.buy("SBIN",100,99)

if __name__=="__main__":unittest.main()
