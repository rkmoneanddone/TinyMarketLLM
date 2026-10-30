import tempfile
import unittest
from pathlib import Path
from decimal import Decimal
from datetime import datetime,timezone,timedelta
from src.autotrader.engine import AutoTrader
from src.autotrader.monitor import PaperMonitor,Quote,csv_replay,save_status
from src.autotrader.hard_checks import HardCheckRejected

class MonitorTests(unittest.TestCase):
    def test_three_stock_observation_only(self):
        monitor=PaperMonitor(AutoTrader())
        result=monitor.run([Quote(s,Decimal("100"),100,200) for s in ("SBIN","BEL","ITC")])
        self.assertEqual(len(result["quotes"]),3)
        self.assertEqual(result["entries_today"],0)
        self.assertEqual(result["errors"],0)
    def test_protective_exit_without_entry_strategy(self):
        e=AutoTrader()
        e.buy("SBIN",800,790)
        m=PaperMonitor(e)
        self.assertIsNone(m.ingest(Quote("SBIN",Decimal("810"))))
        self.assertEqual(m.ingest(Quote("SBIN",Decimal("820")))["reason"],"TAKE_PROFIT")
        self.assertEqual(len(e.positions),0)
    def test_reject_stale_and_outside_watchlist(self):
        m=PaperMonitor(AutoTrader())
        with self.assertRaises(HardCheckRejected):
            m.ingest(Quote("UNKNOWN",Decimal("100")))
        stamp=(datetime.now(timezone.utc)-timedelta(minutes=10)).isoformat()
        with self.assertRaises(HardCheckRejected):
            m.ingest(Quote("SBIN",Decimal("100"),timestamp=stamp))
    def test_replay_csv_and_status(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"prices.csv"
            path.write_text("symbol,price,volume,oi\\nSBIN,100,10,20\\nBEL,200,30,40\\n",encoding="utf8".replace("x","x"))
            # Replace escaped literal newlines for test fixture.
            path.write_text(path.read_text().replace("\\n","\n"))
            e=AutoTrader()
            m=PaperMonitor(e)
            self.assertEqual(m.run(csv_replay(path,e.watchlist))["errors"],0)
            save_status(m,Path(d)/"status.json")
            self.assertTrue((Path(d)/"status.json").exists())
    def test_invalid_quote_does_not_stop_stream(self):
        m=PaperMonitor(AutoTrader())
        r=m.run([Quote("SBIN",Decimal("-1")),Quote("BEL",Decimal("100"))])
        self.assertEqual(r["errors"],1)
        self.assertIn("BEL",r["quotes"])

if __name__=="__main__":
    unittest.main()
