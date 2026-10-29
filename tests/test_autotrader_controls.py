import tempfile
import unittest
from pathlib import Path
from src.autotrader.engine import AutoTrader
from src.autotrader.hard_checks import HardCheckRejected

class ControlTests(unittest.TestCase):
    def test_watchlist_persists(self):
        with tempfile.TemporaryDirectory() as d:
            journal=Path(d)/"paper.json"
            engine=AutoTrader(journal=journal)
            engine.update_watchlist(["TCS","INFY"])
            restored=AutoTrader(journal=journal)
            self.assertEqual(restored.watchlist,("TCS","INFY"))
    def test_cannot_remove_held_stock(self):
        engine=AutoTrader()
        engine.buy("SBIN",800,790)
        with self.assertRaises(HardCheckRejected):
            engine.update_watchlist(["BEL"])
    def test_emergency_flatten_halts(self):
        engine=AutoTrader()
        engine.buy("SBIN",800,790)
        result=engine.emergency_flatten({"SBIN":"810"})
        self.assertTrue(engine.halted)
        self.assertEqual(result["remaining"],[])
        self.assertEqual(result["closed"][0]["reason"],"EMERGENCY_EXIT")
        with self.assertRaises(HardCheckRejected):
            engine.buy("BEL",100,99)
    def test_stop_then_start(self):
        engine=AutoTrader()
        engine.buy("SBIN",800,790)
        result=engine.stop_and_exit({"SBIN":"810"})
        self.assertEqual(result["remaining"],[])
        self.assertTrue(engine.halted)
        self.assertEqual(engine.start(),"STARTED")
        self.assertFalse(engine.halted)
    def test_incomplete_exit_blocks_restart(self):
        engine=AutoTrader()
        engine.buy("SBIN",800,790)
        engine.stop_and_exit({})
        self.assertEqual(engine.halt_reason,"EXIT_INCOMPLETE")
        with self.assertRaises(HardCheckRejected):
            engine.start()
    def test_emergency_cannot_restart(self):
        engine=AutoTrader()
        engine.emergency_flatten({})
        with self.assertRaises(HardCheckRejected):
            engine.start()
    def test_missing_quote_remains_open(self):
        engine=AutoTrader()
        engine.buy("SBIN",800,790)
        result=engine.emergency_flatten({})
        self.assertEqual(result["remaining"],["SBIN"])
        self.assertTrue(engine.halted)
if __name__=="__main__":
    unittest.main()
