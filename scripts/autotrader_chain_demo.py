"""Run: python scripts/autotrader_chain_demo.py (no Dhan API needed)."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.autotrader.engine import AutoTrader

engine=AutoTrader()
print("MODE: PAPER ONLY")
print("WATCHLIST:",engine.watchlist)
print("Demo BUY:",engine.buy("SBIN",800,790))
print("Demo quote at 810:",engine.on_quote("SBIN",810))
print("Demo quote at 820:",engine.on_quote("SBIN",820))
print("TRADE REPORT:",engine.report())
print("DAILY REPORT:",engine.report("daily"))
print("WEEKLY REPORT:",engine.report("weekly"))
