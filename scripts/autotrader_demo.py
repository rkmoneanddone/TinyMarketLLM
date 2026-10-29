"""Offline smoke test: python scripts/autotrader_demo.py"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.autotrader.core import WATCHLIST, MarketSnapshot, make_trade_plan
from src.autotrader.paper import PaperBroker

def main():
    print("AutoTrader | PAPER ONLY | watchlist:", ", ".join(WATCHLIST))
    snapshot = MarketSnapshot(symbol="SBIN", ltp=__import__("decimal").Decimal("800"))
    print("Sample quote:", snapshot)
    plan = make_trade_plan("SBIN", entry="800", stop_loss="790")
    print("Illustrative plan (NOT a signal):", plan)
    print("Paper fill:", PaperBroker().enter(plan))

if __name__ == "__main__":
    main()
