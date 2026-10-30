"""Offline smoke test: python scripts/autotrader_demo.py"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.autotrader.core import WATCHLIST, MarketSnapshot, make_trade_plan
from src.autotrader.paper import PaperBroker
from src.autotrader.hard_checks import AccountState

def main():
    print("AutoTrader | PAPER ONLY | candidate universe:", ", ".join(WATCHLIST))
    plan = make_trade_plan("SBIN", entry="800", stop_loss="790")
    print("Illustrative plan (NOT a signal):", plan)
    account = AccountState(reconciled=True, available_cash=10000)
    print("Paper fill:", PaperBroker().enter(plan, account))

if __name__ == "__main__":
    main()
