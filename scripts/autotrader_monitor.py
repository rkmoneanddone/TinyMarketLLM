"""Paper market replay. Example: python scripts/autotrader_monitor.py --demo --interval 0.1"""
import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.autotrader.engine import AutoTrader
from src.autotrader.monitor import PaperMonitor, Quote, csv_replay, save_status

def sample():
    # Artificial prices only; not market data or a trading recommendation.
    for p in (800,799,798,797,796,800,804,810,820,815,805,790):
        yield Quote("SBIN",Decimal(p),volume=10000,oi=500000)
        yield Quote("BEL",Decimal("250"),volume=20000,oi=300000)
        yield Quote("ITC",Decimal("400"),volume=15000,oi=400000)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",help="Replay a CSV with symbol,price[,volume,oi]")
    p.add_argument("--interval",type=float,default=0.0,help="Seconds between replayed quotes")
    p.add_argument("--demo",action="store_true",help="Enable UNVALIDATED demonstration entry signals")
    p.add_argument("--ledger",help="Optional persisted paper positions and trades")
    p.add_argument("--status",help="Optional JSON dashboard snapshot path")
    p.add_argument("--max-quotes",type=int)
    args=p.parse_args()
    engine=AutoTrader(journal=args.ledger)
    monitor=PaperMonitor(engine,demo_signals=args.demo)
    source=csv_replay(args.csv,engine.watchlist) if args.csv else sample()
    print("PAPER MODE — no Dhan orders, no live feed")
    print(json.dumps(monitor.run(source,interval_seconds=args.interval,max_quotes=args.max_quotes),indent=2))
    if args.status:save_status(monitor,args.status)

if __name__=="__main__":
    main()
