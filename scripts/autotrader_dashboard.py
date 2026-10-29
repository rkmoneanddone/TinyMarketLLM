"""Local paper-only dashboard with optional simulated quote replay.

python scripts/autotrader_dashboard.py
Open http://127.0.0.1:8765
"""
import argparse
import itertools
import threading
import sys
import time
from decimal import Decimal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.autotrader.dashboard import serve
from src.autotrader.engine import AutoTrader
from src.autotrader.monitor import PaperMonitor,Quote,csv_replay

def synthetic():
    """Illustrative prices, not actual market data."""
    series=(800,799,798,797,796,800,804,810,820,815,805,790)
    for price in itertools.cycle(series):
        yield Quote("SBIN",Decimal(price),10000,500000)
        yield Quote("BEL",Decimal(250),20000,300000)
        yield Quote("ITC",Decimal(400),15000,400000)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",help="CSV replay instead of repeating synthetic quotes")
    p.add_argument("--interval",type=float,default=1.0,help="Seconds between quotes")
    p.add_argument("--demo",action="store_true",help="Enable unvalidated demonstration entries")
    p.add_argument("--ledger",help="Persist paper trades to JSON")
    p.add_argument("--port",type=int,default=8765)
    args=p.parse_args()
    if args.interval<=0: p.error("--interval must be positive")
    engine=AutoTrader(journal=args.ledger)
    monitor=PaperMonitor(engine,demo_signals=args.demo)
    lock=threading.RLock()
    source=csv_replay(args.csv,engine.watchlist) if args.csv else synthetic()
    def worker():
        for quote in source:
            if monitor.stopped:break
            with lock:
                try:monitor.ingest(quote)
                except Exception as exc:
                    monitor.errors+=1
                    monitor.events.append(dict(error=str(exc),symbol=quote.symbol))
                    # Unexpected errors halt further entries, but monitoring continues.
                    if not isinstance(exc,ValueError):
                        engine.halt("MONITOR_ERROR")
            time.sleep(args.interval)
    thread=threading.Thread(target=worker,daemon=True)
    thread.start()
    try:serve(monitor,lock=lock,port=args.port)
    except KeyboardInterrupt:pass
    finally:
        monitor.stop()
        thread.join(timeout=2)

if __name__=="__main__":
    main()
