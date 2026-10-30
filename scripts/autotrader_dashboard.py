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
from src.autotrader.instruments import InstrumentCatalog
from src.autotrader.dhan_live_data import DhanReadOnlyFeed

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
    p.add_argument("--dhan-master",help="Optional local Dhan instrument master CSV")
    p.add_argument("--live-dhan",action="store_true",help="Read-only Dhan LTP polling; paper orders ONLY")
    args=p.parse_args()
    if args.interval<=0: p.error("--interval must be positive")
    if not args.ledger:
        args.ledger="data/autotrader/dashboard_paper_ledger.json"
    engine=AutoTrader(journal=args.ledger)
    catalog=InstrumentCatalog(args.dhan_master)
    if args.live_dhan and args.demo: p.error("--demo cannot be used with --live-dhan")
    if args.live_dhan and not args.dhan_master: p.error("--live-dhan requires --dhan-master")
    feed=DhanReadOnlyFeed(catalog) if args.live_dhan else None
    monitor=PaperMonitor(engine,demo_signals=args.demo)
    lock=threading.RLock()
    source=csv_replay(args.csv,engine.watchlist) if args.csv else synthetic()
    def worker():
        if feed is not None:
            while not monitor.stopped:
                try:
                    quotes=feed.snapshot(tuple(engine.watchlist))
                    with lock:
                        for quote in quotes: monitor.ingest(quote)
                except Exception as exc:
                    with lock:
                        monitor.errors+=1
                        monitor.events.append(dict(error="LIVE DATA: "+str(exc),symbol="FEED"))
                        engine.halt("LIVE_DATA_ERROR")
                time.sleep(max(args.interval,1.0))
            return
        for quote in source:
            if monitor.stopped:break
            with lock:
                if quote.symbol not in engine.watchlist:
                    continue
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
    try:serve(monitor,lock=lock,port=args.port,catalog=catalog)
    except KeyboardInterrupt:pass
    finally:
        monitor.stop()
        thread.join(timeout=2)

if __name__=="__main__":
    main()
