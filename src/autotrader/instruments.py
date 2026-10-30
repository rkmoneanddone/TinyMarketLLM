"""Local symbol validation. Never guess a Dhan security ID or trading symbol."""
import csv
from pathlib import Path

DEMO_SYMBOLS={"SBIN":"State Bank of India","BEL":"Bharat Electronics","ITC":"ITC"}

class InstrumentCatalog:
    def __init__(self, path=None):
        self.records={s:{"symbol":s,"name":n,"source":"DEMO_ONLY","security_id":None} for s,n in DEMO_SYMBOLS.items()}
        if path:
            with Path(path).open(encoding="utf-8-sig",newline="") as handle:
                reader=csv.DictReader(handle)
                if not reader.fieldnames or not {"SEM_TRADING_SYMBOL","SEM_SMST_SECURITY_ID","SEM_EXM_EXCH_ID","SEM_SEGMENT"}.issubset(reader.fieldnames):
                    raise ValueError("Dhan master missing required columns")
                for row in reader:
                    if row["SEM_EXM_EXCH_ID"].strip().upper()!="NSE" or row["SEM_SEGMENT"].strip().upper()!="E":
                        continue
                    symbol=row["SEM_TRADING_SYMBOL"].strip().upper()
                    sid=row["SEM_SMST_SECURITY_ID"].strip()
                    if symbol and sid:
                        self.records[symbol]={"symbol":symbol,"name":row.get("SEM_CUSTOM_SYMBOL",symbol),"source":"DHAN_MASTER","security_id":sid}
    def search(self,query):
        q=query.strip().upper()
        if len(q)<2:return []
        results=[r for r in self.records.values() if q in r["symbol"] or q in r["name"].upper()]
        return sorted(results,key=lambda r:(r["symbol"]!=q,not r["symbol"].startswith(q),r["symbol"]))[:20]
    def resolve(self,symbol):
        key=symbol.strip().upper()
        if key not in self.records:
            raise ValueError("Unknown symbol: select an exact result from the instrument catalog. Never guess Dhan mapping.")
        return self.records[key]
