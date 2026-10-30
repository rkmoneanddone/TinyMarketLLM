"""Read-only Dhan live market data adapter. Never imports or calls order APIs.

Requires the user to activate data access and provide credentials via environment.
"""
import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from urllib.request import Request, urlopen
from .monitor import Quote

DHAN_LTP_URL="https://api.dhan.co/v2/marketfeed/ltp"

class DhanReadOnlyFeed:
    def __init__(self,catalog,client_id=None,access_token=None,timeout=8):
        self.catalog=catalog
        self.client_id=client_id or os.environ.get("DHAN_CLIENT_ID")
        self.access_token=access_token or os.environ.get("DHAN_ACCESS_TOKEN")
        self.timeout=timeout
        if not self.client_id or not self.access_token:
            raise ValueError("DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN required")
    def snapshot(self,symbols):
        ids={}
        for symbol in symbols:
            item=self.catalog.resolve(symbol)
            if item["source"]!="DHAN_MASTER" or not item["security_id"]:
                raise ValueError(f"{symbol}: verified Dhan security ID required")
            ids[item["security_id"]]=symbol
        if not ids:return []
        payload=json.dumps({"NSE_EQ":list(map(int,ids))}).encode()
        request=Request(DHAN_LTP_URL,data=payload,method="POST",headers={
            "Content-Type":"application/json","Accept":"application/json",
            "access-token":self.access_token,"client-id":self.client_id})
        with urlopen(request,timeout=self.timeout) as response:
            result=json.load(response)
        if result.get("status")!="success":
            raise RuntimeError("Dhan market feed returned non-success status")
        data=result.get("data",{}).get("NSE_EQ",{})
        quotes=[]
        for sid,symbol in ids.items():
            row=data.get(sid)
            if not isinstance(row,dict) or row.get("last_price") is None:
                continue
            price=Decimal(str(row["last_price"]))
            if not price.is_finite() or price<=0:raise ValueError("Invalid Dhan LTP")
            quotes.append(Quote(symbol,price,timestamp=datetime.now(timezone.utc).isoformat()))
        return quotes
