import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import urlopen,Request
from urllib.error import HTTPError
from src.autotrader.engine import AutoTrader
from src.autotrader.monitor import PaperMonitor
from src.autotrader.dashboard import create_handler

class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.monitor=PaperMonitor(AutoTrader())
        self.server=ThreadingHTTPServer(("127.0.0.1",0),create_handler(self.monitor,threading.RLock()))
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.base="http://127.0.0.1:"+str(self.server.server_port)
    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
    def test_status(self):
        with urlopen(self.base+"/api/status",timeout=2) as r:
            d=json.load(r)
        self.assertEqual(d["mode"],"PAPER")
        self.assertEqual(len(d["watchlist"]),3)
    def test_assets(self):
        for path in ("/","/app.js","/style.css"):
            with urlopen(self.base+path,timeout=2) as r:
                self.assertEqual(r.status,200)
                self.assertTrue(r.read())
    def test_no_trade_endpoints(self):
        with self.assertRaises(HTTPError) as ctx:
            urlopen(Request(self.base+"/api/order",data=b"{}",method="POST"),timeout=2)
        self.assertEqual(ctx.exception.code,405)

if __name__=="__main__":
    unittest.main()
