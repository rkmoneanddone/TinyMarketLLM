"""Read-only localhost dashboard for paper market monitoring.

Uses only Python standard library. No trade endpoints or Dhan credentials.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HTML = Path(__file__).with_name("dashboard.html")

def create_handler(monitor, lock, catalog=None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                body=HTML.read_bytes()
                content_type="text/html; charset=utf-8"
            elif self.path in ("/style.css","/app.js"):
                body=HTML.with_name(self.path[1:]).read_bytes()
                content_type="text/css; charset=utf-8" if self.path.endswith(".css") else "application/javascript; charset=utf-8"
            elif self.path.startswith("/api/instruments?q="):
                from urllib.parse import parse_qs, urlsplit
                query=parse_qs(urlsplit(self.path).query).get("q",[""])[0]
                body=json.dumps(catalog.search(query) if catalog else []).encode("utf-8")
                content_type="application/json; charset=utf-8"
            elif self.path == "/api/status":
                with lock:
                    body=json.dumps(monitor.status()).encode("utf-8")
                content_type="application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type",content_type)
            self.send_header("Content-Length",str(len(body)))
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.send_header("Content-Security-Policy","default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path not in ("/api/watchlist", "/api/emergency", "/api/stop", "/api/start", "/api/exit"):
                body=json.dumps({"error":"Trading endpoint disabled"}).encode("utf-8")
                self.send_response(405)
                self.send_header("Content-Type","application/json; charset=utf-8")
                self.send_header("Content-Length",str(len(body)))
                self.send_header("Cache-Control","no-store")
                self.send_header("Connection","close")
                self.end_headers()
                self.wfile.write(body)
                self.close_connection=True
                return
            try:
                length=int(self.headers.get("Content-Length","0"))
                if not 0 < length <= 4096: raise ValueError("Invalid request size")
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict): raise ValueError("Expected JSON object")
                with lock:
                    if self.path=="/api/watchlist":
                        import re
                        action=payload.get("action")
                        symbol=payload.get("symbol")
                        if action not in ("add","remove") or not isinstance(symbol,str):
                            raise ValueError("Expected add/remove and stock symbol")
                        symbol=symbol.strip().upper()
                        current=list(monitor.engine.watchlist)
                        if action=="add":
                            if catalog is None: raise ValueError("Instrument catalog unavailable")
                            catalog.resolve(symbol)
                            symbols=current if symbol in current else current+[symbol]
                        else:
                            symbols=[s for s in current if s!=symbol]
                            if len(symbols)==len(current): raise ValueError("Symbol not watched")
                        if any(not re.fullmatch(r"[A-Z0-9&.-]{1,24}",s) for s in symbols):
                            raise ValueError("Invalid stock symbol")
                        result={"watchlist":list(monitor.engine.update_watchlist(symbols))}
                        monitor.latest={s:q for s,q in monitor.latest.items() if s in symbols}
                    elif self.path == "/api/exit":
                        symbol=payload.get("symbol")
                        if not isinstance(symbol,str) or symbol not in monitor.engine.positions:
                            raise ValueError("No open position for symbol")
                        q=monitor.latest.get(symbol)
                        if not q: raise ValueError("No observed quote: position remains open")
                        from datetime import datetime, timezone
                        if (datetime.now(timezone.utc)-datetime.fromisoformat(q["updated_at"])).total_seconds()>monitor.max_age_seconds:
                            raise ValueError("Stale quote: position remains open")
                        result=monitor.engine.sell(symbol,q["price"],"MANUAL_PAPER_EXIT")
                    elif self.path == "/api/start":
                        result={"status":monitor.engine.start()}
                    else:
                        from datetime import datetime, timezone
                        quotes={}
                        for symbol,q in monitor.latest.items():
                            stamp=datetime.fromisoformat(q["updated_at"])
                            if (datetime.now(timezone.utc)-stamp).total_seconds() <= monitor.max_age_seconds:
                                quotes[symbol]=q["price"]
                        if self.path == "/api/stop":
                            result={"status":monitor.engine.stop()}
                        else:
                            result=monitor.engine.emergency_flatten(quotes)
                body=json.dumps(result).encode("utf-8")
                code=200
            except (ValueError,KeyError,TypeError) as exc:
                body=json.dumps({"error":str(exc)}).encode("utf-8")
                code=400
            self.send_response(code)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body)))
            self.send_header("Cache-Control","no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass
    return Handler

def serve(monitor, lock=None, host="127.0.0.1", port=8765, catalog=None):
    if host not in ("127.0.0.1","localhost"):
        raise ValueError("Dashboard must bind to localhost only")
    if not 1 <= port <= 65535:
        raise ValueError("Invalid port")
    lock=lock or threading.RLock()
    server=ThreadingHTTPServer(("127.0.0.1",port),create_handler(monitor,lock,catalog))
    try:
        print(f"Paper dashboard: http://127.0.0.1:{port}")
        server.serve_forever()
    finally:
        server.server_close()
