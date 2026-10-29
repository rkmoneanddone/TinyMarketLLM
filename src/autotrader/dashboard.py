"""Read-only localhost dashboard for paper market monitoring.

Uses only Python standard library. No trade endpoints or Dhan credentials.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HTML = Path(__file__).with_name("dashboard.html")

def create_handler(monitor, lock):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                body=HTML.read_bytes()
                content_type="text/html; charset=utf-8"
            elif self.path in ("/style.css","/app.js"):
                body=HTML.with_name(self.path[1:]).read_bytes()
                content_type="text/css; charset=utf-8" if self.path.endswith(".css") else "application/javascript; charset=utf-8"
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
            if self.path not in ("/api/watchlist", "/api/emergency"):
                self.send_error(405,"Live trading endpoints disabled")
                return
            try:
                length=int(self.headers.get("Content-Length","0"))
                if not 0 < length <= 4096: raise ValueError("Invalid request size")
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict): raise ValueError("Expected JSON object")
                with lock:
                    if self.path=="/api/watchlist":
                        import re
                        symbols=payload.get("symbols")
                        if not isinstance(symbols,list) or not all(isinstance(s,str) for s in symbols):
                            raise ValueError("Expected symbols array")
                        symbols=[s.strip().upper() for s in symbols]
                        if any(not re.fullmatch(r"[A-Z0-9&.-]{1,24}",s) for s in symbols):
                            raise ValueError("Invalid stock symbol")
                        result={"watchlist":list(monitor.engine.update_watchlist(symbols))}
                        monitor.latest={s:q for s,q in monitor.latest.items() if s in symbols}
                    else:
                        result=monitor.engine.emergency_flatten({
                            s:q["price"] for s,q in monitor.latest.items()
                        })
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

def serve(monitor, lock=None, host="127.0.0.1", port=8765):
    if host not in ("127.0.0.1","localhost"):
        raise ValueError("Dashboard must bind to localhost only")
    if not 1 <= port <= 65535:
        raise ValueError("Invalid port")
    lock=lock or threading.RLock()
    server=ThreadingHTTPServer(("127.0.0.1",port),create_handler(monitor,lock))
    try:
        print(f"Paper dashboard: http://127.0.0.1:{port}")
        server.serve_forever()
    finally:
        server.server_close()
