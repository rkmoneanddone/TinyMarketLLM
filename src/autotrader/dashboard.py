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
            self.send_error(405,"Dashboard is read-only")

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
