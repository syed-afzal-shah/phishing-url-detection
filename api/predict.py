"""Vercel Python serverless function: /api/predict

GET  /api/predict                -> health check (model loaded?)
GET  /api/predict?url=<url>      -> prediction
POST /api/predict  {"url": ...}  -> prediction
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.phishing_model import get_detector

    get_detector()  # load once at cold start
    LOAD_ERROR = None
except Exception as exc:  # report a readable error instead of crashing
    get_detector = None
    LOAD_ERROR = f"{type(exc).__name__}: {exc}"


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _classify(self, url):
        if LOAD_ERROR:
            self._send_json(500, {"error": "Model could not be loaded.", "detail": LOAD_ERROR})
            return
        try:
            self._send_json(200, get_detector().predict(url))
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": "Prediction failed.", "detail": str(exc)})

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        url = (query.get("url") or [None])[0]
        if url is None:
            self._send_json(
                200,
                {
                    "status": "ok" if not LOAD_ERROR else "error",
                    "model": "character-level CNN (NumPy inference)",
                    "usage": 'GET /api/predict?url=<url>  or  POST {"url": "<url>"}',
                    "detail": LOAD_ERROR,
                },
            )
            return
        self._classify(url)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 50000:
                self._send_json(413, {"error": "Request body too large."})
                return
            payload = json.loads(self.rfile.read(length) or b"{}")
            url = payload.get("url") if isinstance(payload, dict) else None
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": 'Body must be JSON like {"url": "..."}.'})
            return
        self._classify(url)
