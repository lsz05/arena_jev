"""A stand-in System One server for tests and dry runs: deterministic pseudo-random probabilities.

    python -m arena mock-server --port 8199
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def answer(request: dict) -> dict:
    answers = {}
    for name, q in request["questions"].items():
        options = list(q["criteria"])
        seed = hashlib.sha256(json.dumps([request["state"], name, options]).encode()).hexdigest()
        rng = random.Random(seed)
        weights = [rng.random() for _ in options]
        total = sum(weights)
        probs = {o: w / total for o, w in zip(options, weights)}
        best = max(options, key=probs.get)
        answers[name] = {"type": "choice", "choice": best, "confidence": 0.0, "probabilities": probs}
    tokens = len(json.dumps(request)) // 4
    return {"model": "mock", "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}


class _Handler(BaseHTTPRequestHandler):
    fail_every = 0  # for tests: answer 500 on every n-th request
    count = 0

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers["Content-Length"]))
        cls = type(self)
        cls.count += 1
        if cls.fail_every and cls.count % cls.fail_every == 0:
            self.send_response(500)
            self.end_headers()
            return
        data = json.dumps(answer(json.loads(body))).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        data = json.dumps({"data": [{"name": "mock"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def serve(port: int = 8199, fail_every: int = 0, background: bool = False) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"fail_every": fail_every, "count": 0})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    if background:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    else:
        server.serve_forever()
    return server
