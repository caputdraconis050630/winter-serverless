"""Python ML-style function: heavy imports simulate realistic cold-init."""
import time
T0 = time.time()

# Heavy imports (the realistic cold-start cost of ML inference functions)
import numpy as np
import pandas as pd
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, os

# Simulate model load: allocate & warm a matrix
_MODEL = np.random.RandomState(0).randn(500, 500)
_MODEL = _MODEL @ _MODEL.T

INIT_SEC = time.time() - T0


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        t0 = time.time()
        # Small inference-like work
        x = np.random.randn(500)
        y = float(_MODEL @ x @ x)
        body = json.dumps({
            "runtime": "python-ml",
            "init_sec": round(INIT_SEC, 3),
            "work_ms": round((time.time() - t0) * 1000, 2),
            "result": round(y, 2),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    HTTPServer(("", port), Handler).serve_forever()
