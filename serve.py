"""
SAP AI Core serving - late-delivery risk model.
Loads model.pkl from /mnt/models (KServe storage initializer) and serves predictions.
"""
import glob
import json
import os
import tarfile

import joblib
import pandas as pd
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL_ROOT = "/mnt/models"

# -- locate model.pkl (handles both raw files and tgz-archived artifacts) ----
def find_model():
    hits = glob.glob(f"{MODEL_ROOT}/**/model.pkl", recursive=True)
    if hits:
        return hits[0]
    for tgz in glob.glob(f"{MODEL_ROOT}/**/*.tgz", recursive=True):
        with tarfile.open(tgz) as t:
            t.extractall("/tmp/extracted")
    hits = glob.glob("/tmp/extracted/**/model.pkl", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"model.pkl not found under {MODEL_ROOT}")

path = find_model()
print(f"Loading {path}", flush=True)
bundle = joblib.load(path)
model, columns, cat_maps = bundle["model"], bundle["columns"], bundle["cat_maps"]
print(f"Model ready. Features: {columns}", flush=True)

def encode(payload: dict) -> pd.DataFrame:
    row = {}
    for c in columns:
        v = payload.get(c)
        if c in cat_maps:
            cats = cat_maps[c]
            s = "NA" if v is None else str(v)
            row[c] = cats.index(s) if s in cats else -999
        else:
            try:
                row[c] = float(v) if v is not None else -999.0
            except (TypeError, ValueError):
                row[c] = -999.0
    return pd.DataFrame([row], columns=columns)

class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"status": "alive", "features": columns})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n).decode() if n else "{}"
        try:
            payload = json.loads(raw)
        except Exception:
            self._send(400, {"error": "invalid JSON"})
            return
        try:
            X = encode(payload)
            risk = float(model.predict_proba(X)[0, 1])
            self._send(200, {
                "late_risk": round(risk, 4),
                "prediction": "late" if risk > 0.5 else "on-time",
            })
        except Exception as e:
            self._send(500, {"error": str(e)})

    def log_message(self, *a):
        pass

print("Serving on :9001", flush=True)
HTTPServer(("0.0.0.0", 9001), H).serve_forever()
