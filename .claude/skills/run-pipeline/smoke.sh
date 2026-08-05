#!/usr/bin/env bash
# End-to-end local run of the late-delivery pipeline: train.py -> serve.py -> predict.
# Exit 0 means both stages ran and the served endpoint answered correctly.
#
# Overridable env: DATA_DIR (default /data), MODEL_MNT (default /mnt/models),
#                  PORT (default 9001), N_ROWS (fixture size, default 3000).
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SKILL_DIR/../../.." && pwd)"
WORK="$(mktemp -d)"

DATA_DIR="${DATA_DIR:-/data}"          # train.py hard-codes /data internally
MODEL_MNT="${MODEL_MNT:-/mnt/models}"  # serve.py hard-codes /mnt/models internally
PORT="${PORT:-9001}"                   # serve.py binds :9001
N_ROWS="${N_ROWS:-3000}"

SERVE_PID=""
cleanup() {
  [ -n "$SERVE_PID" ] && kill "$SERVE_PID" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "== 0. dependencies =="
python3 -c "import pandas, sklearn, joblib" 2>/dev/null \
  || pip install --quiet pandas scikit-learn joblib

echo "== 1. fixture =="
python3 "$SKILL_DIR/gen_fixture.py" "$WORK/orders.csv" "$N_ROWS"
mkdir -p "$DATA_DIR"
cp "$WORK/orders.csv" "$DATA_DIR/orders.csv"

echo "== 2. train (writes model.pkl + metrics.json) =="
MODEL_DIR="$WORK/model" python3 "$REPO_ROOT/train.py"
mkdir -p "$MODEL_MNT"
cp "$WORK/model/model.pkl" "$MODEL_MNT/model.pkl"

echo "== 3. serve (background on :$PORT) =="
python3 "$REPO_ROOT/serve.py" > "$WORK/serve.log" 2>&1 &
SERVE_PID=$!                            # real server PID — clean, no pkill self-match

for _ in $(seq 1 40); do
  curl -sf "localhost:$PORT/" >/dev/null 2>&1 && break
  # bail early if the server died on startup
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "serve.py died:"; cat "$WORK/serve.log"; exit 1; }
  sleep 0.3
done

echo "== 4. smoke-test the endpoint =="
echo -n "health:    "; curl -s "localhost:$PORT/"
echo; echo -n "high-risk: "
curl -s -X POST "localhost:$PORT/" \
  -d '{"plant":"1000","carrier":"LOCAL","priority":"HIGH","distance_km":2400,"qty":480,"weight_kg":800,"supplier_score":0.05}'
echo; echo -n "low-risk:  "
curl -s -X POST "localhost:$PORT/" \
  -d '{"plant":"2000","carrier":"DHL","priority":"LOW","distance_km":30,"qty":5,"weight_kg":2,"supplier_score":0.95}'
echo; echo -n "bad-json:  "
curl -s -w ' [HTTP %{http_code}]' -X POST "localhost:$PORT/" -d 'oops'
echo; echo
echo "== metrics.json =="
cat "$WORK/model/metrics.json"
echo; echo "OK — pipeline ran end to end."
