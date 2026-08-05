#!/usr/bin/env bash
# End-to-end local run of the late-delivery pipeline: train.py -> serve.py -> predict.
# Exit 0 means both stages ran and the served endpoint answered correctly.
#
# train.py reads /data and serve.py reads /mnt/models — both paths are HARD-CODED
# in the scripts (the SAP AI Core / KServe mount points), so the fixture and the
# model artifact must be staged there; env overrides can't redirect the scripts.
# Creating those dirs may need privilege on a fresh host, so we fall back to sudo.
#
# Overridable env: PORT (default 9001), N_ROWS (fixture size, default 3000).
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SKILL_DIR/../../.." && pwd)"
WORK="$(mktemp -d)"

DATA_DIR="/data"          # train.py: DATA_DIR is a hard-coded constant
MODEL_MNT="/mnt/models"   # serve.py: MODEL_ROOT is a hard-coded constant
PORT="${PORT:-9001}"      # serve.py binds :9001
N_ROWS="${N_ROWS:-3000}"

# Create a dir and make it writable by the current user, using sudo only if needed
# (works as root locally without sudo, and as the non-root CI runner with sudo).
ensure_writable_dir() {
  mkdir -p "$1" 2>/dev/null || sudo mkdir -p "$1"
  [ -w "$1" ] || sudo chmod 0777 "$1"
}

SERVE_PID=""
cleanup() {
  [ -n "$SERVE_PID" ] && kill "$SERVE_PID" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "== 0. dependencies =="
python3 -c "import pandas, sklearn, joblib" 2>/dev/null \
  || pip install --quiet -r "$REPO_ROOT/requirements.txt"

echo "== 1. fixture (-> $DATA_DIR) =="
ensure_writable_dir "$DATA_DIR"
python3 "$SKILL_DIR/gen_fixture.py" "$DATA_DIR/orders.csv" "$N_ROWS"

echo "== 2. train (writes model.pkl + metrics.json) =="
MODEL_DIR="$WORK/model" python3 "$REPO_ROOT/train.py"
ensure_writable_dir "$MODEL_MNT"
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
