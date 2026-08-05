---
name: run-pipeline
description: Run the late-delivery risk pipeline end to end locally — train.py then serve.py — and smoke-test the prediction endpoint. Use when asked to run, start, launch, or verify the pipeline / model / scripts in this repo, or to confirm a change to train.py or serve.py still works.
---

# Run the late-delivery pipeline locally

This repo has no committed dataset and no test suite, so "running" it means:
generate a synthetic `late`-labeled CSV, train a model with `train.py`, serve it
with `serve.py`, and hit the HTTP endpoint. One command does all of it.

## One-command run

```bash
bash .claude/skills/run-pipeline/smoke.sh
```

Exit code `0` means both stages ran and the served endpoint answered correctly.
The script installs deps if missing, generates the fixture, trains, launches the
server in the background, curls it (health + high-risk + low-risk + bad-JSON),
prints `metrics.json`, then cleanly kills the server by its captured PID.

Expected tail of a healthy run:

```
health:    {"status": "alive", "features": ["plant", "carrier", ...]}
high-risk: {"late_risk": 0.9548, "prediction": "late"}
low-risk:  {"late_risk": 0.0261, "prediction": "on-time"}
bad-json:  {"error": "invalid JSON"} [HTTP 400]
...
AUC ≈ 0.9394   (deterministic — fixture and train.py both seed with a fixed value)
OK — pipeline ran end to end.
```

## Why the paths are what they are

The two scripts use fixed runtime paths dictated by SAP AI Core / KServe, not by
this repo. The smoke script stages files into them so the scripts run unmodified:

| Path | Used by | Meaning |
|---|---|---|
| `/data` | `train.py` (hard-coded `DATA_DIR`) | input CSV location |
| `/mnt/models` | `serve.py` (hard-coded `MODEL_ROOT`) | served artifact location |
| `:9001` | `serve.py` | inference port |

Override any of these — plus fixture size — via env vars:

```bash
DATA_DIR=./data MODEL_MNT=./models PORT=9100 N_ROWS=500 \
  bash .claude/skills/run-pipeline/smoke.sh
```

## Running the stages by hand

If you need to poke at one stage:

```bash
# fixture
python3 .claude/skills/run-pipeline/gen_fixture.py /data/orders.csv 3000

# train (MODEL_DIR / N_ESTIMATORS / MIN_SAMPLES_LEAF are the tunable env vars)
MODEL_DIR=./_model python3 train.py
cp ./_model/model.pkl /mnt/models/

# serve — capture the PID so you can stop it cleanly
python3 serve.py & SERVE_PID=$!
curl -s -X POST localhost:9001/ -d '{"distance_km":2400,"carrier":"LOCAL","supplier_score":0.05}'
kill "$SERVE_PID"
```

## Gotchas

- **Don't stop the server with `pkill -f serve.py`.** The pattern matches the
  shell running it (its argv contains `serve.py`), killing your own command and
  producing a spurious exit 143/144. Capture `$!` and `kill` that PID — the smoke
  script already does.
- **Deps aren't pinned.** `pandas` / `scikit-learn` / `joblib` come from the
  container image in production; there's no `requirements.txt`. The smoke script
  `pip install`s them if the import fails.
- **`serve.py` reads the model once at startup.** After retraining, restart the
  server (or re-run the smoke script) to pick up the new `model.pkl`.
- **Determinism:** both the fixture (`gen_fixture.py`) and `train.py` seed with a
  fixed value, so AUC and predictions are reproducible run to run. A changed AUC
  means a real change in the code or fixture, not noise.
