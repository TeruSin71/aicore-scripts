# CLAUDE.md

Guidance for AI assistants (and humans) working in this repository.

## What this repo is

`aicore-scripts` holds the Python scripts for a **late-delivery risk** machine-learning
pipeline that runs on **SAP AI Core**. It is a two-stage pipeline:

| File | Stage | Runs as | Reads from | Writes to |
|------|-------|---------|------------|-----------|
| `train.py` | Training | AI Core training workflow step | CSV under `/data` | model + metrics in `MODEL_DIR` |
| `serve.py` | Serving | KServe inference server | `model.pkl` under `/mnt/models` | HTTP predictions on `:9001` |

The two scripts are the entire codebase — there is no package, no build system, and no
test suite. They are meant to be baked into container images and executed inside SAP
AI Core, so most paths and configuration come from the AI Core runtime, not from this repo.

## The model

- **Task:** binary classification — will an order line be delivered `late` (1) or `on-time` (0).
- **Algorithm:** scikit-learn `RandomForestClassifier` with `class_weight="balanced"`.
- **Label column:** `late`.
- **Artifact:** a single joblib pickle bundling `{"model", "columns", "cat_maps"}`.
  Training writes it and serving reads it — this dict is the contract between the two scripts.

## train.py — training step

- Finds the **first** CSV matched by `glob("/data/**/*.csv")` and loads it with pandas.
  Only one CSV is expected per run.
- Drops rows with a null label, then drops **leakage columns** (`LEAKY` list): identifiers
  and post-delivery fields (`order`, `item`, `order_date`, `actual_gi`, `days_late`,
  `delivered`, …) that would leak the answer. **Keep this list current** — if a new
  outcome-derived or identifier column enters the dataset, add it here or the AUC will be
  misleadingly high.
- **Categorical encoding heuristic:** for each feature column it tries `pd.to_numeric`.
  If at least as many values parse as numbers as were originally non-null, the column is
  treated as genuinely numeric; otherwise it is label-encoded via `pd.Categorical.codes`
  and the category order is saved in `cat_maps`. Missing/unseen values become `-999`.
  Serving **must** reproduce this exact scheme (it does, in `encode()`).
- Stratified 80/20 split, `random_state=42` throughout for reproducibility.
- Prints AUC, a classification report, and top-10 feature importances to stdout (all with
  `flush=True` so logs appear in AI Core's log stream in real time).
- Writes two files to `MODEL_DIR`:
  - `model.pkl` — the joblib bundle (model + columns + cat_maps).
  - `metrics.json` — AUC, row counts, positive rate, `n_estimators`, top features.

## serve.py — serving step

- Locates the model under `/mnt/models` (where KServe's storage initializer mounts the
  artifact). `find_model()` handles both a raw `model.pkl` and a `.tgz` archive that it
  extracts to `/tmp/extracted` — keep both paths working.
- Loads the bundle and rebuilds `model`, `columns`, `cat_maps`.
- `encode(payload)` turns one JSON object into a single-row DataFrame **in the trained
  column order**, applying the same categorical lookup and `-999` sentinel as training.
- Plain-stdlib `http.server` (no framework), listening on `0.0.0.0:9001`:
  - `GET /` → health check `{"status": "alive", "features": [...]}`.
  - `POST /` with a JSON feature object → `{"late_risk": <prob>, "prediction": "late"|"on-time"}`
    (threshold 0.5). Bad JSON → `400`; any prediction error → `500` with the message.
- `log_message` is silenced so the server does not spam per-request access logs.

## Conventions to follow

- **Keep the train/serve contract in sync.** The pickle keys (`model`, `columns`,
  `cat_maps`), the categorical-encoding rule, and the `-999` sentinel are shared. Change
  one script's handling of them and you must change the other, or serving predictions will
  silently diverge from training.
- **Configuration is via environment variables**, read with sensible defaults so the
  scripts run unmodified in the container:
  - `MODEL_DIR` (train, default `/tmp/model`) — where artifacts are written.
  - `N_ESTIMATORS` (train, default `200`) — forest size.
  - `MIN_SAMPLES_LEAF` (train, default `5`) — leaf regularization.
  Add new knobs the same way (`os.environ.get` with a default and a cast) rather than
  hard-coding values.
- **Fixed runtime paths** — `/data` (input), `/mnt/models` (served artifact), port `9001`
  — are dictated by SAP AI Core / KServe. Do not change them casually; they must match the
  AI Core workflow and serving template that live outside this repo.
- **Logging:** use `print(..., flush=True)` — that is how these scripts surface progress in
  AI Core's log stream. There is no logging framework here.
- **Dependencies** are implicit: `pandas`, `scikit-learn`, `joblib`. They are provided by
  the container image (no `requirements.txt` in the repo). If you add an import, make sure
  the image installs it.
- **Style:** standard-library-only server, small procedural scripts, module-level
  docstrings describing the AI Core role. Match this — do not introduce a web framework,
  class hierarchy, or CLI framework for tasks this simple.

## Running locally

No test harness exists; validate by exercising the scripts directly.

```bash
# Train against a local CSV that contains a `late` column
mkdir -p ./data && cp your_orders.csv ./data/
DATA_DIR=/data MODEL_DIR=./_model python train.py   # note: DATA_DIR is hard-coded to /data
# (to run fully locally, point /data at your CSV or edit DATA_DIR temporarily)

# Serve the trained model
mkdir -p /mnt/models && cp ./_model/model.pkl /mnt/models/
python serve.py        # listens on :9001

# Health check and a prediction
curl localhost:9001/
curl -X POST localhost:9001/ -d '{"feature_a": 1, "feature_b": "X"}'
```

`DATA_DIR` in `train.py` is a module constant (`/data`), not an env var — adjust it or the
mount when running outside the container.

## Git workflow

- Commit messages are short, imperative, and describe the change
  (e.g. "Implement late delivery risk model training script").
- Keep commits focused; this repo has a linear history on `main`.
- Do not open a pull request unless explicitly asked.
