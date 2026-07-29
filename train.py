"""
SAP AI Core training step — late-delivery risk model.
Reads a CSV dataset from /data, trains a RandomForest, writes model + metadata to /model.
"""
import glob
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

DATA_DIR = "/data"
MODEL_DIR = "/model"
LABEL = "late"

N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", "200"))
MIN_LEAF = int(os.environ.get("MIN_SAMPLES_LEAF", "5"))

# ---- load ------------------------------------------------------------------
files = glob.glob(f"{DATA_DIR}/**/*.csv", recursive=True)
if not files:
    sys.exit(f"No CSV found under {DATA_DIR}")
print(f"Reading {files[0]}", flush=True)
df = pd.read_csv(files[0], low_memory=False)
print(f"Loaded {len(df):,} rows x {len(df.columns)} cols", flush=True)

# ---- prepare ---------------------------------------------------------------
df = df.dropna(subset=[LABEL])

LEAKY = ["order", "item", "order_date", "planned_gi", "actual_gi", "planned_dlv",
         "days_late", "delivered", "lead_days", "dlv_qty"]
df = df.drop(columns=[c for c in LEAKY if c in df.columns])

y = df.pop(LABEL).astype(int)
X = df.copy()

cat_cols = [c for c in X.columns if X[c].dtype == "object"]
cat_maps = {}
for c in cat_cols:
    cats = X[c].astype("category").cat.categories
    cat_maps[c] = list(cats)
    X[c] = pd.Categorical(X[c], categories=cats).codes
X = X.fillna(-999)

print(f"Features: {list(X.columns)}", flush=True)
print(f"Label balance: {y.mean()*100:.1f}% positive", flush=True)

# ---- train -----------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS, min_samples_leaf=MIN_LEAF,
    class_weight="balanced", n_jobs=-1, random_state=42)
model.fit(X_tr, y_tr)

proba = model.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, proba)
print(f"\nAUC: {auc:.4f}\n", flush=True)
print(classification_report(y_te, (proba > 0.5).astype(int),
                            target_names=["on-time", "late"]), flush=True)

importances = (pd.Series(model.feature_importances_, index=X.columns)
               .sort_values(ascending=False))
print("Top features:\n" + importances.head(10).to_string(), flush=True)

# ---- persist ---------------------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump({"model": model, "columns": list(X.columns), "cat_maps": cat_maps},
            f"{MODEL_DIR}/model.pkl")

with open(f"{MODEL_DIR}/metrics.json", "w") as fh:
    json.dump({
        "auc": round(float(auc), 4),
        "rows_trained": int(len(X_tr)),
        "rows_tested": int(len(X_te)),
        "positive_rate": round(float(y.mean()), 4),
        "n_estimators": N_ESTIMATORS,
        "top_features": {k: round(float(v), 4)
                         for k, v in importances.head(10).items()},
    }, fh, indent=2)

print(f"\nSaved model.pkl and metrics.json to {MODEL_DIR}", flush=True)
