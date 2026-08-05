"""
Generate a synthetic late-delivery orders CSV for local runs of the pipeline.

The dataset mimics what train.py expects: a `late` label (0/1), a mix of
numeric and categorical features, plus a few leakage columns (identifiers and
post-delivery fields) so the LEAKY-drop logic in train.py actually exercises.

Usage:  python gen_fixture.py <output_csv> [n_rows]
"""
import csv
import random
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "orders.csv"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

random.seed(7)  # deterministic fixture -> deterministic AUC
PLANTS = ["1000", "2000", "3000"]
CARRIERS = ["DHL", "UPS", "FEDEX", "LOCAL"]
PRIOS = ["HIGH", "MED", "LOW"]

rows = []
for i in range(N):
    plant = random.choice(PLANTS)
    carrier = random.choice(CARRIERS)
    prio = random.choice(PRIOS)
    dist = random.randint(5, 2500)
    qty = random.randint(1, 500)
    weight = round(random.uniform(0.5, 900), 1)
    supplier_score = round(random.uniform(0, 1), 2)

    # Latent risk: long distance, LOCAL carrier over distance, weak supplier,
    # high quantity — plus noise so the model has to work for its AUC.
    risk = (dist / 2500) * 0.4 \
        + (0.3 if carrier == "LOCAL" and dist > 800 else 0) \
        + (1 - supplier_score) * 0.3 \
        + (qty / 500) * 0.2 \
        + random.uniform(-0.15, 0.15)
    late = 1 if risk > 0.55 else 0

    rows.append({
        # leakage columns (train.py drops these via LEAKY)
        "order": f"OR{i:06d}",
        "item": f"{(i % 10) * 10:04d}",
        "order_date": "2026-01-01",
        "days_late": 3 if late else 0,
        # real features
        "plant": plant,
        "carrier": carrier,
        "priority": prio,
        "distance_km": dist,
        "qty": qty,
        "weight_kg": weight,
        "supplier_score": supplier_score,
        # label
        "late": late,
    })

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

rate = sum(r["late"] for r in rows) / len(rows)
print(f"wrote {len(rows)} rows to {OUT}; late rate: {rate:.3f}")
