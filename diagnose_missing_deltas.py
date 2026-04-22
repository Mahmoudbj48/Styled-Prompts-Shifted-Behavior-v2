"""
diagnose_missing_deltas.py
--------------------------
Run from project root:
    python diagnose_missing_deltas.py

Reads results/aggregated_full_results.csv and produces a clear report of:

  1. Combinations where ALL delta columns are NaN
     → delta computation never ran (compute_delta_metrics.py was not run,
       or the CSV had no baseline strength=0 row)

  2. Combinations where only mirroring columns are missing
     → experiment ran without --experiments mirroring

  3. Per-combination null counts for every delta / mirroring column
     → exact picture of what needs to be re-run or re-computed

Saves a summary Excel file: results/delta_diagnostics.xlsx
"""

import pandas as pd
import numpy as np
from pathlib import Path

AGG_CSV = Path("results/aggregated_full_results.csv")

# ── Column groups ─────────────────────────────────────────────────────────────
DELTA_CORE = [
    "delta_bleu",
    "delta_bertscore_prompt",
    "delta_bertscore_response",
    "delta_activation_similarity",
    "delta_log_prob",
    "delta_jsd_drift",
]
MIRRORING_SOURCE = [
    "mirroring_verdict",
    "mirroring_judge_raw",
    "mirroring_rate_batch",
    "mirroring_rate",
]
DELTA_MIRROR = ["delta_mirroring_rate"]

ALL_TRACKED = DELTA_CORE + MIRRORING_SOURCE + DELTA_MIRROR

KEY_COLS = ["style", "dataset", "model"]

# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading {AGG_CSV} …")
df = pd.read_csv(AGG_CSV, low_memory=False)
print(f"  {len(df):,} rows  ×  {df.shape[1]} cols")

# Add any tracked columns that are absent (so nullability checks work uniformly)
for col in ALL_TRACKED:
    if col not in df.columns:
        df[col] = np.nan

# Exclude baseline rows (strength=0 / 1.0 for length_variation) from null
# checks — baseline delta values are always 0.0, not NaN, so they don't
# distort the counts.  We flag issues on styled rows only.
baseline_mask = (
    ((df["style"] != "Length") & (df["strength"] == 0)) |
    ((df["style"] == "Length")  & (df["strength"] == 1.0))
)
styled = df[~baseline_mask].copy()

# ── Build per-(style, dataset, model) summary ─────────────────────────────────
records = []

for (style, dataset, model), grp in styled.groupby(KEY_COLS, sort=True):
    total_rows = len(grp)

    null_counts = {col: int(grp[col].isna().sum()) for col in ALL_TRACKED}

    core_all_null   = all(null_counts[c] == total_rows for c in DELTA_CORE)
    core_any_null   = any(null_counts[c] > 0           for c in DELTA_CORE)
    mirror_src_null = all(null_counts[c] == total_rows for c in MIRRORING_SOURCE)
    delta_mir_null  = null_counts["delta_mirroring_rate"] == total_rows

    # Classify
    if core_all_null and mirror_src_null:
        status = "MISSING ALL DELTAS + MIRRORING"
    elif core_all_null:
        status = "MISSING ALL DELTAS"
    elif mirror_src_null and delta_mir_null:
        status = "MISSING MIRRORING ONLY"
    elif delta_mir_null and not mirror_src_null:
        status = "MISSING delta_mirroring_rate ONLY"
    elif core_any_null:
        status = "PARTIAL DELTAS MISSING"
    else:
        status = "OK"

    rec = {
        "style":   style,
        "dataset": dataset,
        "model":   model,
        "total_styled_rows": total_rows,
        "status":  status,
    }
    for col in ALL_TRACKED:
        rec[f"null_{col}"] = null_counts[col]
    records.append(rec)

summary = pd.DataFrame(records)

# ── Console report ────────────────────────────────────────────────────────────
STATUS_ORDER = [
    "MISSING ALL DELTAS + MIRRORING",
    "MISSING ALL DELTAS",
    "MISSING MIRRORING ONLY",
    "MISSING delta_mirroring_rate ONLY",
    "PARTIAL DELTAS MISSING",
    "OK",
]

print()
for status in STATUS_ORDER:
    sub = summary[summary["status"] == status]
    if sub.empty:
        continue
    print(f"\n{'━'*64}")
    print(f"  {status}  ({len(sub)} combinations)")
    print(f"{'━'*64}")
    for _, row in sub.iterrows():
        print(f"  {row['style']:<10}  {row['model']:<14}  {row['dataset']}")

# ── Counts ────────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print("SUMMARY COUNTS")
print(f"{'='*64}")
counts = summary["status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
for status, n in counts.items():
    marker = "✓" if status == "OK" else "✗"
    print(f"  {marker}  {n:3d}  {status}")
total = len(summary)
ok    = counts.get("OK", 0)
print(f"\n  Total combinations : {total}")
print(f"  Fully OK           : {ok}  ({100*ok/total:.1f}%)")

# ── Save Excel ────────────────────────────────────────────────────────────────
out_xlsx = Path("results/delta_diagnostics.xlsx")

with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:

    # Sheet 1: full detail
    summary.to_excel(writer, sheet_name="All Combinations", index=False)

    # Sheet 2–5: one sheet per problem category
    for status in STATUS_ORDER[:-1]:   # skip OK
        sub = summary[summary["status"] == status][KEY_COLS + ["total_styled_rows", "status"]]
        if not sub.empty:
            sheet_name = status[:31]   # Excel tab limit
            sub.to_excel(writer, sheet_name=sheet_name, index=False)

    # Sheet 6: action plan
    action_rows = []
    for _, row in summary[summary["status"] != "OK"].iterrows():
        if "ALL DELTAS" in row["status"]:
            action = "Re-run: python utils/compute_delta_metrics.py  (then re-aggregate)"
        elif "MIRRORING" in row["status"]:
            action = (f"Re-run experiment with --experiments mirroring  "
                      f"for {row['model']} / {row['dataset']} / {row['style']}")
        else:
            action = "Investigate partial nulls — check source CSV manually"
        action_rows.append({
            "style":   row["style"],
            "dataset": row["dataset"],
            "model":   row["model"],
            "status":  row["status"],
            "action":  action,
        })
    if action_rows:
        pd.DataFrame(action_rows).to_excel(
            writer, sheet_name="Action Plan", index=False)

print(f"\n[OK]  Saved diagnostics → {out_xlsx}")
