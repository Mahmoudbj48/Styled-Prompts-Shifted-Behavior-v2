"""
fill_checklist_closed.py
------------------------
Run from project root:
    python fill_checklist_closed.py

Fills the checklist for closed-model results only.
Reads  results/closed_models/aggregated_closed_models.csv
and fills  experiment_checklist_to_fill.xlsx  rows where the model column
matches one of the three closed models.

Writes:  experiment_checklist_closed_filled.xlsx

Closed model labels in the checklist must match MODEL_ALIASES below.
Adjust if your checklist uses different notation.
"""

import sys
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from pathlib import Path
from collections import defaultdict

AGG_CSV       = Path("results/closed_models/aggregated_closed_models.csv")
CHECKLIST_IN  = Path("experiment_checklist_to_fill.xlsx")
CHECKLIST_OUT = Path("experiment_checklist_closed_filled.xlsx")

# Map CSV model strings → checklist labels (adjust to match your checklist)
MODEL_ALIASES = {
    "GPT-5.4":             ["gpt-5.4", "gpt5.4"],
    "Gemini-2.5-Flash":    ["gemini-2.5-flash", "gemini-flash-2.5"],
    "Claude-Sonnet-4.6":   ["claude-sonnet-4-6", "claude-sonnet-4.6",
                            "claude-sonnet-46"],
}

def normalise_model(raw: str) -> str:
    r = raw.strip().lower()
    for label, aliases in MODEL_ALIASES.items():
        if r == label.lower() or r in [a.lower() for a in aliases]:
            return label
    return raw  # return as-is if unrecognised

STYLE_LABELS = {"Polite.", "Punct.", "Spacing", "Casing", "Length", "Form"}

# ── Load aggregated data ──────────────────────────────────────────────────────
if not AGG_CSV.exists():
    sys.exit(f"[ERROR] {AGG_CSV} not found. Run aggregate_closed_models.py first.")

print(f"Loading {AGG_CSV} ...")
agg = pd.read_csv(AGG_CSV, low_memory=False)
agg["model_norm"] = agg["model"].apply(normalise_model)

# ── Build lookup: (model_norm, dataset, style) → best run ────────────────────
candidates = defaultdict(list)

for _, row in agg.groupby(["model_norm", "dataset", "style", "run_path"]).size()\
                  .reset_index(name="n_rows").iterrows():
    key = (row["model_norm"], row["dataset"], row["style"])
    sub = agg[
        (agg["model_norm"] == row["model_norm"]) &
        (agg["dataset"]    == row["dataset"])    &
        (agg["style"]      == row["style"])      &
        (agg["run_path"]   == row["run_path"])
    ]
    candidates[key].append({
        "run_path":  row["run_path"],
        "n_rows":    row["n_rows"],
        "n_prompts": sub["prompt_id"].nunique(),
    })

# Keep best (most rows) per combination
best = {}
for key, cands in candidates.items():
    best[key] = sorted(cands, key=lambda c: c["n_rows"], reverse=True)[0]

print(f"  {len(best)} (model x dataset x style) combinations found")

# ── Load checklist ────────────────────────────────────────────────────────────
if not CHECKLIST_IN.exists():
    sys.exit(f"[ERROR] {CHECKLIST_IN} not found.")

wb = openpyxl.load_workbook(CHECKLIST_IN)
ws = wb.active

DONE_BG  = PatternFill("solid", fgColor="C6EFCE")
DONE_FT  = Font(name="Arial", size=9, color="276221")
MISS_BG  = PatternFill("solid", fgColor="FCE4D6")
MISS_FT  = Font(name="Arial", size=9, color="C55A11", italic=True)
NUM_FT   = Font(name="Arial", size=9, bold=True, color="276221")
CENTER   = Alignment(horizontal="center", vertical="center")
LEFT     = Alignment(horizontal="left",   vertical="center", wrap_text=False)

cur_model = cur_dataset = None
filled = not_found = 0

for r in range(3, ws.max_row + 1):
    v_model   = ws.cell(r, 2).value
    v_dataset = ws.cell(r, 3).value
    v_style   = ws.cell(r, 4).value

    if v_model   is not None: cur_model   = str(v_model).strip()
    if v_dataset is not None:
        ds = str(v_dataset).strip()
        if ds.startswith("▶"):
            continue
        cur_dataset = ds

    if v_style is None or str(v_style).strip() not in STYLE_LABELS:
        continue
    if cur_model is None or cur_dataset is None:
        continue

    style = str(v_style).strip()
    key   = (cur_model, cur_dataset, style)
    info  = best.get(key)

    path_cell  = ws.cell(r, 5)
    count_cell = ws.cell(r, 6)

    if info:
        path_cell.value     = Path(info["run_path"]).as_posix()
        path_cell.fill      = DONE_BG
        path_cell.font      = DONE_FT
        path_cell.alignment = LEFT
        count_cell.value     = info["n_prompts"]
        count_cell.fill      = DONE_BG
        count_cell.font      = NUM_FT
        count_cell.alignment = CENTER
        filled += 1
    else:
        # Only mark as missing if this row belongs to a closed model
        closed_labels = {k.lower() for k in MODEL_ALIASES}
        if cur_model.lower() in closed_labels or \
                any(cur_model.lower() in [a.lower() for a in v]
                    for v in MODEL_ALIASES.values()):
            path_cell.value     = "NOT FOUND"
            path_cell.fill      = MISS_BG
            path_cell.font      = MISS_FT
            path_cell.alignment = CENTER
            count_cell.value    = "—"
            count_cell.fill     = MISS_BG
            count_cell.font     = MISS_FT
            count_cell.alignment = CENTER
            not_found += 1

wb.save(CHECKLIST_OUT)
print(f"\n[OK]  Saved → {CHECKLIST_OUT}")
print(f"      Filled:    {filled}")
print(f"      Not found: {not_found}")