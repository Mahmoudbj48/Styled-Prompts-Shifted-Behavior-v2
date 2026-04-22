"""
fill_checklist.py
-----------------
Run from the project root (same folder that contains results/ and the checklist):

    python fill_checklist.py

Folder structure expected
-------------------------
results/
  inter_vs_imper/          ← one sub-folder per style
  length_variation/
  letter_case/
  politeness/
  punctuation/
  spacing/
    run_multi_<dataset>_<timestamp>/         ← one run folder per experiment
      full_results_all_models.csv            ← required: model, prompt_id, category, ...
      plots_metrics/
        combined_means_by_model_place_strength.csv   (optional)

What the script does
--------------------
1. Loop through the 6 style folders.
2. Delete any sub-folder that contains zero files.
3. Read every full_results_all_models.csv and record, for each
   (style × model × dataset) combination, which run folder(s) cover it
   and how many unique prompt_ids were used.
4. For combinations covered by MORE than one run folder:
   - Keep the folder with the most rows (tie → newest timestamp wins).
   - Move the losers to  results/trash/<style_folder>/<run_folder>.
5. Fill the checklist (col E = path, col F = # prompts) for every
   (model × dataset × style) row.
   - Green  = found and filled.
   - Orange = combination not found in any results folder.
6. Print a per-style completion table showing how many of the expected
   (8 models × 6 datasets = 48) combinations are covered.

Output: experiment_checklist_filled.xlsx  (original file is NOT modified)
"""

import re, sys, shutil
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR   = Path("results")
TRASH_DIR     = RESULTS_DIR / "trash"
CHECKLIST_IN  = Path("experiment_checklist_to_fill.xlsx")
CHECKLIST_OUT = Path("experiment_checklist_filled.xlsx")

# ── Style: folder name → checklist label ──────────────────────────────────────
STYLE_FOLDER_TO_LABEL = {
    "inter_vs_imper":   "Form",
    "length_variation": "Length",
    "letter_case":      "Casing",
    "politeness":       "Polite.",
    "punctuation":      "Punct.",
    "spacing":          "Spacing",
}

# ── Dataset: category column value → checklist label ──────────────────────────
CATEGORY_TO_DATASET = {
    "truthful_qa":       "TruthfulQA",
    "natural_questions": "Natural Questions",
    "alpaca":            "Alpaca",
    "simpleqa_verified": "SimpleQA Verified",
    "trivia_qa":         "TriviaQA",
    "hotpot_qa":         "HotpotQA",
}

# ── Model: CSV value → checklist label ────────────────────────────────────────
# Add any additional aliases your cluster uses to the lists below.
MODEL_TO_LABEL = {
    "L3.2-3B":   ["l3.2-3b", "llama-3.2-3b-instruct",
                  "meta-llama/llama-3.2-3b-instruct"],
    "L3.1-8B":   ["l3.1-8b", "llama-3.1-8b-instruct",
                  "meta-llama/llama-3.1-8b-instruct"],
    "G-2B":      ["g-2b",    "gemma-2b-it",  "google/gemma-2b-it"],
    "G4-E4B":    ["g4-e4b",  "gemma-4-e4b-it", "google/gemma-4-e4b-it"],
    "G-7B":      ["g-7b",    "gemma-7b-it",  "google/gemma-7b-it"],
    "Q2.5-1.5B": ["q2.5-1.5b", "qwen2.5-1.5b-instruct",
                  "qwen/qwen2.5-1.5b-instruct"],
    "Q2.5-7B":   ["q2.5-7b",   "qwen2.5-7b-instruct",
                  "qwen/qwen2.5-7b-instruct"],
    "Q3.5-9B":   ["q3.5-9b",   "qwen3.5-9b", "qwen/qwen3.5-9b"],
}

# Build a flat lookup: lowercase_alias → checklist_label
_MODEL_LOOKUP = {}
for label, aliases in MODEL_TO_LABEL.items():
    _MODEL_LOOKUP[label.lower()] = label
    for alias in aliases:
        _MODEL_LOOKUP[alias.lower()] = label

def normalise_model(raw: str) -> str:
    return _MODEL_LOOKUP.get(raw.strip().lower(), raw.strip())


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Scan results/ and collect candidates
# ══════════════════════════════════════════════════════════════════════════════
# candidates[(style_label, model_label, dataset_label)] = list of dicts:
#   { run_dir, n_rows, n_unique_prompts }

candidates = defaultdict(list)

print("━" * 64)
print("PHASE 1  Scanning results/")
print("━" * 64)

if not RESULTS_DIR.is_dir():
    sys.exit(f"[ERROR] '{RESULTS_DIR}' folder not found. "
             "Run this script from the project root.")

for style_dir in sorted(RESULTS_DIR.iterdir()):
    # Skip the trash folder and any non-directories
    if not style_dir.is_dir() or style_dir.name == "trash":
        continue

    style_label = STYLE_FOLDER_TO_LABEL.get(style_dir.name)
    if style_label is None:
        print(f"  [SKIP] Unknown style folder '{style_dir.name}'")
        continue

    print(f"\n  {style_dir.name}  →  '{style_label}'")

    for run_dir in sorted(style_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        # ── Step 2: delete empty folders ─────────────────────────────────
        # Ignore hidden/OS system files (desktop.ini, .DS_Store, etc.)
        # that Windows/OneDrive may leave in "empty" folders.
        IGNORE_NAMES = {".ds_store", "desktop.ini", "thumbs.db"}
        real_files = [
            f for f in run_dir.rglob("*")
            if f.is_file()
            and f.name.lower() not in IGNORE_NAMES
            and not f.name.startswith(".")
            and not f.name.startswith("$")
        ]
        if not real_files:
            try:
                shutil.rmtree(run_dir)
                print(f"    [DELETED-EMPTY]  {run_dir.name}")
            except PermissionError as exc:
                # OneDrive / Windows Explorer may hold a sync lock — skip safely
                print(f"    [SKIP-LOCKED]    {run_dir.name}  "
                      f"(cannot delete: {exc.strerror})")
            continue

        # ── Read full_results_all_models.csv ──────────────────────────────
        csv_path = run_dir / "full_results_all_models.csv"
        if not csv_path.exists():
            print(f"    [NO CSV]  {run_dir.name}  (skipped)")
            continue

        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in
                             ["model", "prompt_id"])
        except Exception as exc:
            print(f"    [READ ERROR]  {run_dir.name}: {exc}")
            continue

        # Dataset comes from the folder name, NOT from the category column.
        # category in e.g. TruthfulQA holds internal topic labels, not the
        # dataset name.  Folder name pattern: run_multi_<dataset>_<timestamp>
        import re as _re
        m = _re.match(r"run_multi_(.+?)_\d{8}_\d+$", run_dir.name)
        if not m:
            print(f"    [WARN] Cannot parse dataset from folder name: {run_dir.name}")
            continue
        dataset_slug  = m.group(1)          # e.g. "truthful_qa", "hotpot_qa"
        dataset_label = CATEGORY_TO_DATASET.get(dataset_slug)
        if dataset_label is None:
            print(f"    [WARN] Unknown dataset slug '{dataset_slug}' "
                  f"in {run_dir.name} — skipping")
            continue

        # Each model in this CSV + the folder-derived dataset = one combination
        for model_raw, grp in df.groupby("model"):
            model_label = normalise_model(model_raw)
            key = (style_label, model_label, dataset_label)

            candidates[key].append({
                "run_dir":          run_dir,
                "n_rows":           len(grp),
                "n_unique_prompts": grp["prompt_id"].nunique(),
            })

        # Summary line
        models_found = sorted({normalise_model(m) for m in df["model"].unique()})
        print(f"    [OK]  {run_dir.name}")
        print(f"          models={models_found}  dataset='{dataset_label}'")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Deduplicate: keep best run, trash the rest
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "━" * 64)
print("PHASE 2  Deduplication → trash")
print("━" * 64)

TRASH_DIR.mkdir(parents=True, exist_ok=True)

# best[(style, model, dataset)] = single winning candidate dict
best    = {}
trashed = 0

for key, cands in candidates.items():
    style_label, model_label, dataset_label = key

    # Sort: most rows first; equal rows → newest folder name (timestamp order)
    cands_sorted = sorted(
        cands,
        key=lambda c: (c["n_rows"], c["run_dir"].name),
        reverse=True,
    )

    winner = cands_sorted[0]
    best[key] = winner

    losers = cands_sorted[1:]
    if not losers:
        continue  # no duplicates

    print(f"\n  {style_label} | {model_label} | {dataset_label}")
    print(f"    KEEP   {winner['run_dir'].name}  ({winner['n_rows']} rows)")

    for loser in losers:
        src  = loser["run_dir"]
        # Preserve the style sub-folder name inside trash/
        dest = TRASH_DIR / src.parent.name / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.move(str(src), str(dest))
            print(f"    TRASH  {src.name}  ({loser['n_rows']} rows)")
        trashed += 1

if trashed == 0:
    print("  No duplicates found.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Fill the checklist
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "━" * 64)
print("PHASE 3  Filling checklist")
print("━" * 64)

if not CHECKLIST_IN.exists():
    sys.exit(f"[ERROR] Checklist '{CHECKLIST_IN}' not found.")

wb = openpyxl.load_workbook(CHECKLIST_IN)
ws = wb.active

# Cell style helpers
def _fill(hex_color):
    return PatternFill("solid", fgColor=hex_color)

STYLE_FOUND = {
    "path_fill":    _fill("C6EFCE"),
    "path_font":    Font(name="Arial", size=9, color="276221"),
    "path_align":   Alignment(horizontal="left",   vertical="center", wrap_text=False),
    "count_fill":   _fill("C6EFCE"),
    "count_font":   Font(name="Arial", size=9, bold=True, color="276221"),
    "count_align":  Alignment(horizontal="center", vertical="center"),
}
STYLE_MISSING = {
    "path_fill":    _fill("FCE4D6"),
    "path_font":    Font(name="Arial", size=9, color="C55A11", italic=True),
    "path_align":   Alignment(horizontal="center", vertical="center"),
    "count_fill":   _fill("FCE4D6"),
    "count_font":   Font(name="Arial", size=9, color="C55A11", italic=True),
    "count_align":  Alignment(horizontal="center", vertical="center"),
}

ALL_STYLE_LABELS = set(STYLE_FOLDER_TO_LABEL.values())

# Walk the sheet, inheriting family / model / dataset from merged cells
cur_model   = None
cur_dataset = None
filled = missing = 0

for row in range(3, ws.max_row + 1):
    v_model   = ws.cell(row, 2).value
    v_dataset = ws.cell(row, 3).value
    v_style   = ws.cell(row, 4).value

    # Inherit non-None values (cells are merged vertically)
    if v_model   is not None: cur_model   = str(v_model).strip()
    if v_dataset is not None:
        v_dataset = str(v_dataset).strip()
        if v_dataset.startswith("▶"):   # SGS summary row — skip
            continue
        cur_dataset = v_dataset

    if v_style is None:
        continue
    style = str(v_style).strip()
    if style not in ALL_STYLE_LABELS:
        continue

    if cur_model is None or cur_dataset is None:
        continue

    key  = (style, cur_model, cur_dataset)
    info = best.get(key)

    path_cell  = ws.cell(row, 5)
    count_cell = ws.cell(row, 6)

    if info:
        # Use forward slashes so the path is readable on all OS
        path_str = info["run_dir"].as_posix()
        s = STYLE_FOUND
        path_cell.value     = path_str
        path_cell.fill      = s["path_fill"]
        path_cell.font      = s["path_font"]
        path_cell.alignment = s["path_align"]
        count_cell.value     = info["n_unique_prompts"]
        count_cell.fill      = s["count_fill"]
        count_cell.font      = s["count_font"]
        count_cell.alignment = s["count_align"]
        filled += 1
    else:
        s = STYLE_MISSING
        path_cell.value     = "NOT FOUND"
        path_cell.fill      = s["path_fill"]
        path_cell.font      = s["path_font"]
        path_cell.alignment = s["path_align"]
        count_cell.value     = "—"
        count_cell.fill      = s["count_fill"]
        count_cell.font      = s["count_font"]
        count_cell.alignment = s["count_align"]
        missing += 1

wb.save(CHECKLIST_OUT)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Summary
# ══════════════════════════════════════════════════════════════════════════════

EXPECTED = 8 * 6   # 8 models × 6 datasets = 48 combinations per style

print("\n" + "━" * 64)
print("SUMMARY")
print("━" * 64)
print(f"  Checklist saved  →  {CHECKLIST_OUT}")
print(f"  Rows filled      :  {filled}")
print(f"  Rows not found   :  {missing}")
print(f"  Folders trashed  :  {trashed}")
print()
print(f"  Per-style coverage  (expected {EXPECTED} combinations each):")
print(f"  {'Style folder':<22}  {'Found':>5}  {'Status'}")
print(f"  {'─'*22}  {'─'*5}  {'─'*20}")

for folder_name, style_label in sorted(STYLE_FOLDER_TO_LABEL.items(),
                                        key=lambda x: x[1]):
    found = sum(1 for (sl, _, _) in best if sl == style_label)
    status = "✓  complete" if found == EXPECTED else f"✗  {found}/{EXPECTED} missing {EXPECTED-found}"
    print(f"  {folder_name:<22}  {found:>5}  {status}")

if missing > 0:
    print()
    print("  Missing combinations (results not yet computed):")
    all_models   = list(MODEL_TO_LABEL.keys())
    all_datasets = list(CATEGORY_TO_DATASET.values())
    for sl, ml, dl in sorted(
        {(sl, ml, dl)
         for sl in STYLE_FOLDER_TO_LABEL.values()
         for ml in all_models
         for dl in all_datasets}
        - best.keys()
    ):
        print(f"    {sl:<8}  {ml:<12}  {dl}")