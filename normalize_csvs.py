"""
normalize_csvs.py
-----------------
Run from project root:
    python normalize_csvs.py

For every full_results_all_models.csv found under results/<style>/*/:
  1. Drops any column NOT in the required list
  2. Overwrites the file in-place
  3. Reports which files are missing mirroring_rate
"""

import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
SKIP_DIRS   = {"trash", "empty_results"}

REQUIRED_COLS = [
    "model",
    "prompt_id",
    "place",
    "strength",
    "category",
    "prompt_orig",
    "prompt_pert",
    "bertscore_prompt",
    "response_orig",
    "response_pert",
    "bleu",
    "bertscore_response",
    "activation_similarity",
    "delta_log_prob",
    "entropy_shift",
    "mirroring_rate",
]

updated           = 0
errors            = 0
missing_mirroring = []   # (style, run_folder)

for style_dir in sorted(RESULTS_DIR.iterdir()):
    if not style_dir.is_dir() or style_dir.name in SKIP_DIRS:
        continue

    for run_dir in sorted(style_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        csv_path = run_dir / "full_results_all_models.csv"
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            print(f"  [ERR] {csv_path}: {e}")
            errors += 1
            continue

        # Flag if mirroring_rate is absent or entirely NaN
        if "mirroring_rate" not in df.columns or df["mirroring_rate"].isna().all():
            missing_mirroring.append((style_dir.name, run_dir.name))

        # Keep only columns that exist in BOTH the file and the required list
        # (preserve original order from REQUIRED_COLS)
        cols_to_keep = [c for c in REQUIRED_COLS if c in df.columns]
        df = df[cols_to_keep]
        df.to_csv(csv_path, index=False)
        updated += 1

# ── Report ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  CSVs updated             : {updated}")
print(f"  Errors                   : {errors}")
print(f"  Missing mirroring_rate   : {len(missing_mirroring)}")
print(f"{'='*60}")

if missing_mirroring:
    print("\n  Files where mirroring_rate is fully missing:")
    print(f"  {'Style':<22}  Run folder")
    print(f"  {'─'*22}  {'─'*40}")
    for style, run in sorted(missing_mirroring):
        print(f"  {style:<22}  {run}")
    print(f"\n  → Re-run these with --experiments mirroring")
else:
    print("\n  mirroring_rate is present in all files ✓")
