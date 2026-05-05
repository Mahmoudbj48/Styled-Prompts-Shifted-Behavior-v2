"""
aggregate_closed_models.py
--------------------------
Run from project root:
    python aggregate_closed_models.py

Scans results/closed_models/*/ and produces:
    results/closed_models/aggregated_closed_models.csv

Each row gains: variation | dataset | run_path
- activation_similarity / delta_activation_similarity stay NaN
- inter_vs_imper mirroring_rate / delta_mirroring_rate set to NaN
- length_variation mirroring_rate / delta_mirroring_rate set to NaN
  (length mirroring is not meaningful for closed models without a judge)
- SGS can be computed from any non-NaN cells

Output columns (identical schema to open-source aggregated file):
  variation, dataset, run_path,
  model, prompt_id, place, strength, category,
  prompt_orig, prompt_pert, bertscore_prompt,
  response_orig, response_pert,
  bleu, bertscore_response, activation_similarity,
  delta_log_prob, entropy_shift, mirroring_rate,
  delta_bleu, delta_bertscore_prompt, delta_bertscore_response,
  delta_activation_similarity, delta_mirroring_rate
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

CLOSED_DIR  = Path("results/closed_models")
OUTPUT_CSV  = CLOSED_DIR / "aggregated_closed_models.csv"

OUTPUT_COLS = [
    "variation", "dataset", "run_path",
    "model", "prompt_id", "place", "strength", "category",
    "prompt_orig", "prompt_pert", "bertscore_prompt",
    "response_orig", "response_pert",
    "bleu", "bertscore_response", "activation_similarity",
    "delta_log_prob", "entropy_shift", "mirroring_rate",
    "delta_bleu", "delta_bertscore_prompt", "delta_bertscore_response",
    "delta_activation_similarity", "delta_mirroring_rate", "delta_entropy",
]

# ── Mappings ──────────────────────────────────────────────────────────────────
VARIATION_FOLDER_TO_LABEL = {
    "inter_vs_imper":   "Form",
    "length_variation": "Length",
    "letter_case":      "Casing",
    "politeness":       "Polite.",
    "punctuation":      "Punct.",
    "spacing":          "Spacing",
}

DATASET_SLUG_TO_LABEL = {
    "truthful_qa":       "TruthfulQA",
    "natural_questions": "Natural Questions",
    "alpaca":            "Alpaca",
    "simpleqa_verified": "SimpleQA Verified",
    "trivia_qa":         "TriviaQA",
    "hotpot_qa":         "HotpotQA",
}

# Styles where mirroring is not applicable for closed models
MIRRORING_NA_STYLES = {"Form", "Length"}


def detect_variation(folder_name: str) -> str | None:
    for slug, label in VARIATION_FOLDER_TO_LABEL.items():
        if f"_{slug}_" in folder_name:
            return label
    return None


def detect_dataset(folder_name: str) -> str | None:
    for slug in sorted(DATASET_SLUG_TO_LABEL, key=len, reverse=True):
        if f"_{slug}_" in folder_name:
            return DATASET_SLUG_TO_LABEL[slug]
    return None


# ── Scan ──────────────────────────────────────────────────────────────────────
if not CLOSED_DIR.is_dir():
    raise SystemExit(f"[ERROR] {CLOSED_DIR} not found.")

frames = []

for run_dir in sorted(CLOSED_DIR.iterdir()):
    if not run_dir.is_dir() or run_dir.name in {"trash", "empty_results"}:
        continue

    csv_path = run_dir / "full_results_all_models.csv"
    if not csv_path.exists():
        print(f"  [SKIP]  {run_dir.name}")
        continue

    variation_label   = detect_variation(run_dir.name)
    dataset_label = detect_dataset(run_dir.name)

    if variation_label is None:
        print(f"  [WARN]  Cannot detect variation from {run_dir.name}")
        continue
    if dataset_label is None:
        print(f"  [WARN]  Cannot detect dataset from {run_dir.name}")
        continue

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        print(f"  [ERR]   {run_dir.name}: {exc}")
        continue

    # Add meta columns
    for col in ("variation", "dataset", "run_path"):
        if col in df.columns:
            df = df.drop(columns=[col])
    df.insert(0, "variation",    variation_label)
    df.insert(1, "dataset",  dataset_label)
    df.insert(2, "run_path", str(run_dir))

    # Force activation columns to NaN (always unavailable for closed models)
    df["activation_similarity"]       = np.nan
    df["delta_activation_similarity"] = np.nan

    # Force mirroring to NaN for variations where it is not applicable
    if variation_label in MIRRORING_NA_STYLES:
        df["mirroring_rate"]       = np.nan
        df["delta_mirroring_rate"] = np.nan

    # Ensure all output columns exist
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[OUTPUT_COLS]
    frames.append(df)

    n_mirror_valid = df["mirroring_rate"].notna().sum()
    n_conf_valid   = df["delta_log_prob"].notna().sum()
    print(f"  [OK]  {run_dir.name}")
    print(f"        variation={variation_label}  dataset={dataset_label}"
          f"  rows={len(df)}"
          f"  mirroring_valid={n_mirror_valid}"
          f"  confidence_valid={n_conf_valid}")

print()

if not frames:
    raise SystemExit("[WARN] No CSVs found — nothing to write.")

agg = pd.concat(frames, ignore_index=True, sort=False)
agg.to_csv(OUTPUT_CSV, index=False)

print(f"[OK]  {OUTPUT_CSV}")
print(f"      {len(agg):,} rows  x  {agg.shape[1]} cols")
print(f"      models   : {sorted(agg['model'].unique())}")
print(f"      datasets : {sorted(agg['dataset'].unique())}")
print(f"      variations   : {sorted(agg['variation'].unique())}")
print(f"\n  NaN counts on key columns:")
for col in ["activation_similarity", "delta_log_prob", "entropy_shift",
            "mirroring_rate", "delta_mirroring_rate"]:
    if col in agg.columns:
        n = agg[col].isna().sum()
        print(f"    {col:<35}  NaN={n:,} / {len(agg):,}")