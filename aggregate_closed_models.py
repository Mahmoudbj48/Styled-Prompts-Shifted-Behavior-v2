"""
aggregate_closed_models.py
--------------------------
Aggregates all closed-source model results.

Run from project root:
    python aggregate_closed_models.py

Scans: results/closed_models/run_<model>_<dataset>_<style>_<timestamp>/full_results_all_models.csv
Writes: results/closed_models/aggregated_closed_models.csv

Notes:
- activation_similarity / delta_activation_similarity forced to NaN (unavailable for closed models)
- mirroring_rate / delta_mirroring_rate forced to NaN for Form and Length styles
"""

import numpy as np
import pandas as pd
from pathlib import Path

CLOSED_DIR = Path("results/closed_models")
OUTPUT_CSV = CLOSED_DIR / "aggregated_closed_models.csv"

STYLE_MAP = {
    "inter_vs_imper":   "Form",
    "length_variation": "Length",
    "letter_case":      "Casing",
    "politeness":       "Polite.",
    "punctuation":      "Punct.",
    "spacing":          "Spacing",
}

DATASET_MAP = {
    "truthful_qa":       "TruthfulQA",
    "natural_questions": "Natural Questions",
    "alpaca":            "Alpaca",
    "simpleqa_verified": "SimpleQA Verified",
    "trivia_qa":         "TriviaQA",
    "hotpot_qa":         "HotpotQA",
}

MIRRORING_NA_STYLES = {"Form", "Length"}

OUTPUT_COLS = [
    "style", "dataset", "run_path",
    "model", "prompt_id", "place", "strength", "category",
    "prompt_orig", "prompt_pert", "bertscore_prompt",
    "response_orig", "response_pert",
    "bleu", "bertscore_response", "activation_similarity",
    "delta_log_prob", "entropy_shift", "mirroring_rate",
    "delta_bleu", "delta_bertscore_prompt", "delta_bertscore_response",
    "delta_activation_similarity", "delta_mirroring_rate", "delta_entropy",
]

if not CLOSED_DIR.is_dir():
    raise SystemExit(f"[ERROR] {CLOSED_DIR} not found.")

frames = []

for run_dir in sorted(CLOSED_DIR.iterdir()):
    if not run_dir.is_dir() or run_dir.name in {"trash", "empty_results"}:
        continue
    csv_path = run_dir / "full_results_all_models.csv"
    if not csv_path.exists():
        print(f"  [NO CSV] {run_dir.name}")
        continue

    # Detect style from folder name
    style_label = None
    for slug, label in STYLE_MAP.items():
        if f"_{slug}_" in run_dir.name:
            style_label = label
            break
    if style_label is None:
        print(f"  [SKIP] Cannot detect style from {run_dir.name}")
        continue

    # Detect dataset from folder name (longest match first)
    dataset_label = None
    for slug in sorted(DATASET_MAP, key=len, reverse=True):
        if f"_{slug}_" in run_dir.name:
            dataset_label = DATASET_MAP[slug]
            break
    if dataset_label is None:
        print(f"  [SKIP] Cannot detect dataset from {run_dir.name}")
        continue

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        print(f"  [ERR] {run_dir.name}: {exc}")
        continue

    for col in ("style", "dataset", "run_path"):
        if col in df.columns:
            df = df.drop(columns=[col])
    df.insert(0, "style",    style_label)
    df.insert(1, "dataset",  dataset_label)
    df.insert(2, "run_path", run_dir.as_posix())

    df["activation_similarity"]       = np.nan
    df["delta_activation_similarity"] = np.nan

    if style_label in MIRRORING_NA_STYLES:
        df["mirroring_rate"]       = np.nan
        df["delta_mirroring_rate"] = np.nan

    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[OUTPUT_COLS]
    frames.append(df)
    print(f"  [OK] {style_label} | {dataset_label} | {run_dir.name} | {len(df):,} rows")

if not frames:
    raise SystemExit("[WARN] No CSV files found.")

agg = pd.concat(frames, ignore_index=True, sort=False)
agg.to_csv(OUTPUT_CSV, index=False)
print(f"\n[OK] {OUTPUT_CSV}")
print(f"     {len(agg):,} rows x {agg.shape[1]} cols")
print(f"     models:   {sorted(agg['model'].unique())}")
print(f"     datasets: {sorted(agg['dataset'].unique())}")
print(f"     styles:   {sorted(agg['style'].unique())}")