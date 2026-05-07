"""
aggregate_results.py
--------------------
Aggregates all open-source model results.

Run from project root:
    python aggregate_results.py

Scans: results/<style_folder>/run_multi_<dataset>_<timestamp>/full_results_all_models.csv
Writes: results/aggregated_full_results.csv
"""

import re
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")

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

SKIP_DIRS = {"trash", "empty_results", "closed_models"}

frames = []

for style_dir in sorted(RESULTS_DIR.iterdir()):
    if not style_dir.is_dir() or style_dir.name in SKIP_DIRS:
        continue
    style_label = STYLE_MAP.get(style_dir.name)
    if style_label is None:
        print(f"[SKIP] Unknown style folder: {style_dir.name!r}")
        continue

    for run_dir in sorted(style_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        csv_path = run_dir / "full_results_all_models.csv"
        if not csv_path.exists():
            print(f"  [NO CSV] {run_dir.name}")
            continue

        m = re.match(r"run_multi_(.+?)_\d{8}_\d+$", run_dir.name)
        if not m:
            print(f"  [SKIP] Cannot parse folder name: {run_dir.name}")
            continue
        dataset_label = DATASET_MAP.get(m.group(1))
        if dataset_label is None:
            print(f"  [SKIP] Unknown dataset slug '{m.group(1)}' in {run_dir.name}")
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

        for col in OUTPUT_COLS:
            if col not in df.columns:
                df[col] = float("nan")
        df = df[OUTPUT_COLS]
        frames.append(df)
        print(f"  [OK] {style_label} | {dataset_label} | {run_dir.name} | {len(df):,} rows")

if not frames:
    raise SystemExit("[WARN] No CSV files found.")

agg = pd.concat(frames, ignore_index=True, sort=False)
out = RESULTS_DIR / "aggregated_full_results.csv"
agg.to_csv(out, index=False)
print(f"\n[OK] {out}")
print(f"     {len(agg):,} rows x {agg.shape[1]} cols")
print(f"     models:   {sorted(agg['model'].unique())}")
print(f"     datasets: {sorted(agg['dataset'].unique())}")
print(f"     styles:   {sorted(agg['style'].unique())}")