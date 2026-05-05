"""
aggregate_results.py
--------------------
Run from project root:
    python aggregate_results.py

Produces one file in results/:
    aggregated_full_results.csv  — all full_results_all_models.csv files combined

Each row gains three leading columns:  variation | dataset | run_path
Missing columns across files are unioned and left NaN where absent.
Dataset is derived from the run folder name (not the 'category' column).
"""

import re
import shutil
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")

# ── Style folder name -> checklist label ───────────────────────────────────────
VARIATION_MAP = {
    "politeness":       "Polite.",
    "punctuation":      "Punct.",
    "spacing":          "Spacing",
    "letter_case":      "Casing",
    "length_variation": "Length",
    "inter_vs_imper":   "Form",
}

# ── Dataset slug (from folder name) -> checklist label ─────────────────────────
DATASET_MAP = {
    "truthful_qa":       "TruthfulQA",
    "natural_questions": "Natural Questions",
    "alpaca":            "Alpaca",
    "simpleqa_verified": "SimpleQA Verified",
    "trivia_qa":         "TriviaQA",
    "hotpot_qa":         "HotpotQA",
}


def extract_dataset_slug(folder_name: str) -> str:
    """Extract dataset slug from run folder name.

    run_multi_hotpot_qa_20260416_152829  ->  hotpot_qa
    run_multi_natural_questions_20260303_125434  ->  natural_questions
    """
    m = re.match(r"run_multi_(.+?)_\d{8}_\d+$", folder_name)
    if m:
        return m.group(1)
    # fallback
    slug = re.sub(r"^run_multi_", "", folder_name)
    slug = re.sub(r"_\d{8}_\d+$", "", slug)
    return slug


def read_with_meta(csv_path: Path, variation_label: str,
                   dataset_label: str, run_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Drop any pre-existing meta columns to avoid duplicates
    for col in ("variation", "dataset", "run_path"):
        if col in df.columns:
            df = df.drop(columns=[col])
    df.insert(0, "variation",    variation_label)
    df.insert(1, "dataset",  dataset_label)
    df.insert(2, "run_path", run_path)
    return df


# ── Desired output columns (in order) ────────────────────────────────────────
# variation / dataset / run_path are prepended by read_with_meta;
# all other columns below come from the CSV itself.
# Columns absent in a given file are added as NaN automatically by pd.concat.
OUTPUT_COLS = [
    "variation", "dataset", "run_path",
    "model", "prompt_id", "place", "strength", "category",
    "prompt_orig", "prompt_pert",
    "bertscore_prompt",
    "response_orig", "response_pert",
    "bleu", "bertscore_response", "activation_similarity",
    "delta_log_prob", "entropy_shift",
    "mirroring_rate",
    "delta_bleu", "delta_bertscore_prompt", "delta_bertscore_response",
    "delta_activation_similarity", "delta_mirroring_rate", "delta_entropy",
]

frames = []

for variation_dir in sorted(RESULTS_DIR.iterdir()):
    if not variation_dir.is_dir() or variation_dir.name == "trash":
        continue
    variation_label = VARIATION_MAP.get(variation_dir.name)
    if variation_label is None:
        print(f"[WARN] Unknown variation folder: {variation_dir.name!r} — skipping")
        continue

    print(f"\n  {variation_dir.name}  ->  '{variation_label}'")

    for run_dir in sorted(variation_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        dataset_slug  = extract_dataset_slug(run_dir.name)
        dataset_label = DATASET_MAP.get(dataset_slug)
        if dataset_label is None:
            print(f"    [WARN] Unknown dataset slug {dataset_slug!r} "
                  f"(from {run_dir.name}) — using slug as-is")
            dataset_label = dataset_slug

        rel_path = str(run_dir.relative_to(Path(".")))

        full_csv = run_dir / "full_results_all_models.csv"
        if not full_csv.exists():
            empty_dir = RESULTS_DIR / "empty_results" / variation_dir.name
            empty_dir.mkdir(parents=True, exist_ok=True)
            dest = empty_dir / run_dir.name
            if not dest.exists():
                import shutil
                shutil.move(str(run_dir), str(dest))
                print(f"    [MOVED-EMPTY]  {run_dir.name}  ->  empty_results/{variation_dir.name}/")
            else:
                print(f"    [MISS]  {run_dir.name}  (already in empty_results, skipping move)")
            continue

        try:
            df = read_with_meta(full_csv, variation_label, dataset_label, rel_path)
        except Exception as exc:
            print(f"    [ERR]   {run_dir.name}: {exc}")
            continue

        frames.append(df)
        print(f"    [OK]    {run_dir.name}  "
              f"->  {len(df):,} rows  "
              f"{df['model'].nunique()} model(s)  "
              f"cols={df.shape[1]}")

print()

if not frames:
    print("[WARN] No full_results_all_models.csv files found — nothing to write.")
else:
    agg = pd.concat(frames, ignore_index=True, sort=False)

    # Keep only the desired columns, in order.
    # Columns present in OUTPUT_COLS but absent from the data are added as NaN.
    # Columns in the data but not in OUTPUT_COLS are silently dropped.
    for col in OUTPUT_COLS:
        if col not in agg.columns:
            agg[col] = float("nan")
    agg = agg[OUTPUT_COLS]

    out = RESULTS_DIR / "aggregated_full_results.csv"
    agg.to_csv(out, index=False)
    print(f"[OK]  aggregated_full_results.csv")
    print(f"      {len(agg):,} rows  x  {agg.shape[1]} cols")
    print(f"      variations:   {sorted(agg['variation'].unique())}")
    print(f"      datasets: {sorted(agg['dataset'].unique())}")
    print(f"      models:   {sorted(agg['model'].unique())}")