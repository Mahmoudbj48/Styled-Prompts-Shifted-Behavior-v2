"""
Compute per-prompt delta metrics relative to the baseline strength.

    delta_X(prompt_i, strength=s) = X(prompt_i, s) - X(prompt_i, baseline)

For delta_log_prob the sign is flipped (baseline − variant → negate to get
variant − baseline direction consistent with all other metrics).

Baselines per style:
    spacing, punctuation, letter_case, politeness  → strength == 0
    length_variation                               → strength == 1.0
    inter_vs_imper                                 → strength == "interrogative"

Usage
-----
Run as a script to backfill all existing CSVs in-place:

    python utils/compute_delta_metrics.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Baseline strength value for each style
BASELINE_STRENGTH: dict[str, Union[int, float, str]] = {
    "spacing":        0,
    "punctuation":    0,
    "letter_case":    0,
    "politeness":     0,
    "inter_vs_imper": "interrogative",
    "length_variation": 1.0,
}

# sign = -1 means the raw value is (baseline − variant), so we negate to get
# (variant − baseline) like every other delta column.
METRIC_SIGN: dict[str, int] = {
    "delta_log_prob": -1,
}

# source column → delta column name
DELTA_COL: dict[str, str] = {
    "bleu":                  "delta_bleu",
    "bertscore_prompt":      "delta_bertscore_prompt",
    "bertscore_response":    "delta_bertscore_response",
    "activation_similarity": "delta_activation_similarity",
    "delta_log_prob":        "delta_log_prob",   # overwrites itself (after sign flip)
    "jsd_drift":             "delta_jsd_drift",
    "mirroring_rate":        "delta_mirroring_rate",
}


def add_delta_columns(df: pd.DataFrame, style: str) -> pd.DataFrame:
    """Return *df* with delta columns added (or overwritten).

    Groups by (model, <id_col>, place) to find each prompt's baseline row,
    then subtracts to produce normalised delta values.
    """
    if df.empty:
        return df

    # Convert mirroring_verdict (YES/NO string) → mirroring_rate (float) if needed
    if "mirroring_verdict" in df.columns and "mirroring_rate" not in df.columns:
        df = df.copy()
        df["mirroring_rate"] = (
            df["mirroring_verdict"].astype(str).str.strip().str.upper() == "YES"
        ).astype(float)

    baseline_strength = BASELINE_STRENGTH[style]

    # Detect the ID column
    if "prompt_id" in df.columns:
        id_col = "prompt_id"
    elif "problem_id" in df.columns:
        id_col = "problem_id"
    else:
        return df

    group_keys = ["model", id_col, "place"]
    group_keys = [k for k in group_keys if k in df.columns]

    if not group_keys:
        return df

    # For inter_vs_imper the strength column contains strings; cast to str
    # for a reliable equality check so "interrogative" is found correctly.
    strength_col = df["strength"].astype(str) if style == "inter_vs_imper" \
                   else df["strength"]
    baseline_mask = strength_col == str(baseline_strength) if style == "inter_vs_imper" \
                    else df["strength"] == baseline_strength

    base = df[baseline_mask].set_index(group_keys)

    for src_col, delta_col in DELTA_COL.items():
        if src_col not in df.columns:
            continue
        if src_col not in base.columns:
            continue

        sign = METRIC_SIGN.get(delta_col, 1)

        idx = pd.MultiIndex.from_arrays([df[k] for k in group_keys])
        baseline_vals = base[src_col].reindex(idx).values

        df[delta_col] = sign * (df[src_col].values - baseline_vals)

    return df


# ---------------------------------------------------------------------------
# Back-fill entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Backfill delta columns in all existing style CSV files in-place."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from plots.sensitivity_analysis import STYLE_CSV_PATHS, COT_RUN_DIRS

    updated = 0
    skipped = 0

    # Style CSVs
    for style, rel_paths in STYLE_CSV_PATHS.items():
        for rel in rel_paths:
            csv_path = PROJECT_ROOT / rel
            if not csv_path.exists():
                skipped += 1
                continue
            df = pd.read_csv(csv_path)
            df = add_delta_columns(df, style)
            df.to_csv(csv_path, index=False)
            updated += 1
            print(f"  updated {csv_path.relative_to(PROJECT_ROOT)}")

    # CoT CSVs
    for style, run_dirs in COT_RUN_DIRS.items():
        for run_dir_rel in run_dirs:
            csv_path = PROJECT_ROOT / run_dir_rel / "results_with_cot_analysis.csv"
            if not csv_path.exists():
                skipped += 1
                continue
            df = pd.read_csv(csv_path)
            df = add_delta_columns(df, style)
            df.to_csv(csv_path, index=False)
            updated += 1
            print(f"  updated {csv_path.relative_to(PROJECT_ROOT)}")

    print(f"\nDone. updated={updated}  skipped(missing)={skipped}")


if __name__ == "__main__":
    main()