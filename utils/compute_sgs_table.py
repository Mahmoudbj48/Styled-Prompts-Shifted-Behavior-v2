"""
compute_sgs_table.py
--------------------
Run from project root:
    python compute_sgs_table.py

Implements the full normalized SGS pipeline:

  Step 1: Load delta columns from results/aggregated_full_results.csv
  Step 2: Normalize each delta by global mean|delta_k| across all rows
  Step 3: SGS per (model x dataset x style family)
              = mean_i [ std_v_in_F [ delta_norm_k(i,v) ] ]
  Step 4: SGS per (model x dataset)
              = mean over 6 style families  (equal weight)
  Step 5: Total SGS per (model x metric)
              = mean over 6 datasets  (equal weight)

Outputs:
  results/sgs_scores_raw.csv         raw SGS (Step 3, pre-normalization)
  results/sgs_per_family.csv         Step 3 detail: per (model, dataset, family)
  results/sgs_scores_normalized.csv  Steps 4+5 final table values
  results/normalization_denominators.csv  c_k values (for paper appendix)
  results/sgs_table.tex              filled LaTeX table
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

AGG_CSV = Path("results/aggregated_full_results.csv")

# ── Metric columns → LaTeX headers ───────────────────────────────────────────
METRIC_MAP = {
    "delta_activation_similarity": r"$\Delta$-Cos",
    "delta_bleu":                  r"$\Delta$-BLEU",
    "delta_bertscore_response":    r"$\Delta$-BERT",
    "delta_log_prob":              r"$\Delta$-Prob",
    "entropy_shift":               r"$\Delta$-Ent",
    "delta_mirroring_rate":        r"$\Delta$-MR",
}
METRICS = list(METRIC_MAP.keys())

# ── Style folder name → style family label ────────────────────────────────────
# 6 families, equal weight in Step 4
STYLE_FAMILIES = {
    "Polite.":  "Politeness",
    "Punct.":   "Punctuation",
    "Spacing":  "Spacing",
    "Casing":   "LetterCase",
    "Length":   "Length",
    "Form":     "InterVsImper",
}
FAMILY_ORDER = list(STYLE_FAMILIES.keys())   # keys match 'style' col in CSV

# ── Model display names ───────────────────────────────────────────────────────
MODEL_DISPLAY = {
    "L3.2-3B":   r"\textbf{L-3B}",
    "L3.1-8B":   r"\textbf{L-8B}",
    "G-2B":      r"\textbf{G-2B}",
    "G4-E4B":    r"\textbf{G4-E4B}",
    "G-7B":      r"\textbf{G-7B}",
    "Q2.5-1.5B": r"\textbf{Q-1.5B}",
    "Q2.5-7B":   r"\textbf{Q-7B}",
    "Q3.5-9B":   r"\textbf{Q3.5-9B}",
}
MODEL_ORDER = list(MODEL_DISPLAY.keys())

# ── Dataset display names ─────────────────────────────────────────────────────
DATASET_DISPLAY = {
    "TruthfulQA":        r"TruthfulQA \cite{lin2022truthfulqa}",
    "Natural Questions": r"Natural Questions \cite{kwiatkowski2019natural}",
    "Alpaca":            r"Alpaca \cite{taori2023stanford-alpaca}",
    "SimpleQA Verified": r"SimpleQA Verified \cite{haas_simpleqa_2026}",
    "TriviaQA":          r"TriviaQA \cite{joshi2017triviaqa}",
    "HotpotQA":          r"HotpotQA \cite{yang2018hotpotqa}",
}
DATASET_ORDER = list(DATASET_DISPLAY.keys())


# ══════════════════════════════════════════════════════════════════════════════
# Load
# ══════════════════════════════════════════════════════════════════════════════

if not AGG_CSV.exists():
    sys.exit(f"[ERROR] {AGG_CSV} not found. Run aggregate_results.py first.")

print(f"Loading {AGG_CSV} ...")
df = pd.read_csv(AGG_CSV, low_memory=False)
print(f"  {len(df):,} rows")
print(f"  models   : {sorted(df['model'].unique())}")
print(f"  datasets : {sorted(df['dataset'].unique())}")
print(f"  styles   : {sorted(df['style'].unique())}")

for col in METRICS:
    if col not in df.columns:
        df[col] = float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — normalize each delta by global mean|delta_k|
# ══════════════════════════════════════════════════════════════════════════════

print("\nStep 2 — global mean|delta| per metric  (c_k):")
c_k = {}
for metric in METRICS:
    vals = df[metric].dropna()
    c_k[metric] = float(vals.abs().mean()) if not vals.empty else float("nan")
    print(f"  {metric:<35}  c_k = {c_k[metric]:.6f}")

# Add normalized columns alongside originals
for metric in METRICS:
    norm_col = metric + "_norm"
    if not np.isnan(c_k[metric]) and c_k[metric] > 0:
        df[norm_col] = df[metric] / c_k[metric]
    else:
        df[norm_col] = float("nan")

METRICS_NORM = [m + "_norm" for m in METRICS]

# Save c_k for the paper appendix
pd.DataFrame([
    {"metric": m, "c_k": c_k[m], "description": "global mean|delta|"}
    for m in METRICS
]).to_csv("results/normalization_denominators.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — SGS per (model × dataset × style family)
#   SGS_k(m, j, F) = mean_i  std_{v in V_F} [ delta_norm_k(i, v) ]
# ══════════════════════════════════════════════════════════════════════════════

print("\nStep 3 — SGS per (model × dataset × style family) ...")

family_records = []

for model in MODEL_ORDER:
    for dataset in DATASET_ORDER:
        for style_key in FAMILY_ORDER:
            slice_df = df[
                (df["model"]   == model)  &
                (df["dataset"] == dataset) &
                (df["style"]   == style_key)
            ]

            row = {
                "model":   model,
                "dataset": dataset,
                "family":  STYLE_FAMILIES[style_key],
            }

            for metric, norm_col in zip(METRICS, METRICS_NORM):
                if slice_df.empty or norm_col not in slice_df.columns \
                        or slice_df[norm_col].isna().all():
                    row[metric] = float("nan")
                else:
                    # per-prompt std across all variants of this family
                    per_prompt_std = (
                        slice_df.groupby("prompt_id")[norm_col]
                        .std(ddof=1)
                        .dropna()
                    )
                    row[metric] = float(per_prompt_std.mean()) \
                        if not per_prompt_std.empty else float("nan")

            family_records.append(row)

df_family = pd.DataFrame(family_records)
df_family.to_csv("results/sgs_per_family.csv", index=False)
print(f"  Saved  → results/sgs_per_family.csv  ({len(df_family)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — SGS per (model × dataset)
#   average equally across 6 style families
# ══════════════════════════════════════════════════════════════════════════════

print("\nStep 4 — averaging over style families (equal weight) ...")

# sgs4[model][dataset][metric]
sgs4 = {}
for model in MODEL_ORDER:
    sgs4[model] = {}
    for dataset in DATASET_ORDER:
        per_family = df_family[
            (df_family["model"]   == model) &
            (df_family["dataset"] == dataset)
        ]
        sgs4[model][dataset] = {}
        for metric in METRICS:
            vals = per_family[metric].dropna()
            sgs4[model][dataset][metric] = float(vals.mean()) \
                if not vals.empty else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Total SGS per (model × metric)
#   average equally across 6 datasets
# ══════════════════════════════════════════════════════════════════════════════

print("Step 5 — averaging over datasets (equal weight) ...")

sgs5 = {}   # sgs5[model][metric]
for model in MODEL_ORDER:
    sgs5[model] = {}
    for metric in METRICS:
        vals = [sgs4[model][ds][metric] for ds in DATASET_ORDER
                if not np.isnan(sgs4[model][ds][metric])]
        sgs5[model][metric] = float(np.mean(vals)) if vals else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# Save normalized scores CSV
# ══════════════════════════════════════════════════════════════════════════════

norm_rows = []
for model in MODEL_ORDER:
    for dataset in DATASET_ORDER:
        row = {"model": model, "dataset": dataset}
        row.update(sgs4[model][dataset])
        norm_rows.append(row)
    # Total SGS row
    total_row = {"model": model, "dataset": "Total SGS"}
    total_row.update(sgs5[model])
    norm_rows.append(total_row)

pd.DataFrame(norm_rows).to_csv("results/sgs_scores_normalized.csv", index=False)
print(f"  Saved  → results/sgs_scores_normalized.csv")


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX table
# ══════════════════════════════════════════════════════════════════════════════

def fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return r"\text{---}"
    return f"{v:.3f}"


lines = [
    r"\begin{table}[ht]",
    r"    \centering",
    r"    \tiny",
    r"    \caption{Normalized Stylistic Generalization Score (SGS) per model and "
    r"dataset (Eq.~\ref{eq:sgs_norm}). Each cell is the mean per-prompt standard "
    r"deviation of normalized deltas, averaged equally across style families. "
    r"Lower values indicate greater generalization robustness.}",
    r"    \label{tab:sgs_results}",
    r"    \setlength{\tabcolsep}{3pt}",
    r"    \renewcommand{\arraystretch}{1.1}",
    r"    \begin{tabular}{ll cccccc}",
    r"        \toprule",
    r"        \textbf{Model} & \textbf{Dataset} & \textbf{Activations} "
    r"& \multicolumn{2}{c}{\textbf{Generation Q.}} "
    r"& \multicolumn{2}{c}{\textbf{Confidence}} & \textbf{Mirroring} \\",
    r"        \cmidrule(lr){3-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-8}",
    r"        & & $\Delta$-Cos & $\Delta$-BLEU & $\Delta$-BERT "
    r"& $\Delta$-Prob & $\Delta$-Ent & $\Delta$-MR \\",
    r"        \midrule",
]

for model_idx, model in enumerate(MODEL_ORDER):
    lines.append(f"        % --- {model} ---")
    lines.append(
        rf"        \multirow{{{len(DATASET_ORDER)}}}{{*}}{{{MODEL_DISPLAY[model]}}}"
    )

    for dataset in DATASET_ORDER:
        cells = " & ".join(fmt(sgs4[model][dataset][m]) for m in METRICS)
        lines.append(f"        & {DATASET_DISPLAY[dataset]} & {cells} \\\\")

    total_cells = " & ".join(
        r"\textbf{" + fmt(sgs5[model][m]) + "}" for m in METRICS
    )
    lines.append(
        r"        \rowcolor[gray]{.95} \multicolumn{2}{r}{\textbf{Total SGS}} "
        f"& {total_cells} \\\\"
    )

    if model_idx < len(MODEL_ORDER) - 1:
        lines.append(r"        \midrule")

lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
latex = "\n".join(lines)

Path("results/sgs_table.tex").write_text(latex)
print(f"\n[OK]  LaTeX table                  → results/sgs_table.tex")
print(f"[OK]  Per-family detail            → results/sgs_per_family.csv")
print(f"[OK]  Normalized scores            → results/sgs_scores_normalized.csv")
print(f"[OK]  Normalization denominators   → results/normalization_denominators.csv")

print("\n" + "=" * 64)
print(latex)