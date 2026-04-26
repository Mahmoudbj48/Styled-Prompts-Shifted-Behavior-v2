"""
compute_sgs_closed_table.py
---------------------------
Run from project root:
    python compute_sgs_closed_table.py

Identical pipeline to compute_sgs_table.py but reads from
results/closed_models/aggregated_closed_models.csv.

Normalization: global min-max per metric (same as the open-source table).
NaN cells (activation, confidence for Gemini/Claude, mirroring for
Form/Length) are silently excluded at every aggregation step.

Outputs:
  results/closed_models/sgs_closed_normalization_params.csv
  results/closed_models/sgs_per_family_closed.csv
  results/closed_models/sgs_closed_scores_normalized.csv
  results/closed_models/sgs_closed_table.tex
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

AGG_CSV = Path("results/closed_models/aggregated_closed_models.csv")
OUT_DIR = Path("results/closed_models")

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

STYLE_FAMILIES = {
    "Polite.": "Politeness",
    "Punct.":  "Punctuation",
    "Spacing": "Spacing",
    "Casing":  "LetterCase",
    "Length":  "Length",
    "Form":    "InterVsImper",
}
FAMILY_ORDER = list(STYLE_FAMILIES.keys())

# Adjust these to match the exact strings in your aggregated CSV's model column
MODEL_DISPLAY = {
    "gpt-5.4":           r"\textbf{GPT-5.4}",
    "gemini-2.5-flash":  r"\textbf{Gemini-2.5-Flash}",
    "claude-sonnet-4-6": r"\textbf{Claude-Sonnet-4.6}",
}

DATASET_DISPLAY = {
    "TruthfulQA":        r"TruthfulQA \cite{lin2022truthfulqa}",
    "Natural Questions": r"Natural Questions \cite{kwiatkowski2019natural}",
    "Alpaca":            r"Alpaca \cite{taori2023stanford-alpaca}",
    "SimpleQA Verified": r"SimpleQA Verified \cite{haas_simpleqa_2026}",
    "TriviaQA":          r"TriviaQA \cite{joshi2017triviaqa}",
    "HotpotQA":          r"HotpotQA \cite{yang2018hotpotqa}",
}


# ══════════════════════════════════════════════════════════════════════════════
# Load
# ══════════════════════════════════════════════════════════════════════════════

if not AGG_CSV.exists():
    sys.exit(f"[ERROR] {AGG_CSV} not found. Run aggregate_closed_models.py first.")

print(f"Loading {AGG_CSV} ...")
df = pd.read_csv(AGG_CSV, low_memory=False)
print(f"  {len(df):,} rows")
print(f"  models:   {sorted(df['model'].unique())}")
print(f"  datasets: {sorted(df['dataset'].unique())}")
print(f"  styles:   {sorted(df['style'].unique())}")

# Build ordered lists from data (preserving display order where possible)
MODEL_ORDER   = [m for m in MODEL_DISPLAY if m in df["model"].unique()] + \
                [m for m in df["model"].unique() if m not in MODEL_DISPLAY]
DATASET_ORDER = [d for d in DATASET_DISPLAY if d in df["dataset"].unique()] + \
                [d for d in df["dataset"].unique() if d not in DATASET_DISPLAY]

for col in METRICS:
    if col not in df.columns:
        df[col] = np.nan


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — min-max normalization (global per metric)
# ══════════════════════════════════════════════════════════════════════════════

print("\nStep 2 — global min-max normalization per metric:")
norm_params = {}
for metric in METRICS:
    vals = df[metric].dropna()
    if vals.empty:
        norm_params[metric] = dict(min=np.nan, max=np.nan, span=np.nan)
        print(f"  {metric:<35}  all NaN — skipped")
    else:
        lo, hi = float(vals.min()), float(vals.max())
        span = hi - lo
        norm_params[metric] = dict(min=lo, max=hi, span=span)
        print(f"  {metric:<35}  min={lo:.4f}  max={hi:.4f}  span={span:.4f}")

    norm_col = metric + "_norm"
    span = norm_params[metric]["span"]
    lo   = norm_params[metric]["min"]
    if not np.isnan(span) and span > 0:
        df[norm_col] = (df[metric] - lo) / span
    else:
        df[norm_col] = np.nan

pd.DataFrame([{"metric": m, **norm_params[m]} for m in METRICS])\
  .to_csv(OUT_DIR / "sgs_closed_normalization_params.csv", index=False)
print("  Saved → sgs_closed_normalization_params.csv")

METRICS_NORM = [m + "_norm" for m in METRICS]


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — SGS per (model × dataset × style family)
#   SGS_k(m, j, F) = mean_i [ std_{v in V_F} [ delta_norm_k(i,v) ] ]
#   NaN-safe: all-NaN slices return NaN and are excluded downstream
# ══════════════════════════════════════════════════════════════════════════════

print("\nStep 3 — SGS per (model × dataset × style family) ...")
family_records = []

for model in MODEL_ORDER:
    for dataset in DATASET_ORDER:
        for style_key in FAMILY_ORDER:
            sl = df[
                (df["model"]   == model)   &
                (df["dataset"] == dataset) &
                (df["style"]   == style_key)
            ]
            row = {"model": model, "dataset": dataset,
                   "family": STYLE_FAMILIES[style_key]}

            for metric, norm_col in zip(METRICS, METRICS_NORM):
                if sl.empty or norm_col not in sl.columns \
                        or sl[norm_col].isna().all():
                    row[metric] = np.nan
                else:
                    per_prompt_std = (
                        sl.groupby("prompt_id")[norm_col]
                        .std(ddof=1).dropna()
                    )
                    row[metric] = float(per_prompt_std.mean()) \
                        if not per_prompt_std.empty else np.nan

            family_records.append(row)

df_fam = pd.DataFrame(family_records)
df_fam.to_csv(OUT_DIR / "sgs_per_family_closed.csv", index=False)
print(f"  Saved → sgs_per_family_closed.csv  ({len(df_fam)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — SGS per (model × dataset): mean over style families (NaN-safe)
# ══════════════════════════════════════════════════════════════════════════════

print("Step 4 — averaging over style families ...")
sgs4 = {}
for model in MODEL_ORDER:
    sgs4[model] = {}
    for dataset in DATASET_ORDER:
        sub = df_fam[
            (df_fam["model"]   == model) &
            (df_fam["dataset"] == dataset)
        ]
        sgs4[model][dataset] = {
            m: float(sub[m].dropna().mean())
               if not sub[m].dropna().empty else np.nan
            for m in METRICS
        }


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Total SGS per (model × metric): mean over datasets (NaN-safe)
# ══════════════════════════════════════════════════════════════════════════════

print("Step 5 — averaging over datasets ...")
sgs5 = {}
for model in MODEL_ORDER:
    sgs5[model] = {}
    for metric in METRICS:
        vals = [sgs4[model][d][metric] for d in DATASET_ORDER
                if not np.isnan(sgs4[model][d][metric])]
        sgs5[model][metric] = float(np.mean(vals)) if vals else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# Save normalized scores CSV
# ══════════════════════════════════════════════════════════════════════════════

norm_rows = []
for model in MODEL_ORDER:
    for dataset in DATASET_ORDER:
        r = {"model": model, "dataset": dataset}
        r.update(sgs4[model][dataset])
        norm_rows.append(r)
    r = {"model": model, "dataset": "Total SGS"}
    r.update(sgs5[model])
    norm_rows.append(r)

pd.DataFrame(norm_rows).to_csv(
    OUT_DIR / "sgs_closed_scores_normalized.csv", index=False
)
print("  Saved → sgs_closed_scores_normalized.csv")


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX table
# ══════════════════════════════════════════════════════════════════════════════

def fmt(v) -> str:
    """Format an SGS value as a 3-decimal string, or a LaTeX dash for NaN/None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return r"\text{---}"
    return f"{v:.3f}"


caption = (
    r"SGS for closed-source models --- \textit{Min-max normalization}: "
    r"each delta is rescaled to $[0,1]$ using the global minimum and maximum "
    r"across all rows. $\Delta$-Cos is unavailable (requires white-box access); "
    r"$\Delta$-Prob and $\Delta$-Ent are available only for GPT-5.4. "
    r"\text{---}\ indicates the metric is not applicable for that model. "
    r"Lower values indicate greater generalization robustness."
)

lines = [
    r"\begin{table}[ht]",
    r"    \centering",
    r"    \tiny",
    f"    \\caption{{{caption}}}",
    r"    \label{tab:sgs_closed}",
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
    display = MODEL_DISPLAY.get(model, r"\textbf{" + model + "}")
    lines.append(f"        % --- {model} ---")
    lines.append(
        rf"        \multirow{{{len(DATASET_ORDER)}}}{{*}}{{{display}}}"
    )

    for dataset in DATASET_ORDER:
        cells = " & ".join(fmt(sgs4[model][dataset][m]) for m in METRICS)
        ds_label = DATASET_DISPLAY.get(dataset, dataset)
        lines.append(f"        & {ds_label} & {cells} \\\\")

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

(OUT_DIR / "sgs_closed_table.tex").write_text(latex)

print(f"\n[OK]  sgs_closed_table.tex")
print(f"[OK]  sgs_closed_scores_normalized.csv")
print(f"\n{'='*64}")
print(latex)