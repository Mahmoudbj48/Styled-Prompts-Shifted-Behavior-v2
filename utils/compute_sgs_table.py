"""
compute_sgs_table.py
--------------------
Run from project root:
    python compute_sgs_table.py

Produces two SGS tables with different normalizations:
  results/sgs_table_meanabs.tex   — normalized by global mean|delta_k|
  results/sgs_table_minmax.tex    — normalized by global min-max per metric
  results/sgs_scores_meanabs.csv  — numerical values (mean-abs)
  results/sgs_scores_minmax.csv   — numerical values (min-max)
  results/normalization_params.csv — min, max, mean|delta| per metric

Pipeline (same for both methods, only Step 2 differs):
  Step 1: load delta columns from aggregated_full_results.csv
  Step 2: normalize each delta globally per metric
  Step 3: SGS(m, j, F) = mean_i [ std_{v in V_F} [ delta_norm(i,v) ] ]
  Step 4: SGS(m, j)    = mean over 6 style families  (equal weight)
  Step 5: SGS(m)       = mean over 6 datasets         (equal weight)
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

# ── Style family keys (match 'style' column in CSV) ──────────────────────────
STYLE_FAMILIES = {
    "Polite.": "Politeness",
    "Punct.":  "Punctuation",
    "Spacing": "Spacing",
    "Casing":  "LetterCase",
    "Length":  "Length",
    "Form":    "InterVsImper",
}
FAMILY_ORDER = list(STYLE_FAMILIES.keys())

# ── Model short notations ─────────────────────────────────────────────────────
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
df_base = pd.read_csv(AGG_CSV, low_memory=False)
print(f"  {len(df_base):,} rows")
print(f"  models:   {sorted(df_base['model'].unique())}")
print(f"  datasets: {sorted(df_base['dataset'].unique())}")
print(f"  styles:   {sorted(df_base['style'].unique())}")

for col in METRICS:
    if col not in df_base.columns:
        df_base[col] = float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# Compute global normalization parameters (once, shared by both methods)
# ══════════════════════════════════════════════════════════════════════════════

print("\nComputing global normalization parameters per metric:")
norm_params = {}
for metric in METRICS:
    vals = df_base[metric].dropna()
    if vals.empty:
        norm_params[metric] = dict(mean_abs=np.nan, min=np.nan, max=np.nan, span=np.nan)
    else:
        lo, hi = float(vals.min()), float(vals.max())
        norm_params[metric] = dict(
            mean_abs = float(vals.abs().mean()),
            min      = lo,
            max      = hi,
            span     = hi - lo,
        )
    p = norm_params[metric]
    print(f"  {metric:<35}  "
          f"mean|Δ|={p['mean_abs']:.5f}  "
          f"min={p['min']:.4f}  max={p['max']:.4f}  span={p['span']:.4f}")

# Save parameters for appendix / reproducibility
pd.DataFrame([{"metric": m, **norm_params[m]} for m in METRICS]
             ).to_csv("results/normalization_params.csv", index=False)
print("  Saved → results/normalization_params.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Steps 3-5: compute SGS given a normalized copy of the dataframe
# Returns sgs4[model][dataset][metric] and sgs5[model][metric]
# ══════════════════════════════════════════════════════════════════════════════

def compute_sgs(df: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    """
    Steps 3-5 on a dataframe that already has <metric>_norm columns.
    Returns (sgs4, sgs5, df_family).
    """
    METRICS_NORM = [m + "_norm" for m in METRICS]

    # ── Step 3: SGS per (model × dataset × style family) ─────────────────────
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
                        row[metric] = float("nan")
                    else:
                        per_prompt_std = (
                            sl.groupby("prompt_id")[norm_col]
                            .std(ddof=1).dropna()
                        )
                        row[metric] = float(per_prompt_std.mean()) \
                            if not per_prompt_std.empty else float("nan")
                family_records.append(row)

    df_family = pd.DataFrame(family_records)

    # ── Step 4: mean over style families (equal weight) ───────────────────────
    sgs4 = {}
    for model in MODEL_ORDER:
        sgs4[model] = {}
        for dataset in DATASET_ORDER:
            sub = df_family[
                (df_family["model"]   == model) &
                (df_family["dataset"] == dataset)
            ]
            sgs4[model][dataset] = {
                m: float(sub[m].dropna().mean())
                if not sub[m].dropna().empty else float("nan")
                for m in METRICS
            }

    # ── Step 5: mean over datasets (equal weight) ─────────────────────────────
    sgs5 = {}
    for model in MODEL_ORDER:
        sgs5[model] = {}
        for metric in METRICS:
            vals = [sgs4[model][ds][metric] for ds in DATASET_ORDER
                    if not np.isnan(sgs4[model][ds][metric])]
            sgs5[model][metric] = float(np.mean(vals)) if vals else float("nan")

    return sgs4, sgs5, df_family


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX table builder
# ══════════════════════════════════════════════════════════════════════════════

def fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return r"\text{---}"
    return f"{v:.3f}"


def build_latex(sgs4: dict, sgs5: dict, caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \tiny",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
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
            rf"        \multirow{{{len(DATASET_ORDER)}}}{{*}}"
            rf"{{{MODEL_DISPLAY[model]}}}"
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
    return "\n".join(lines)


def save_scores_csv(sgs4: dict, sgs5: dict, path: str) -> None:
    rows = []
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            row = {"model": model, "dataset": dataset}
            row.update(sgs4[model][dataset])
            rows.append(row)
        total = {"model": model, "dataset": "Total SGS"}
        total.update(sgs5[model])
        rows.append(total)
    pd.DataFrame(rows).to_csv(path, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Run Method 1 — Mean Absolute Delta normalization
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 64)
print("METHOD 1: Mean Absolute Delta normalization")
print("=" * 64)

df_ma = df_base.copy()
for metric in METRICS:
    c = norm_params[metric]["mean_abs"]
    df_ma[metric + "_norm"] = df_ma[metric] / c \
        if (not np.isnan(c) and c > 0) else float("nan")

sgs4_ma, sgs5_ma, df_fam_ma = compute_sgs(df_ma)

caption_ma = (
    r"SGS per model and dataset --- \textit{Mean-abs normalization}: "
    r"each delta is divided by the global mean absolute delta of that metric "
    r"across all rows, so a value of 1.0 means the per-prompt instability "
    r"equals the typical perturbation effect. "
    r"Lower values indicate greater generalization robustness."
)
latex_ma = build_latex(sgs4_ma, sgs5_ma, caption_ma, "tab:sgs_meanabs")

Path("results/sgs_table_meanabs.tex").write_text(latex_ma)
save_scores_csv(sgs4_ma, sgs5_ma, "results/sgs_scores_meanabs.csv")
df_fam_ma["norm_method"] = "mean_abs"

print("[OK]  results/sgs_table_meanabs.tex")
print("[OK]  results/sgs_scores_meanabs.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Run Method 2 — Min-Max normalization
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 64)
print("METHOD 2: Min-Max normalization")
print("=" * 64)

df_mm = df_base.copy()
for metric in METRICS:
    lo   = norm_params[metric]["min"]
    span = norm_params[metric]["span"]
    df_mm[metric + "_norm"] = (df_mm[metric] - lo) / span \
        if (not np.isnan(span) and span > 0) else float("nan")

sgs4_mm, sgs5_mm, df_fam_mm = compute_sgs(df_mm)

caption_mm = (
    r"SGS per model and dataset --- \textit{Min-max normalization}: "
    r"each delta is rescaled to $[0,1]$ using the global minimum and maximum "
    r"of that metric across all rows, so 0 means no change and 1 means the "
    r"largest observed perturbation effect. "
    r"Lower values indicate greater generalization robustness."
)
latex_mm = build_latex(sgs4_mm, sgs5_mm, caption_mm, "tab:sgs_minmax")

Path("results/sgs_table_minmax.tex").write_text(latex_mm)
save_scores_csv(sgs4_mm, sgs5_mm, "results/sgs_scores_minmax.csv")
df_fam_mm["norm_method"] = "minmax"

print("[OK]  results/sgs_table_minmax.tex")
print("[OK]  results/sgs_scores_minmax.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Summary: print both Total SGS rows side by side for quick comparison
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 64)
print("TOTAL SGS COMPARISON — Mean-Abs  vs  Min-Max")
print("=" * 64)
header = f"{'Model':<12}" + "".join(
    f"  {m.replace('delta_','').replace('_similarity',''):<8}" for m in METRICS
)
print(f"\n{'Mean-Abs normalization':}")
print(header)
for model in MODEL_ORDER:
    row = f"{model:<12}"
    for m in METRICS:
        v = sgs5_ma[model][m]
        row += f"  {v:8.3f}" if not np.isnan(v) else f"  {'---':>8}"
    print(row)

print(f"\n{'Min-Max normalization':}")
print(header)
for model in MODEL_ORDER:
    row = f"{model:<12}"
    for m in METRICS:
        v = sgs5_mm[model][m]
        row += f"  {v:8.3f}" if not np.isnan(v) else f"  {'---':>8}"
    print(row)

print("\n[DONE]")