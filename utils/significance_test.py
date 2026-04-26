"""
significance_test.py
--------------------
Run from project root:
    python utils/significance_test.py

Statistical significance of SGS for each (model, metric) and
(model, metric, dataset) combination.

Definition
----------
For prompt i, s_i = std_{v in V}[delta_k(i,v)] where V is the full set of
stylistic variants available for that prompt. SGS = mean_i[s_i].

Null hypothesis:   H0: E[s_i] = 0  (perfect generalisation)
Alternative:       H1: E[s_i] > 0  (one-sided, greater)
Test:              scipy.stats.ttest_1samp(stds, popmean=0, alternative='greater')

Significance markers (appended as LaTeX superscripts)
------------------------------------------------------
  p < 0.001  →  ^{***}
  p < 0.01   →  ^{**}
  p < 0.05   →  ^{*}
  else       →  (none)

Outputs
-------
  results/significance_by_dataset.csv
  results/significance_by_model.csv
  results/sgs_table_meanabs.tex          (open models, mean-abs norm, with markers)
  results/sgs_table_minmax.tex           (open models, min-max norm, with markers)
  results/closed_models/significance_closed_by_dataset.csv
  results/closed_models/significance_closed_by_model.csv
  results/closed_models/sgs_closed_table.tex  (closed models, with markers)

Requires
--------
  results/aggregated_full_results.csv         (open models — run aggregate_results.py first)
  results/closed_models/aggregated_closed_models.csv
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
OPEN_AGG_CSV   = Path("results/aggregated_full_results.csv")
CLOSED_AGG_CSV = Path("results/closed_models/aggregated_closed_models.csv")
RESULTS_DIR    = Path("results")
CLOSED_DIR     = Path("results/closed_models")

# ── Metrics ───────────────────────────────────────────────────────────────────
METRIC_MAP = {
    "delta_activation_similarity": r"$\Delta$-Cos",
    "delta_bleu":                  r"$\Delta$-BLEU",
    "delta_bertscore_response":    r"$\Delta$-BERT",
    "delta_log_prob":              r"$\Delta$-Prob",
    "entropy_shift":               r"$\Delta$-Ent",
    "delta_mirroring_rate":        r"$\Delta$-MR",
}
METRICS = list(METRIC_MAP.keys())

# ── Style families ────────────────────────────────────────────────────────────
STYLE_FAMILIES = {
    "Polite.": "Politeness",
    "Punct.":  "Punctuation",
    "Spacing": "Spacing",
    "Casing":  "LetterCase",
    "Length":  "Length",
    "Form":    "InterVsImper",
}
FAMILY_ORDER = list(STYLE_FAMILIES.keys())

# ── Open-model display ────────────────────────────────────────────────────────
OPEN_MODEL_DISPLAY = {
    "L3.2-3B":   r"\textbf{L-3B}",
    "L3.1-8B":   r"\textbf{L-8B}",
    "G-2B":      r"\textbf{G-2B}",
    "G4-E4B":    r"\textbf{G4-E4B}",
    "G-7B":      r"\textbf{G-7B}",
    "Q2.5-1.5B": r"\textbf{Q-1.5B}",
    "Q2.5-7B":   r"\textbf{Q-7B}",
    "Q3.5-9B":   r"\textbf{Q3.5-9B}",
}
OPEN_MODEL_ORDER = list(OPEN_MODEL_DISPLAY.keys())

# ── Closed-model display ──────────────────────────────────────────────────────
CLOSED_MODEL_DISPLAY = {
    "gpt-5.4":           r"\textbf{GPT-5.4}",
    "gemini-2.5-flash":  r"\textbf{Gemini-2.5-Flash}",
    "claude-sonnet-4-6": r"\textbf{Claude-Sonnet-4.6}",
}

# ── Dataset display ───────────────────────────────────────────────────────────
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
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def sig_marker(p) -> str:
    """Return a LaTeX superscript significance marker for p-value p."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return r"^{\dagger}"
    if p < 0.01:
        return r"^{**}"
    if p < 0.05:
        return r"^{*}"
    return ""


def fmt(v, marker: str = "") -> str:
    """Format an SGS value with optional significance marker."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return r"\ding{55}"
    if marker:
        return f"{v:.3f}${marker}$"
    return f"{v:.3f}"


# ══════════════════════════════════════════════════════════════════════════════
# Core: significance computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_significance(df: pd.DataFrame,
                         MODEL_ORDER: list,
                         DATASET_ORDER: list,
                         METRICS: list) -> tuple[dict, dict]:
    """
    Compute one-sided t-test p-values for SGS > 0.

    Per (model, dataset, metric):
      - s_i = std_{v in V_d}[delta_k(i,v)]  (all variants for that dataset)
      - H0: E[s_i]=0   H1: E[s_i]>0

    Per (model, metric):
      - s_{d,i} = std_{v in V_d}[delta_k(d,i,v)]  pooled across all datasets
      - H0: E[s_{d,i}]=0   H1: E[s_{d,i}]>0

    Returns
    -------
    sig4[model][dataset][metric] = p-value (float, possibly nan)
    sig5[model][metric]          = p-value (float, possibly nan)
    """
    sig4: dict = {}
    sig5: dict = {}

    for model in MODEL_ORDER:
        sig4[model] = {}
        sig5[model] = {}
        df_model = df[df["model"] == model]

        # ── per (model, dataset, metric) ──────────────────────────────────────
        for dataset in DATASET_ORDER:
            sig4[model][dataset] = {}
            df_md = df_model[df_model["dataset"] == dataset]
            for metric in METRICS:
                if metric not in df_md.columns or df_md[metric].isna().all():
                    sig4[model][dataset][metric] = np.nan
                    continue
                stds = (
                    df_md.groupby("prompt_id")[metric]
                    .std(ddof=1)
                    .dropna()
                )
                if len(stds) < 2:
                    sig4[model][dataset][metric] = np.nan
                    continue
                _, p = stats.ttest_1samp(stds, popmean=0, alternative="greater")
                sig4[model][dataset][metric] = float(p)

        # ── per (model, metric) ────────────────────────────────────────────────
        for metric in METRICS:
            if metric not in df_model.columns or df_model[metric].isna().all():
                sig5[model][metric] = np.nan
                continue
            stds = (
                df_model.groupby(["dataset", "prompt_id"])[metric]
                .std(ddof=1)
                .dropna()
            )
            if len(stds) < 2:
                sig5[model][metric] = np.nan
                continue
            _, p = stats.ttest_1samp(stds, popmean=0, alternative="greater")
            sig5[model][metric] = float(p)

    return sig4, sig5


# ══════════════════════════════════════════════════════════════════════════════
# SGS computation (replicates compute_sgs_table.py pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def compute_sgs(df: pd.DataFrame,
                MODEL_ORDER: list,
                DATASET_ORDER: list,
                METRICS: list) -> tuple[dict, dict]:
    """
    Steps 3-5: compute SGS from a dataframe that already has <metric>_norm columns.

    Returns
    -------
    sgs4[model][dataset][metric]  — SGS per (model × dataset)
    sgs5[model][metric]           — SGS per model (mean over datasets)
    """
    METRICS_NORM = [m + "_norm" for m in METRICS]

    # Step 3: per (model × dataset × style family)
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
                        row[metric] = (float(per_prompt_std.mean())
                                       if not per_prompt_std.empty else np.nan)
                family_records.append(row)

    df_fam = pd.DataFrame(family_records)

    # Step 4: mean over style families
    sgs4: dict = {}
    for model in MODEL_ORDER:
        sgs4[model] = {}
        for dataset in DATASET_ORDER:
            sub = df_fam[(df_fam["model"] == model) & (df_fam["dataset"] == dataset)]
            sgs4[model][dataset] = {
                m: float(sub[m].dropna().mean())
                   if not sub[m].dropna().empty else np.nan
                for m in METRICS
            }

    # Step 5: mean over datasets
    sgs5: dict = {}
    for model in MODEL_ORDER:
        sgs5[model] = {}
        for metric in METRICS:
            vals = [sgs4[model][d][metric] for d in DATASET_ORDER
                    if not np.isnan(sgs4[model][d][metric])]
            sgs5[model][metric] = float(np.mean(vals)) if vals else np.nan

    return sgs4, sgs5


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX table builder (with significance markers)
# ══════════════════════════════════════════════════════════════════════════════

def build_latex(sgs4: dict, sgs5: dict,
                sig4: dict, sig5: dict,
                caption: str, label: str,
                MODEL_ORDER: list, DATASET_ORDER: list,
                MODEL_DISPLAY: dict, DATASET_DISPLAY: dict,
                METRICS: list) -> str:
    """
    Build a LaTeX table identical to compute_sgs_table.py but with
    significance markers appended to each cell.

    Dataset-level cells use sig4[model][dataset][metric].
    Total-SGS cells use sig5[model][metric].
    """
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
        display = MODEL_DISPLAY.get(model, r"\textbf{" + model + "}")
        lines.append(f"        % --- {model} ---")
        lines.append(
            rf"        \multirow{{{len(DATASET_ORDER)}}}{{*}}{{{display}}}"
        )

        for dataset in DATASET_ORDER:
            cells = " & ".join(
                fmt(sgs4[model][dataset][m],
                    sig_marker(sig4[model][dataset][m]))
                for m in METRICS
            )
            ds_label = DATASET_DISPLAY.get(dataset, dataset)
            lines.append(f"        & {ds_label} & {cells} \\\\")

        total_cells = " & ".join(
            r"\textbf{" + fmt(sgs5[model][m],
                              sig_marker(sig5[model][m])) + "}"
            for m in METRICS
        )
        lines.append(
            r"        \rowcolor[gray]{.95} \multicolumn{2}{r}{\textbf{Total SGS}} "
            f"& {total_cells} \\\\"
        )
        if model_idx < len(MODEL_ORDER) - 1:
            lines.append(r"        \midrule")

    lines += [r"        \bottomrule", r"    \end{tabular}"]
    lines += [
        r"    \begin{tablenotes}",
        r"        \footnotesize",
        r"        \item $^{*}p < 0.05$,\ $^{**}p < 0.01$,\ $^{\dagger}p < 0.001$"
        r" (one-sample one-sided $t$-test: $H_0\colon\mathbb{E}[s_i]=0$,"
        r" $H_1\colon\mathbb{E}[s_i]>0$, where $s_i = \mathrm{std}_{v\in\mathcal{V}}[\delta_k(i,v)]$).",
        r"    \end{tablenotes}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CSV savers
# ══════════════════════════════════════════════════════════════════════════════

def save_significance_csvs(sig4: dict, sig5: dict,
                           MODEL_ORDER: list, DATASET_ORDER: list,
                           METRICS: list,
                           path_dataset: Path, path_model: Path) -> None:
    # per (model, dataset, metric)
    rows4 = []
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            row = {"model": model, "dataset": dataset}
            for metric in METRICS:
                p = sig4[model][dataset][metric]
                row[f"p_{metric}"] = p
                row[f"sig_{metric}"] = (
                    "***" if (not np.isnan(p) and p < 0.001) else
                    "**"  if (not np.isnan(p) and p < 0.01)  else
                    "*"   if (not np.isnan(p) and p < 0.05)  else
                    "ns"
                )
            rows4.append(row)
    pd.DataFrame(rows4).to_csv(path_dataset, index=False)

    # per (model, metric)
    rows5 = []
    for model in MODEL_ORDER:
        row = {"model": model}
        for metric in METRICS:
            p = sig5[model][metric]
            row[f"p_{metric}"] = p
            row[f"sig_{metric}"] = (
                "***" if (not np.isnan(p) and p < 0.001) else
                "**"  if (not np.isnan(p) and p < 0.01)  else
                "*"   if (not np.isnan(p) and p < 0.05)  else
                "ns"
            )
        rows5.append(row)
    pd.DataFrame(rows5).to_csv(path_model, index=False)


def print_significance_summary(sig4: dict, sig5: dict,
                                MODEL_ORDER: list, DATASET_ORDER: list,
                                METRICS: list, title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    col_w = 10
    header = f"  {'Model':<14} {'Dataset':<22}" + "".join(
        f"  {m.replace('delta_','').replace('_similarity',''):>{col_w}}"
        for m in METRICS
    )
    print("\n  --- Per (model, dataset) ---")
    print(header)
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            row_str = f"  {model:<14} {dataset:<22}"
            for metric in METRICS:
                p = sig4[model][dataset][metric]
                mk = sig_marker(p) or "ns"
                row_str += f"  {mk:>{col_w}}"
            print(row_str)

    print("\n  --- Per (model) [Total SGS] ---")
    print(f"  {'Model':<14}" + "".join(
        f"  {m.replace('delta_','').replace('_similarity',''):>{col_w}}"
        for m in METRICS
    ))
    for model in MODEL_ORDER:
        row_str = f"  {model:<14}"
        for metric in METRICS:
            p = sig5[model][metric]
            mk = sig_marker(p) or "ns"
            row_str += f"  {mk:>{col_w}}"
        print(row_str)


# ══════════════════════════════════════════════════════════════════════════════
# OPEN MODELS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  OPEN MODELS")
print("="*70)

if not OPEN_AGG_CSV.exists():
    print(f"\n[SKIP] {OPEN_AGG_CSV} not found.")
    print("       Run aggregate_results.py first to generate it.")
else:
    print(f"\nLoading {OPEN_AGG_CSV} ...")
    df_open = pd.read_csv(OPEN_AGG_CSV, low_memory=False)
    print(f"  {len(df_open):,} rows")

    for col in METRICS:
        if col not in df_open.columns:
            df_open[col] = np.nan

    # Filter to known models/datasets; respect order
    open_models_in_data  = [m for m in OPEN_MODEL_ORDER
                            if m in df_open["model"].unique()]
    open_datasets_in_data = [d for d in DATASET_ORDER
                             if d in df_open["dataset"].unique()]

    # ── Significance (on raw deltas) ──────────────────────────────────────────
    print("Computing significance tests (open models)...")
    sig4_open, sig5_open = compute_significance(
        df_open, open_models_in_data, open_datasets_in_data, METRICS
    )
    print_significance_summary(
        sig4_open, sig5_open,
        open_models_in_data, open_datasets_in_data, METRICS,
        "Open Models — Significance Summary"
    )
    save_significance_csvs(
        sig4_open, sig5_open,
        open_models_in_data, open_datasets_in_data, METRICS,
        RESULTS_DIR / "significance_by_dataset.csv",
        RESULTS_DIR / "significance_by_model.csv",
    )
    print(f"\n[OK]  results/significance_by_dataset.csv")
    print(f"[OK]  results/significance_by_model.csv")

    # ── Global normalization params ───────────────────────────────────────────
    norm_params_open: dict = {}
    for metric in METRICS:
        vals = df_open[metric].dropna()
        if vals.empty:
            norm_params_open[metric] = dict(mean_abs=np.nan, min=np.nan,
                                            max=np.nan, span=np.nan)
        else:
            lo, hi = float(vals.min()), float(vals.max())
            norm_params_open[metric] = dict(
                mean_abs=float(vals.abs().mean()),
                min=lo, max=hi, span=hi - lo,
            )

    # ── Method 1: Mean-abs normalization ─────────────────────────────────────
    print("\n--- Method 1: Mean-Abs normalisation ---")
    df_ma = df_open.copy()
    for metric in METRICS:
        c = norm_params_open[metric]["mean_abs"]
        df_ma[metric + "_norm"] = (
            df_ma[metric] / c
            if (not np.isnan(c) and c > 0) else np.nan
        )

    sgs4_ma, sgs5_ma = compute_sgs(
        df_ma, open_models_in_data, open_datasets_in_data, METRICS
    )
    caption_ma = (
        r"SGS per model and dataset --- \textit{Mean-abs normalization}: "
        r"each delta is divided by the global mean absolute delta of that metric. "
        r"Superscripts denote significance of $H_1\colon\mathrm{SGS}>0$ "
        r"($^{*}p<0.05$, $^{**}p<0.01$, $^{\dagger}p<0.001$). "
        r"Lower values indicate greater generalisation robustness."
    )
    latex_ma = build_latex(
        sgs4_ma, sgs5_ma, sig4_open, sig5_open,
        caption_ma, "tab:sgs_meanabs",
        open_models_in_data, open_datasets_in_data,
        OPEN_MODEL_DISPLAY, DATASET_DISPLAY, METRICS,
    )
    (RESULTS_DIR / "sgs_table_meanabs.tex").write_text(latex_ma)
    print("[OK]  results/sgs_table_meanabs.tex")

    # ── Method 2: Min-max normalization ──────────────────────────────────────
    print("--- Method 2: Min-Max normalisation ---")
    df_mm = df_open.copy()
    for metric in METRICS:
        lo   = norm_params_open[metric]["min"]
        span = norm_params_open[metric]["span"]
        df_mm[metric + "_norm"] = (
            (df_mm[metric] - lo) / span
            if (not np.isnan(span) and span > 0) else np.nan
        )

    sgs4_mm, sgs5_mm = compute_sgs(
        df_mm, open_models_in_data, open_datasets_in_data, METRICS
    )
    caption_mm = (
        r"SGS per model and dataset --- \textit{Min-max normalization}: "
        r"each delta is rescaled to $[0,1]$ using global min/max. "
        r"Superscripts denote significance of $H_1\colon\mathrm{SGS}>0$ "
        r"($^{*}p<0.05$, $^{**}p<0.01$, $^{\dagger}p<0.001$). "
        r"Lower values indicate greater generalisation robustness."
    )
    latex_mm = build_latex(
        sgs4_mm, sgs5_mm, sig4_open, sig5_open,
        caption_mm, "tab:sgs_minmax",
        open_models_in_data, open_datasets_in_data,
        OPEN_MODEL_DISPLAY, DATASET_DISPLAY, METRICS,
    )
    (RESULTS_DIR / "sgs_table_minmax.tex").write_text(latex_mm)
    print("[OK]  results/sgs_table_minmax.tex")


# ══════════════════════════════════════════════════════════════════════════════
# CLOSED MODELS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  CLOSED MODELS")
print("="*70)

if not CLOSED_AGG_CSV.exists():
    print(f"\n[SKIP] {CLOSED_AGG_CSV} not found.")
    print("       Run aggregate_closed_models.py first to generate it.")
    sys.exit(0)

print(f"\nLoading {CLOSED_AGG_CSV} ...")
df_closed = pd.read_csv(CLOSED_AGG_CSV, low_memory=False)
print(f"  {len(df_closed):,} rows")
print(f"  models:   {sorted(df_closed['model'].unique())}")
print(f"  datasets: {sorted(df_closed['dataset'].unique())}")

for col in METRICS:
    if col not in df_closed.columns:
        df_closed[col] = np.nan

closed_models_in_data   = [m for m in CLOSED_MODEL_DISPLAY
                           if m in df_closed["model"].unique()] + \
                          [m for m in df_closed["model"].unique()
                           if m not in CLOSED_MODEL_DISPLAY]
closed_datasets_in_data = [d for d in DATASET_ORDER
                           if d in df_closed["dataset"].unique()] + \
                          [d for d in df_closed["dataset"].unique()
                           if d not in DATASET_DISPLAY]

# ── Significance (on raw deltas) ──────────────────────────────────────────────
print("\nComputing significance tests (closed models)...")
sig4_cl, sig5_cl = compute_significance(
    df_closed, closed_models_in_data, closed_datasets_in_data, METRICS
)
print_significance_summary(
    sig4_cl, sig5_cl,
    closed_models_in_data, closed_datasets_in_data, METRICS,
    "Closed Models — Significance Summary"
)
save_significance_csvs(
    sig4_cl, sig5_cl,
    closed_models_in_data, closed_datasets_in_data, METRICS,
    CLOSED_DIR / "significance_closed_by_dataset.csv",
    CLOSED_DIR / "significance_closed_by_model.csv",
)
print(f"\n[OK]  results/closed_models/significance_closed_by_dataset.csv")
print(f"[OK]  results/closed_models/significance_closed_by_model.csv")

# ── Min-max normalization ─────────────────────────────────────────────────────
print("\n--- Min-Max normalisation (closed) ---")
norm_params_cl: dict = {}
for metric in METRICS:
    vals = df_closed[metric].dropna()
    if vals.empty:
        norm_params_cl[metric] = dict(min=np.nan, max=np.nan, span=np.nan)
    else:
        lo, hi = float(vals.min()), float(vals.max())
        norm_params_cl[metric] = dict(min=lo, max=hi, span=hi - lo)
    norm_col = metric + "_norm"
    sp = norm_params_cl[metric]["span"]
    lo = norm_params_cl[metric]["min"]
    df_closed[norm_col] = (
        (df_closed[metric] - lo) / sp
        if (not np.isnan(sp) and sp > 0) else np.nan
    )

sgs4_cl, sgs5_cl = compute_sgs(
    df_closed, closed_models_in_data, closed_datasets_in_data, METRICS
)

caption_cl = (
    r"SGS for closed-source models --- \textit{Min-max normalization}: "
    r"each delta is rescaled to $[0,1]$. "
    r"$\Delta$-Cos requires white-box access (unavailable). "
    r"$\Delta$-Prob and $\Delta$-Ent are available only for GPT-5.4. "
    r"\text{---}\ indicates the metric is not applicable. "
    r"Superscripts denote significance of $H_1\colon\mathrm{SGS}>0$ "
    r"($^{*}p<0.05$, $^{**}p<0.01$, $^{\dagger}p<0.001$). "
    r"Lower values indicate greater generalisation robustness."
)
latex_cl = build_latex(
    sgs4_cl, sgs5_cl, sig4_cl, sig5_cl,
    caption_cl, "tab:sgs_closed",
    closed_models_in_data, closed_datasets_in_data,
    CLOSED_MODEL_DISPLAY, DATASET_DISPLAY, METRICS,
)
(CLOSED_DIR / "sgs_closed_table.tex").write_text(latex_cl)
print("[OK]  results/closed_models/sgs_closed_table.tex")

# ══════════════════════════════════════════════════════════════════════════════
# SPLIT TABLES  (representative + remaining)
# ══════════════════════════════════════════════════════════════════════════════

# Representative open models (one per open-source family)
OPEN_REP_MODELS     = ["L3.1-8B", "G4-E4B", "Q3.5-9B"]
# Remaining open models
OPEN_REMAIN_MODELS  = ["L3.2-3B", "G-2B", "G-7B", "Q2.5-1.5B", "Q2.5-7B"]

SIG_LEGEND = (
    r"$^{*}p<0.05$, $^{**}p<0.01$, $^{\dagger}p<0.001$ "
    r"(one-sided $t$-test: $H_0\colon\mathbb{E}[s_i]=0$, "
    r"$H_1\colon\mathbb{E}[s_i]>0$, "
    r"$s_i=\mathrm{std}_{v\in\mathcal{V}}[\delta_k(i,v)]$)."
)

TABLE_HEADER = [
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

TABLE_FOOTER = [
    r"        \bottomrule",
    r"    \end{tabular}",
    r"    \begin{tablenotes}",
    r"        \footnotesize",
    r"        \item " + SIG_LEGEND,
    r"    \end{tablenotes}",
    r"\end{table}",
]


def _model_rows(model, sgs4, sgs5, sig4, sig5,
                datasets, model_display, dataset_display, metrics):
    """Return lines for one model block (dataset rows + Total SGS row)."""
    display = model_display.get(model, r"\textbf{" + model + "}")
    lines = [f"        % --- {model} ---"]
    lines.append(rf"        \multirow{{{len(datasets)}}}{{*}}{{{display}}}")
    for dataset in datasets:
        cells = " & ".join(
            fmt(sgs4[model][dataset][m], sig_marker(sig4[model][dataset][m]))
            for m in metrics
        )
        ds_label = dataset_display.get(dataset, dataset)
        lines.append(f"        & {ds_label} & {cells} \\\\")
    total_cells = " & ".join(
        r"\textbf{" + fmt(sgs5[model][m], sig_marker(sig5[model][m])) + "}"
        for m in metrics
    )
    lines.append(
        r"        \rowcolor[gray]{.95} \multicolumn{2}{r}{\textbf{Total SGS}} "
        f"& {total_cells} \\\\"
    )
    return lines


def build_combined_latex(
    # closed
    sgs4_cl, sgs5_cl, sig4_cl, sig5_cl,
    closed_models, closed_datasets,
    # open representative
    sgs4_op, sgs5_op, sig4_op, sig5_op,
    open_rep_models, open_datasets,
    # shared
    closed_model_display, open_model_display,
    dataset_display, metrics,
    caption, label,
) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \tiny",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \setlength{\tabcolsep}{3pt}",
        r"    \renewcommand{\arraystretch}{1.1}",
    ] + TABLE_HEADER

    # ── Section label: closed models ─────────────────────────────────────────
    lines.append(
        r"        \multicolumn{8}{l}{\textit{Closed-source models}} \\"
    )
    lines.append(r"        \midrule")

    for i, model in enumerate(closed_models):
        lines += _model_rows(model, sgs4_cl, sgs5_cl, sig4_cl, sig5_cl,
                             closed_datasets, closed_model_display,
                             dataset_display, metrics)
        if i < len(closed_models) - 1:
            lines.append(r"        \midrule")

    # ── Section label: representative open models ─────────────────────────────
    lines.append(r"        \midrule")
    lines.append(
        r"        \multicolumn{8}{l}{\textit{Open-source models (representative, one per family)}} \\"
    )
    lines.append(r"        \midrule")

    open_rep_in_data = [m for m in open_rep_models if m in sgs4_op]
    for i, model in enumerate(open_rep_in_data):
        lines += _model_rows(model, sgs4_op, sgs5_op, sig4_op, sig5_op,
                             open_datasets, open_model_display,
                             dataset_display, metrics)
        if i < len(open_rep_in_data) - 1:
            lines.append(r"        \midrule")

    lines += TABLE_FOOTER
    return "\n".join(lines)


def build_remaining_latex(
    sgs4, sgs5, sig4, sig5,
    remaining_models, datasets,
    model_display, dataset_display, metrics,
    caption, label,
) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \tiny",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \setlength{\tabcolsep}{3pt}",
        r"    \renewcommand{\arraystretch}{1.1}",
    ] + TABLE_HEADER

    rem_in_data = [m for m in remaining_models if m in sgs4]
    for i, model in enumerate(rem_in_data):
        lines += _model_rows(model, sgs4, sgs5, sig4, sig5,
                             datasets, model_display, dataset_display, metrics)
        if i < len(rem_in_data) - 1:
            lines.append(r"        \midrule")

    lines += TABLE_FOOTER
    return "\n".join(lines)


# ── Generate the two split tables (need both open and closed data) ─────────────
_have_open   = "sgs4_mm"  in dir() or "sgs4_mm" in vars()
_have_closed = "sgs4_cl"  in dir() or "sgs4_cl" in vars()

# Use locals/globals to check variables set inside the if-block above
import builtins as _bi
_g = globals()
_have_open   = "sgs4_mm" in _g
_have_closed = "sgs4_cl" in _g

if _have_open and _have_closed:
    print("\n" + "="*70)
    print("  SPLIT TABLES")
    print("="*70)

    # ── Table 1: closed + representative open ─────────────────────────────────
    caption_main = (
        r"SGS per dataset for the three representative open-source models, "
        r"one per family, and the three closed-source models. "
        r"Lower values indicate greater generalisation robustness. "
        r"The shaded row pools all six datasets equally (three for closed models). "
        r"Full open-source results in Table~\ref{tab:sgs_minmax_full}. "
        r"\text{---}\ = metric not applicable. "
        r"Superscripts: $^{*}p<0.05$, $^{**}p<0.01$, $^{\dagger}p<0.001$."
    )
    latex_main = build_combined_latex(
        sgs4_cl, sgs5_cl, sig4_cl, sig5_cl,
        closed_models_in_data, closed_datasets_in_data,
        sgs4_mm, sgs5_mm, sig4_open, sig5_open,
        OPEN_REP_MODELS, open_datasets_in_data,
        CLOSED_MODEL_DISPLAY, OPEN_MODEL_DISPLAY,
        DATASET_DISPLAY, METRICS,
        caption_main, "tab:sgs_main",
    )
    (RESULTS_DIR / "sgs_table_main.tex").write_text(latex_main)
    print("[OK]  results/sgs_table_main.tex")

    # ── Table 2: remaining open models ────────────────────────────────────────
    caption_remain = (
        r"SGS per dataset for the remaining open-source models "
        r"(complement of Table~\ref{tab:sgs_main}). "
        r"Lower values indicate greater generalisation robustness. "
        r"The shaded row pools all six datasets equally. "
        r"Superscripts: $^{*}p<0.05$, $^{**}p<0.01$, $^{\dagger}p<0.001$."
    )
    latex_remain = build_remaining_latex(
        sgs4_mm, sgs5_mm, sig4_open, sig5_open,
        OPEN_REMAIN_MODELS, open_datasets_in_data,
        OPEN_MODEL_DISPLAY, DATASET_DISPLAY, METRICS,
        caption_remain, "tab:sgs_remaining",
    )
    (RESULTS_DIR / "sgs_table_remaining.tex").write_text(latex_remain)
    print("[OK]  results/sgs_table_remaining.tex")

elif not _have_open:
    print("\n[SKIP] Split tables require open-model data — run aggregate_results.py first.")

print("\n[DONE]")
