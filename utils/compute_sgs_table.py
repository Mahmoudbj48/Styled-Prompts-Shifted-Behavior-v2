"""
compute_sgs_table.py
--------------------
Driver for the paper-aligned SGS pipeline.

This is the SINGLE active entry point for SGS. The math primitives live in
``utils.sgs_core``. The legacy modules ``utils/significance_test.py`` and
``utils/compute_sgs_closed_table.py`` are deprecated shims that delegate
to this driver.

Run from project root:
    python utils/compute_sgs_table.py

Pipeline (matches paper appendix and ``sgs_core`` docstring):
  1. Load aggregated open + closed CSVs
  2. Sanity-check baseline deltas ~= 0
  3. Subsample TruthfulQA to 16 common prompts (seed=42)
  4. Normalise deltas by fixed R_X
  5. Per-prompt RMS instability s_X(i)
  6. SGS per dataset and total
  7. Fixed epsilon thresholds {0.01, 0.05, 0.10}, primary = 0.05
  8. One-sided one-sample t-test of s_X(i) vs eps

Outputs (paths relative to project root):
  results/sgs_table_main.tex                # closed + representative open, eps=0.05
  results/sgs_table_remaining.tex           # remaining open models, eps=0.05
  results/sgs_table_appendix_01.tex         # all models, eps=0.01
  results/sgs_table_appendix_10.tex         # all models, eps=0.10
  results/closed_models/sgs_closed_table.tex
  results/sgs/epsilon_appendix_table.tex
  results/sgs/{per_prompt_s,sgs_per_dataset,sgs_total_per_model,
               epsilon_table,significance_per_dataset,
               significance_per_model,epsilon_significance_summary,
               baseline_check}.csv
  results/sgs/chosen_truthful_qa_pids.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve project root irrespective of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

import sgs_core as sc  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
SGS_DIR     = RESULTS_DIR / "sgs"
SGS_TABLES  = SGS_DIR / "tables"
SGS_FIGURES = SGS_DIR / "figures"
CLOSED_DIR  = RESULTS_DIR / "closed_models"

# ---------------------------------------------------------------------------
# Display config
# ---------------------------------------------------------------------------

OPEN_MODEL_DISPLAY = {
    "L3.2-3B":   r"\textbf{L-3B}",
    "L3.1-8B":   r"\textbf{L-8B}",
    "G-2B":      r"\textbf{G-2B}",
    "G-7B":      r"\textbf{G-7B}",
    "G4-E4B":    r"\textbf{G4-E4B}",
    "Q2.5-1.5B": r"\textbf{Q-1.5B}",
    "Q2.5-7B":   r"\textbf{Q-7B}",
    "Q3.5-9B":   r"\textbf{Q3.5-9B}",
}
OPEN_MODEL_ORDER = list(OPEN_MODEL_DISPLAY.keys())

CLOSED_MODEL_DISPLAY = {
    "gpt-5.4":           r"\textbf{GPT-5.4}",
    "gemini-2.5-flash":  r"\textbf{Gemini-2.5-Flash}",
    "claude-sonnet-4-6": r"\textbf{Claude-Sonnet-4.6}",
}
CLOSED_MODEL_ORDER = list(CLOSED_MODEL_DISPLAY.keys())

# Three representative open models (one per family) for the main paper table
OPEN_REP_MODELS    = ["L3.1-8B", "G4-E4B", "Q3.5-9B"]
# Remaining open models go to the appendix "remaining" table
OPEN_REMAIN_MODELS = [m for m in OPEN_MODEL_ORDER if m not in OPEN_REP_MODELS]

DATASET_DISPLAY = {
    "TruthfulQA":        r"TruthfulQA \cite{lin2022truthfulqa}",
    "Natural Questions": r"Natural Questions \cite{kwiatkowski2019natural}",
    "Alpaca":            r"Alpaca \cite{taori2023stanford-alpaca}",
    "SimpleQA Verified": r"SimpleQA Verified \cite{haas_simpleqa_2026}",
    "TriviaQA":          r"TriviaQA \cite{joshi2017triviaqa}",
    "HotpotQA":          r"HotpotQA \cite{yang2018hotpotqa}",
}
DATASET_ORDER = list(DATASET_DISPLAY.keys())

AXIS_DISPLAY = {
    "delta_activation_similarity": r"$\Delta$-Cos",
    "delta_bleu":                  r"$\Delta$-BLEU",
    "delta_bertscore_response":    r"$\Delta$-BERT",
    "delta_log_prob":              r"$\Delta$-Prob",
    "delta_entropy":               r"$\Delta$-Ent",
    "delta_mirroring_rate":        r"$\Delta$-MR",
}
AXES = list(AXIS_DISPLAY.keys())     # same order as sgs_core.SGS_AXES

# Significance legend (used in every table footer)
SIG_LEGEND = (
    r"$^{*}p<0.05$, $^{**}p<0.01$, $^{\dagger}p<0.001$ "
    r"(one-sample one-sided $t$-test, $H_1\!:\mathrm{SGS}>\varepsilon$). "
    r"\xmark{} = axis unavailable for that model."
)


# ---------------------------------------------------------------------------
# Cell formatting
# ---------------------------------------------------------------------------

def _fmt_cell(sgs: float, marker: str) -> str:
    """Format one SGS cell. NaN -> ``\\xmark``, else rounded with optional
    significance marker as a math superscript.

    NB: ``\\xmark`` requires the user's preamble to define it, e.g.::
        \\usepackage{pifont}
        \\newcommand{\\xmark}{\\ding{55}}
    """
    if sgs is None or (isinstance(sgs, float) and np.isnan(sgs)):
        return r"\xmark{}"
    if marker:
        return f"{sgs:.3f}${marker}$"
    return f"{sgs:.3f}"


def _eps_tag(eps: float) -> str:
    """e.g. 0.05 -> '05' for filename / column suffix."""
    return f"{int(round(eps * 100)):02d}"


# ---------------------------------------------------------------------------
# Pivot helpers — long sgs_core output to (model, dataset)-keyed dicts
# ---------------------------------------------------------------------------

def _pivot_sgs_per_dataset(df_long: pd.DataFrame) -> dict:
    """Return nested dict[model][dataset][axis] = sgs (NaN if missing)."""
    out: dict = {}
    for (model, dataset), sub in df_long.groupby(["model", "dataset"]):
        out.setdefault(model, {})[dataset] = {
            r["axis"]: r["sgs"] for _, r in sub.iterrows()
        }
    return out


def _pivot_sgs_total(df_long: pd.DataFrame) -> dict:
    """Return dict[model][axis] = sgs."""
    out: dict = {}
    for model, sub in df_long.groupby("model"):
        out[model] = {r["axis"]: r["sgs"] for _, r in sub.iterrows()}
    return out


def _pivot_sig_per_dataset(df_long: pd.DataFrame, eps: float) -> dict:
    """Return dict[model][dataset][axis] = sig_marker for the requested eps."""
    tag = _eps_tag(eps)
    col = f"sig_{tag}"
    out: dict = {}
    for (model, dataset), sub in df_long.groupby(["model", "dataset"]):
        out.setdefault(model, {})[dataset] = {
            r["axis"]: (r[col] if col in sub.columns else "")
            for _, r in sub.iterrows()
        }
    return out


def _pivot_sig_total(df_long: pd.DataFrame, eps: float) -> dict:
    """Return dict[model][axis] = sig_marker."""
    tag = _eps_tag(eps)
    col = f"sig_{tag}"
    out: dict = {}
    for model, sub in df_long.groupby("model"):
        out[model] = {r["axis"]: (r[col] if col in sub.columns else "")
                      for _, r in sub.iterrows()}
    return out


# ---------------------------------------------------------------------------
# LaTeX builders
# ---------------------------------------------------------------------------

def _per_dataset_header(axes: list[str]) -> list[str]:
    """LaTeX header rows for a per-dataset table covering the given axes
    (subset of SGS_AXES, in display order). Auto-builds the column-group
    multicolumn row and the cmidrules so the closed-only variant (which
    drops Δ-Cos) renders correctly."""
    family_of = {
        "delta_activation_similarity": "Activation",
        "delta_bleu":                  "Generation Consistency",
        "delta_bertscore_response":    "Generation Consistency",
        "delta_log_prob":              "Confidence",
        "delta_entropy":               "Confidence",
        "delta_mirroring_rate":        "Mirroring",
    }
    n_ax = len(axes)
    colspec = "ll " + "c" * n_ax

    # Group axes (column 1=Model, column 2=Dataset, axes start at col 3)
    runs = []
    cur = None; cur_start = None
    for j, a in enumerate(axes, start=3):
        f = family_of[a]
        if f != cur:
            if cur is not None:
                runs.append((cur, cur_start, j - 1))
            cur, cur_start = f, j
    runs.append((cur, cur_start, n_ax + 2))

    fam_cells = [r"\textbf{Model}", r"\textbf{Dataset}"]
    for fam, cs, ce in runs:
        if cs == ce:
            fam_cells.append(rf"\textbf{{{fam}}}")
        else:
            fam_cells.append(rf"\multicolumn{{{ce - cs + 1}}}{{c}}{{\textbf{{{fam}}}}}")
    family_row = " & ".join(fam_cells) + r" \\"

    cmid_parts = [rf"\cmidrule(lr){{{cs}-{ce}}}" for _, cs, ce in runs]
    cmid_row = " ".join(cmid_parts)

    axis_row = " & ".join(["", ""] + [AXIS_DISPLAY[a] for a in axes]) + r" \\"

    return [
        rf"    \begin{{tabular}}{{{colspec}}}",
        r"        \toprule",
        f"        {family_row}",
        f"        {cmid_row}",
        f"        {axis_row}",
        r"        \midrule",
    ]


def _model_block(
    model: str,
    model_display: str,
    datasets: list[str],
    sgs_per_ds: dict, sig_per_ds: dict,
    sgs_total: dict, sig_total: dict,
    axes: list[str] | None = None,
) -> list[str]:
    """Build the LaTeX rows for a single model: dataset rows + Total SGS row.

    ``axes`` defaults to the full 6-axis set; pass a subset (e.g. without
    ``delta_activation_similarity``) for closed-only tables.
    """
    if axes is None:
        axes = AXES
    lines = [f"        % --- {model} ---"]
    lines.append(rf"        \multirow{{{len(datasets)}}}{{*}}{{{model_display}}}")
    for dataset in datasets:
        ds_sgs = sgs_per_ds.get(model, {}).get(dataset, {})
        ds_sig = sig_per_ds.get(model, {}).get(dataset, {})
        cells = " & ".join(
            _fmt_cell(ds_sgs.get(a, np.nan), ds_sig.get(a, ""))
            for a in axes
        )
        ds_label = DATASET_DISPLAY.get(dataset, dataset)
        lines.append(f"        & {ds_label} & {cells} \\\\")

    total_sgs = sgs_total.get(model, {})
    total_sig = sig_total.get(model, {})
    total_cells = " & ".join(
        r"\textbf{" + _fmt_cell(total_sgs.get(a, np.nan),
                                 total_sig.get(a, "")) + "}"
        for a in axes
    )
    lines.append(
        r"        \rowcolor[gray]{.95} \multicolumn{2}{r}{\textbf{Total SGS}} "
        f"& {total_cells} \\\\"
    )
    return lines


def _wrap_table(
    header_lines: list[str],
    body_lines: list[str],
    caption: str, label: str,
    width_full: bool = True,
    font_size: str = r"    \tiny",
) -> str:
    """Wrap header + body with LaTeX table boilerplate. ACL/EMNLP style:
    caption below, ``table*`` for full-page-width when ``width_full``, ``[t]``
    placement."""
    env = "table*" if width_full else "table"
    return "\n".join(
        [
            rf"\begin{{{env}}}[t]",
            r"    \centering",
            font_size,
            r"    \setlength{\tabcolsep}{3pt}",
            r"    \renewcommand{\arraystretch}{1.1}",
        ]
        + header_lines
        + body_lines
        + [
            r"        \bottomrule",
            r"    \end{tabular}",
            f"    \\caption{{{caption}}}",
            f"    \\label{{{label}}}",
            rf"\end{{{env}}}",
        ]
    )


def build_per_dataset_table(
    models_in_order: list[tuple[str, str]],
    sgs_per_ds: dict, sig_per_ds: dict,
    sgs_total: dict, sig_total: dict,
    datasets: list[str],
    caption: str, label: str,
    axes: list[str] | None = None,
    section_headers: list[tuple[int, str]] | None = None,
) -> str:
    """Per-dataset breakdown table.

    ``axes`` is a subset of SGS_AXES in display order (defaults to all 6).
    ``section_headers`` inserts ``\\multicolumn`` group labels between
    model blocks (used to split closed vs. open in combined tables)."""
    if axes is None:
        axes = AXES
    section_headers = section_headers or []
    sec_by_idx = dict(section_headers)
    n_cols = 2 + len(axes)
    body: list[str] = []
    for i, (model, display) in enumerate(models_in_order):
        if i in sec_by_idx:
            if i > 0:
                body.append(r"        \midrule")
            body.append(
                rf"        \multicolumn{{{n_cols}}}{{l}}"
                rf"{{\textit{{{sec_by_idx[i]}}}}} \\"
            )
            body.append(r"        \midrule")
        elif i > 0:
            body.append(r"        \midrule")
        body += _model_block(model, display, datasets,
                             sgs_per_ds, sig_per_ds,
                             sgs_total, sig_total, axes=axes)
    header = _per_dataset_header(axes)
    return _wrap_table(header, body, caption, label,
                       width_full=True, font_size=r"    \tiny")


# ---------------------------------------------------------------------------
# Totals-only tables (compact main-paper tables)
# ---------------------------------------------------------------------------

def _totals_header(axes: list[str]) -> list[str]:
    """LaTeX header for a totals-only table covering ``axes`` (subset of
    SGS_AXES, in display order). Builds the column-group multicolumn row
    based on which family each axis belongs to."""
    # ColSpec: `l` for the model column, then one `c` per axis
    colspec = "l " + "c" * len(axes)

    # Group axes into the same families used in build_per_dataset_table
    family_of = {
        "delta_activation_similarity": "Activation",
        "delta_bleu":                  "Generation Consistency",
        "delta_bertscore_response":    "Generation Consistency",
        "delta_log_prob":              "Confidence",
        "delta_entropy":               "Confidence",
        "delta_mirroring_rate":        "Mirroring",
    }

    # Build the family multicolumn row + cmidrules dynamically
    fam_runs: list[tuple[str, int, int]] = []   # (family, start_col, end_col) 1-indexed (col 1 = Model)
    cur_fam = None
    cur_start = None
    for j, ax in enumerate(axes, start=2):     # axes start at column 2 (after Model)
        fam = family_of[ax]
        if fam != cur_fam:
            if cur_fam is not None:
                fam_runs.append((cur_fam, cur_start, j - 1))
            cur_fam = fam
            cur_start = j
    fam_runs.append((cur_fam, cur_start, len(axes) + 1))

    cells = [r"\textbf{Model}"]
    for fam, cs, ce in fam_runs:
        if cs == ce:
            cells.append(rf"\textbf{{{fam}}}")
        else:
            cells.append(rf"\multicolumn{{{ce - cs + 1}}}{{c}}{{\textbf{{{fam}}}}}")
    family_row = " & ".join(cells) + r" \\"

    cmid_parts = []
    for fam, cs, ce in fam_runs:
        cmid_parts.append(rf"\cmidrule(lr){{{cs}-{ce}}}")
    cmid_row = " ".join(cmid_parts)

    axis_row = " & ".join([""] + [AXIS_DISPLAY[a] for a in axes]) + r" \\"

    return [
        rf"    \begin{{tabular}}{{{colspec}}}",
        r"        \toprule",
        f"        {family_row}",
        f"        {cmid_row}",
        f"        {axis_row}",
        r"        \midrule",
    ]


def _totals_body(
    models_in_order: list[tuple[str, str]],
    axes: list[str],
    sgs_total: dict,
    sig_total: dict,
) -> list[str]:
    """One row per model -- model display label + value^marker per axis."""
    lines: list[str] = []
    for model, display in models_in_order:
        cells = [display]
        for a in axes:
            sgs = sgs_total.get(model, {}).get(a, np.nan)
            mark = sig_total.get(model, {}).get(a, "")
            cells.append(_fmt_cell(sgs, mark))
        lines.append(f"        " + " & ".join(cells) + r" \\")
    return lines


def build_totals_only_table(
    models_in_order: list[tuple[str, str]],
    axes: list[str],
    sgs_total: dict, sig_total: dict,
    caption: str, label: str,
) -> str:
    """Compact totals-only table (no per-dataset rows). Kept for backward
    compatibility -- the current paper layout uses per-dataset tables for
    closed and open instead."""
    header = _totals_header(axes)
    body = _totals_body(models_in_order, axes, sgs_total, sig_total)
    return _wrap_table(header, body, caption, label,
                       width_full=True, font_size=r"    \small")


def build_epsilon_appendix_table(eps_summary: pd.DataFrame) -> str:
    """One-row-per-axis LaTeX table summarising significance counts at each
    fixed epsilon threshold.

    Columns: ``axis``, then for each eps in {0.01, 0.05, 0.10} the count of
    (provider, model) cells reaching p<0.05, p<0.01, p<0.001.
    """
    epsilons = sc.FIXED_EPSILONS
    n_eps = len(epsilons)

    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{4pt}",
        rf"    \begin{{tabular}}{{l{'c' * n_eps}}}",
        r"        \toprule",
    ]

    eps_labels = " & ".join(rf"$\varepsilon={eps:.2f}$" for eps in epsilons)
    lines.append(rf"        \textbf{{Axis}} & {eps_labels} \\")
    lines.append(r"        \midrule")

    for _, row in eps_summary.iterrows():
        axis_label = AXIS_DISPLAY.get(row["axis"], row["axis"])
        cells = []
        for eps in epsilons:
            tag = _eps_tag(eps)
            n5  = int(row.get(f"n_sig05_{tag}",   0))
            n1  = int(row.get(f"n_sig01_{tag}",   0))
            n01 = int(row.get(f"n_sig001_{tag}",  0))
            cells.append(f"{n5} / {n1} / {n01}")
        lines.append(f"        {axis_label} & " + " & ".join(cells) + r" \\")

    cap = (
        r"Significance counts by axis at each fixed-$\varepsilon$ threshold. "
        r"Each cell shows the number of (provider, model) pairs with $p<0.05$ / "
        r"$p<0.01$ / $p<0.001$ in the one-sided $t$-test of "
        r"$\mathrm{SGS}_X>\varepsilon$ ($n$ = number of models with that "
        r"axis available). "
        r"Format: $\#(p<0.05)\,/\,\#(p<0.01)\,/\,\#(p<0.001)$."
    )
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        f"    \\caption{{{cap}}}",
        r"    \label{tab:sgs_epsilon_robustness}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stale-artefact cleanup
# ---------------------------------------------------------------------------

STALE_FILES = [
    RESULTS_DIR / "sgs_table.tex",
    RESULTS_DIR / "sgs_table_meanabs.tex",
    RESULTS_DIR / "sgs_table_minmax.tex",
    RESULTS_DIR / "sgs_scores_meanabs.csv",
    RESULTS_DIR / "sgs_scores_minmax.csv",
    RESULTS_DIR / "normalization_params.csv",
    RESULTS_DIR / "significance_by_dataset.csv",
    RESULTS_DIR / "significance_by_model.csv",
    # Old combined / split / totals tables replaced by the per-dataset
    # tables now living in results/sgs/tables/
    RESULTS_DIR / "sgs_table_main.tex",
    RESULTS_DIR / "sgs_table_remaining.tex",
    RESULTS_DIR / "sgs_table_appendix.tex",
    RESULTS_DIR / "sgs_table_appendix_01.tex",
    RESULTS_DIR / "sgs_table_appendix_10.tex",
    RESULTS_DIR / "sgs_table_appendix_eps01.tex",
    RESULTS_DIR / "sgs_table_appendix_eps10.tex",
    RESULTS_DIR / "sgs_table_open_main.tex",
    RESULTS_DIR / "sgs_table_open_eps01.tex",
    RESULTS_DIR / "sgs_table_open_eps10.tex",
    RESULTS_DIR / "sgs_table_closed_main.tex",
    RESULTS_DIR / "sgs_table_closed_eps01.tex",
    RESULTS_DIR / "sgs_table_closed_eps10.tex",
    SGS_DIR     / "epsilon_appendix_table.tex",   # moved to tables/epsilon_robustness.tex
    CLOSED_DIR  / "sgs_closed_table.tex",
    CLOSED_DIR  / "sgs_closed_scores_normalized.csv",
    CLOSED_DIR  / "sgs_closed_normalization_params.csv",
    CLOSED_DIR  / "sgs_per_family_closed.csv",
    CLOSED_DIR  / "significance_closed_by_dataset.csv",
    CLOSED_DIR  / "significance_closed_by_model.csv",
]


def remove_stale_outputs() -> None:
    """Delete LaTeX/CSV files from the previous data-dependent normalisation
    pipeline so there is no ambiguity about which pipeline produced which
    artefacts."""
    deleted = []
    for p in STALE_FILES:
        if p.exists():
            p.unlink()
            deleted.append(p.relative_to(PROJECT_ROOT))
    if deleted:
        print("[clean] removed stale artefacts:")
        for p in deleted:
            print(f"          - {p}")
    else:
        print("[clean] no stale artefacts present")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> None:
    SGS_DIR.mkdir(parents=True, exist_ok=True)
    SGS_TABLES.mkdir(parents=True, exist_ok=True)
    SGS_FIGURES.mkdir(parents=True, exist_ok=True)
    CLOSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  SGS PIPELINE  (fixed R_X normalisation, RMS, fixed epsilon)")
    print("=" * 72)

    res = sc.run_full_pipeline()

    # ─── Step A: dump CSVs ────────────────────────────────────────────────
    res["baseline_violations"].to_csv(SGS_DIR / "baseline_check.csv", index=False)
    res["per_prompt"].to_csv(SGS_DIR / "per_prompt_s.csv", index=False)
    res["sgs_per_dataset"].to_csv(SGS_DIR / "sgs_per_dataset.csv", index=False)
    res["sgs_total"].to_csv(SGS_DIR / "sgs_total_per_model.csv", index=False)
    res["epsilons"].to_csv(SGS_DIR / "epsilon_table.csv", index=False)
    res["sig_dataset"].to_csv(SGS_DIR / "significance_per_dataset.csv", index=False)
    res["sig_model"].to_csv(SGS_DIR / "significance_per_model.csv", index=False)
    res["eps_summary"].to_csv(SGS_DIR / "epsilon_significance_summary.csv", index=False)
    (SGS_DIR / "chosen_truthful_qa_pids.json").write_text(
        json.dumps(res["tqa_record"], indent=2)
    )
    print("\n[write]   results/sgs/*.csv  +  chosen_truthful_qa_pids.json")

    # ─── Step B: pivot for table builders ─────────────────────────────────
    sgs_per_ds = _pivot_sgs_per_dataset(res["sgs_per_dataset"])
    sgs_tot    = _pivot_sgs_total(res["sgs_total"])

    # ─── Step C: produce LaTeX tables ─────────────────────────────────────
    # All paper-facing tables now live in results/sgs/tables/.
    #   Table 1 (main body):  closed per-dataset (3 datasets, no Δ-Cos)
    #   Table 2 (appendix):   open per-dataset   (6 datasets, all 6 axes)
    #   Table 3 (appendix):   epsilon robustness counts
    closed_models_in_data = [m for m in CLOSED_MODEL_ORDER if m in sgs_tot]
    open_all_in_data      = [m for m in OPEN_MODEL_ORDER   if m in sgs_tot]

    AXES_ALL    = AXES                                                        # all 6
    AXES_CLOSED = [a for a in AXES if a != "delta_activation_similarity"]     # 5 (no Δ-Cos)
    CLOSED_DATASETS_MAIN = ["TruthfulQA", "Alpaca", "SimpleQA Verified"]      # main-paper Table 1

    sig_per_ds_primary = _pivot_sig_per_dataset(res["sig_dataset"], sc.PRIMARY_EPSILON)
    sig_tot_primary    = _pivot_sig_total(res["sig_model"], sc.PRIMARY_EPSILON)

    # ── Table 1: closed models, per-dataset on 3 main-paper datasets ─────
    models_closed = [(m, CLOSED_MODEL_DISPLAY[m]) for m in closed_models_in_data]
    cap1 = (
        rf"Per-dataset SGS for the three closed-source models on the three "
        rf"datasets common to all closed providers, at $\varepsilon="
        rf"{sc.PRIMARY_EPSILON:g}$. The activation axis ($\Delta$-Cos) is "
        rf"omitted because it requires white-box access (unavailable for all "
        rf"closed models). $\Delta$-Prob and $\Delta$-Ent are available only "
        rf"for GPT-5.4. " + SIG_LEGEND
    )
    target1 = SGS_TABLES / "sgs_closed_per_dataset.tex"
    latex1 = build_per_dataset_table(
        models_closed, sgs_per_ds, sig_per_ds_primary, sgs_tot, sig_tot_primary,
        CLOSED_DATASETS_MAIN, cap1, "tab:sgs_closed_per_dataset",
        axes=AXES_CLOSED,
    )
    target1.write_text(latex1)
    print(f"[write]   {target1.relative_to(PROJECT_ROOT)}")

    # ── Table 2 (appendix): open models, per-dataset on all 6 datasets ──
    models_open = [(m, OPEN_MODEL_DISPLAY[m]) for m in open_all_in_data]
    cap2 = (
        rf"Per-dataset SGS for the eight open-source models on all six "
        rf"evaluation datasets, at $\varepsilon={sc.PRIMARY_EPSILON:g}$. "
        rf"The shaded ``Total SGS'' row pools prompts across all datasets. "
        + SIG_LEGEND
    )
    target2 = SGS_TABLES / "sgs_open_per_dataset.tex"
    latex2 = build_per_dataset_table(
        models_open, sgs_per_ds, sig_per_ds_primary, sgs_tot, sig_tot_primary,
        DATASET_ORDER, cap2, "tab:sgs_open_per_dataset",
        axes=AXES_ALL,
    )
    target2.write_text(latex2)
    print(f"[write]   {target2.relative_to(PROJECT_ROOT)}")

    # ─── Step D: epsilon robustness table (Table 3, appendix) ─────────────
    eps_latex = build_epsilon_appendix_table(res["eps_summary"])
    target = SGS_TABLES / "epsilon_robustness.tex"
    target.write_text(eps_latex)
    print(f"[write]   {target.relative_to(PROJECT_ROOT)}")

    # ─── Step E: clean up artefacts from the old pipeline ─────────────────
    remove_stale_outputs()

    # ─── Step F: summary printout ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Total SGS preview (primary eps = 0.05)")
    print("=" * 72)
    piv = res["sgs_total"].pivot_table(
        index=["provider", "model"], columns="axis", values="sgs"
    ).round(4)
    piv.columns = [c.replace("delta_", "")
                    .replace("_similarity", "")
                    .replace("_response", "")
                    .replace("_rate", "")
                   for c in piv.columns]
    print(piv.to_string())
    print("\n[done]")


if __name__ == "__main__":
    main()
