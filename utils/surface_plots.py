# utils/surface_plots.py
"""
NeurIPS-style plotting utilities for surface noise experiments (spacing, punctuation, letter_case).

Goals (NeurIPS 2025 template-friendly figures):
- Clean, legible, publication-ready.
- No tiny text; consistent font sizes.
- No clutter; legend outside or compact.
- White background, thin grid (optional), clear axes labels.
- Suitable DPI and tight bounding boxes for LaTeX \includegraphics[width=...].

Used in two ways:
1) Imported by experiments/{spacing,punctuation,letter_case}.py to plot the run outputs (by loading saved CSVs).
2) Run standalone to combine multiple runs / multiple CSVs into one set of plots.

Standalone example:
  python utils/surface_plots.py \
    --plot_inputs results/spacing/run_multi_truthful_qa_20260101_120000 \
                 results/spacing/run_multi_truthful_qa_20260101_121000 \
    --plot_out_dir results/spacing/combined_plots \
    --dataset truthful_qa \
    --places prefix suffix global \
    --strengths 0 1 5 20 50 100 \
    --save_pdf
"""
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse


# --------------------------
# NeurIPS-style matplotlib defaults
# --------------------------
def apply_neurips_style() -> None:
    """
    Apply conservative, LaTeX-friendly style that matches NeurIPS paper aesthetics:
    - readable fonts
    - clean lines
    - no heavy background
    """
    plt.rcParams.update({
        # Figure
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        # Fonts: Matplotlib will pick a sensible serif/sans; keep sizes NeurIPS-readable.
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,

        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 4,

        # Axes
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Grid (very light)
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.7,

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # Tight layout behavior
        "figure.autolayout": False,
    })


PLOT_METRICS_DEFAULT = [
    "bertscore_prompt",
    "bleu",
    "bertscore_response",
    "delta_log_prob",
    "entropy_shift",
    "jsd_drift",
    "activation_similarity",
    # "mirroring_rate",
]


# --------------------------
# I/O helpers
# --------------------------
def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_results_csvs(paths_or_dirs: List[str]) -> pd.DataFrame:
    """
    Accepts:
      - run directories containing full_results_all_models.csv or *_results.csv
      - direct CSV file paths
    Returns merged per-example dataframe.
    Adds `run_id` from directory name for traceability.
    """
    dfs = []
    for p in paths_or_dirs:
        p = os.path.expanduser(p)

        if os.path.isfile(p) and p.endswith(".csv"):
            df = pd.read_csv(p)
            run_id = os.path.basename(os.path.dirname(p)) or "run"
            df["run_id"] = run_id
            dfs.append(df)
            continue

        if os.path.isdir(p):
            run_id = os.path.basename(os.path.normpath(p))
            preferred = os.path.join(p, "full_results_all_models.csv")
            if os.path.exists(preferred):
                df = pd.read_csv(preferred)
                df["run_id"] = run_id
                dfs.append(df)
                continue

            # fallback: per-model CSVs
            cand = []
            for fn in os.listdir(p):
                if fn.endswith("_results.csv"):
                    cand.append(os.path.join(p, fn))
            for fp in sorted(cand):
                df = pd.read_csv(fp)
                df["run_id"] = run_id
                dfs.append(df)
            continue

        raise FileNotFoundError(f"plot input not found: {p}")

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# --------------------------
# Aggregation helpers
# --------------------------
def build_means_from_rows(df_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Build df_mean from per-example rows.
    Also recompute mirroring_rate from verdicts if available.
    """
    if df_rows.empty:
        return pd.DataFrame()

    needed = {"model", "place", "strength"}
    missing = [c for c in needed if c not in df_rows.columns]
    if missing:
        raise ValueError(f"Cannot build means: missing columns {missing}")

    metric_cols = [c for c in [
        "bertscore_prompt",
        "bleu",
        "bertscore_response",
        "delta_log_prob",
        "entropy_shift",
        "jsd_drift",
        "activation_similarity",
        "mirroring_rate_batch",
    ] if c in df_rows.columns]

    df_rows = _coerce_numeric(df_rows, ["strength"] + metric_cols)

    df_mean = (
        df_rows.groupby(["model", "place", "strength"], dropna=False)[metric_cols]
        .mean()
        .reset_index()
        .sort_values(["model", "place", "strength"])
    )

    # Preferred: recompute mirroring_rate from per-example verdicts
    if "mirroring_verdict" in df_rows.columns:
        tmp = df_rows[df_rows["mirroring_verdict"].isin(["YES", "NO"])].copy()
        if not tmp.empty:
            agg = (
                tmp.groupby(["model", "place", "strength"], dropna=False)["mirroring_verdict"]
                .value_counts()
                .unstack(fill_value=0)
                .reset_index()
            )
            if "YES" not in agg.columns:
                agg["YES"] = 0
            if "NO" not in agg.columns:
                agg["NO"] = 0
            agg["mirroring_yes"] = agg["YES"].astype(int)
            agg["mirroring_total"] = (agg["YES"] + agg["NO"]).astype(int)
            agg["mirroring_rate"] = agg["mirroring_yes"] / agg["mirroring_total"].replace(0, np.nan)

            df_mean = df_mean.merge(
                agg[["model", "place", "strength", "mirroring_yes", "mirroring_total", "mirroring_rate"]],
                on=["model", "place", "strength"],
                how="left",
            )

    return df_mean


# --------------------------
# Plotting
# --------------------------
def _format_axis(ax, xlabel: str, ylabel: str, title: Optional[str] = None) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Keep grid light and behind data
    ax.set_axisbelow(True)

    # Ensure integer ticks on x if x is strength
    try:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    except Exception:
        pass


def _save_figure(fig, out_png: str, *, save_pdf: bool, dpi: int = 300) -> None:
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        out_pdf = os.path.splitext(out_png)[0] + ".pdf"
        fig.savefig(out_pdf, bbox_inches="tight")


def plot_metric_lines_general(
        df_mean: pd.DataFrame,
        metric: str,
        strengths: Optional[List[int]],
        out_path_png: str,
        *,
        dataset_name: str,
        style_name: str = "surface_noise",
        places: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        include_title: bool = False,
        legend_outside: bool = True,
        save_pdf: bool = False,
) -> None:
    """
    NeurIPS-friendly line plot: metric vs strength.
    Legend placed outside (recommended for NeurIPS to avoid clutter).
    """
    if df_mean.empty or metric not in df_mean.columns:
        return

    df = df_mean.copy()
    if places:
        df = df[df["place"].isin(places)]
    if models:
        df = df[df["model"].isin(models)]
    if df.empty:
        return

    if strengths is None:
        strengths_sorted = sorted(df["strength"].dropna().unique().tolist())
    else:
        strengths_sorted = sorted(list(set(int(x) for x in strengths)))

    # NeurIPS columns are narrow; choose modest figure size.
    # ~3.4in width is single-column; ~7in is double-column.
    # Here we default to double-column friendly.
    fig_w, fig_h = 6.8, 2.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Plot per (model, place)
    for model in sorted(df["model"].unique().tolist()):
        for place in sorted(df["place"].unique().tolist()):
            sub = df[(df["model"] == model) & (df["place"] == place)].copy()
            if sub.empty:
                continue
            sub = sub.set_index("strength").reindex(strengths_sorted)
            y = sub[metric].values
            ax.plot(strengths_sorted, y, marker="o", label=f"{model}/{place}")

    title = f"{metric} vs strength ({style_name}, {dataset_name})" if include_title else None
    _format_axis(ax, xlabel="Strength", ylabel=metric, title=title)

    # Legend
    if legend_outside:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=False,
            ncol=1,
        )
    else:
        ax.legend(frameon=False)

    fig.tight_layout()
    _save_figure(fig, out_png=out_path_png, save_pdf=save_pdf)
    plt.close(fig)


def make_all_plots_from_csvs(
        *,
        plot_inputs: List[str],
        out_dir: str,
        strengths: Optional[List[int]],
        places_filter: Optional[List[str]],
        models_filter: Optional[List[str]],
        dataset_name: str,
        style_name: str = "surface_noise",
        save_pdf: bool = False,
        include_title: bool = False,
        legend_outside: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CSVs, compute means, save combined means, and make NeurIPS-friendly plots.
    Returns (df_rows, df_mean).
    """
    apply_neurips_style()
    os.makedirs(out_dir, exist_ok=True)

    df_rows = load_results_csvs(plot_inputs)
    if df_rows.empty:
        raise SystemExit("Plotting: no rows loaded from plot inputs.")

    df_mean = build_means_from_rows(df_rows)

    mean_path = os.path.join(out_dir, "combined_means_by_model_place_strength.csv")
    df_mean.to_csv(mean_path, index=False)
    print(f"✓ Saved combined means table: {mean_path}")

    for metric in PLOT_METRICS_DEFAULT:
        if metric not in df_mean.columns:
            continue
        out_png = os.path.join(out_dir, f"{metric}_vs_strength.png")
        plot_metric_lines_general(
            df_mean=df_mean,
            metric=metric,
            strengths=strengths,
            out_path_png=out_png,
            dataset_name=dataset_name,
            style_name=style_name,
            places=places_filter,
            models=models_filter,
            include_title=include_title,
            legend_outside=legend_outside,
            save_pdf=save_pdf,
        )
        print(f"✓ Plot saved: {out_png}" + (" (+pdf)" if save_pdf else ""))

    return df_rows, df_mean


# --------------------------
# Standalone CLI
# --------------------------
def _select_strengths_for_plot(
        strengths: Optional[List[int]],
        strength_range: Optional[Tuple[int, int]],
        strength_step: int,
) -> Optional[List[int]]:
    if strengths:
        return [int(x) for x in strengths]
    if strength_range is not None:
        lo, hi = strength_range
        if strength_step <= 0:
            raise ValueError("--strength_step must be >= 1")
        if lo > hi:
            lo, hi = hi, lo
        return list(range(int(lo), int(hi) + 1, int(strength_step)))
    return None  # auto from CSV


def main():
    parser = argparse.ArgumentParser(description="Plot surface noise experiment results by loading saved CSVs (NeurIPS-style).")
    parser.add_argument("--plot_inputs", nargs="+", required=True,
                        help="Run dirs or CSV paths to combine.")
    parser.add_argument("--plot_out_dir", type=str, required=True,
                        help="Output directory for combined plots + combined means CSV.")
    parser.add_argument("--dataset", type=str, default="unknown_dataset")
    parser.add_argument("--style_name", type=str, default="surface_noise",
                        help="Style name for plot titles (spacing, punctuation, letter_case, or surface_noise)")

    parser.add_argument("--places", nargs="+", default=None,
                        help="Filter places to plot (e.g. prefix suffix global).")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter models to plot (e.g. L3.2-1B).")

    parser.add_argument("--strengths", nargs="+", type=int, default=None)
    parser.add_argument("--strength_range", nargs=2, type=int, default=None, metavar=("LO", "HI"))
    parser.add_argument("--strength_step", type=int, default=1)

    # NeurIPS-friendly export options
    parser.add_argument("--save_pdf", action="store_true", help="Also save a PDF for each figure (recommended for LaTeX).")
    parser.add_argument("--include_title", action="store_true", help="Add titles on figures (often omitted in NeurIPS; captions used instead).")
    parser.add_argument("--legend_inside", action="store_true", help="Put legend inside the plot (default is outside).")

    args = parser.parse_args()

    strengths = _select_strengths_for_plot(
        strengths=args.strengths,
        strength_range=tuple(args.strength_range) if args.strength_range else None,
        strength_step=int(args.strength_step),
    )

    make_all_plots_from_csvs(
        plot_inputs=args.plot_inputs,
        out_dir=args.plot_out_dir,
        strengths=strengths,
        places_filter=args.places,
        models_filter=args.models,
        dataset_name=args.dataset,
        style_name=args.style_name,
        save_pdf=bool(args.save_pdf),
        include_title=bool(args.include_title),
        legend_outside=not bool(args.legend_inside),
    )


if __name__ == "__main__":
    main()