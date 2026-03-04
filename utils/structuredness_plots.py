"""
utils/structuredness_plots.py

Plotting utilities for structuredness-style experiments
(length variation, inter_vs_imper).

Designed to be:
  1. Imported by experiment files (e.g. experiments/length_variation.py,
     experiments/interrogative_vs_imperative.py) after the run to generate
     plots from the results DataFrame.
  2. Run standalone on saved CSVs / run directories.

Standalone example:
  python utils/structuredness_plots.py \
    --plot_inputs results/length_variation/run_multi_truthful_qa_20260225_120000 \
    --plot_out_dir results/length_variation/combined_plots \
    --dataset truthful_qa \
    --style_name length_variation
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from utils.plot_utils import (
    apply_neurips_style,
    _coerce_numeric,
    load_results_csvs,
    _format_axis,
    _save_figure,
)


# =========================================================================
# Default metrics to plot (same set as surface_plots for consistency)
# =========================================================================

PLOT_METRICS_DEFAULT = [
    "bertscore_prompt",
    "bleu",
    "bertscore_response",
    "delta_log_prob",
    "entropy_shift",
    "jsd_drift",
    "activation_similarity",
    "mirroring_rate",
]


# =========================================================================
# I/O helpers
# =========================================================================

def _is_numeric(val) -> bool:
    """Return True if val can be interpreted as a number."""
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False




# =========================================================================
# Aggregation helpers
# =========================================================================

def build_means_from_rows(df_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Build df_mean from per-example rows.
    Also recompute mirroring_rate from verdicts if available.

    Works for both numeric strengths (length variation multipliers)
    and categorical strengths (inter_vs_imper modes).
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

    # Only coerce strength to numeric if all values *are* numeric
    raw_strengths = df_rows["strength"].dropna().unique().tolist()
    all_numeric = all(_is_numeric(s) for s in raw_strengths)
    cols_to_coerce = list(metric_cols)
    if all_numeric:
        cols_to_coerce.append("strength")
    df_rows = _coerce_numeric(df_rows, cols_to_coerce)

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


# =========================================================================
# Axis / figure helpers
# =========================================================================

def _format_axis(ax, xlabel: str, ylabel: str, title: Optional[str] = None) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_axisbelow(True)


# =========================================================================
# Metric line plot  (metric vs strength / mode)
# =========================================================================

def plot_metric_lines(
        df_mean: pd.DataFrame,
        metric: str,
        strengths: Optional[list],
        out_path_png: str,
        *,
        dataset_name: str,
        style_name: str = "structuredness",
        places: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        include_title: bool = False,
        legend_outside: bool = True,
        save_pdf: bool = False,
        ylabel: Optional[str] = None,
) -> None:
    """
    NeurIPS-friendly line plot: metric vs strength (numeric or categorical).

    Handles:
      - numeric x-axis (length multipliers like 0.25, 0.5, 1.5, …)
      - categorical x-axis (modes like "original", "interrogative", "imperative" for inter_vs_imper)
        with "original" always placed left-most.
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

    # --- determine x-axis values and order ---
    if strengths is None:
        raw_strengths = df["strength"].dropna().unique().tolist()
        all_numeric = all(_is_numeric(s) for s in raw_strengths)
        if all_numeric:
            strengths_sorted = sorted(raw_strengths, key=lambda x: float(x))
        else:
            # "original" first, then alphabetical
            strengths_sorted = sorted(
                raw_strengths,
                key=lambda x: (0, x) if str(x).lower() == "original" else (1, str(x).lower()),
            )
    else:
        all_numeric = all(_is_numeric(s) for s in strengths)
        if all_numeric:
            strengths_sorted = sorted(strengths, key=lambda x: float(x))
        else:
            strengths_sorted = sorted(
                strengths,
                key=lambda x: (0, x) if str(x).lower() == "original" else (1, str(x).lower()),
            )

    is_categorical = any(not _is_numeric(s) for s in strengths_sorted)

    # --- figure ---
    fig_w, fig_h = 6.8, 2.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if is_categorical:
        x_positions = list(range(len(strengths_sorted)))
    else:
        x_positions = [float(s) for s in strengths_sorted]

    for model in sorted(df["model"].unique().tolist()):
        for place in sorted(df["place"].unique().tolist()):
            sub = df[(df["model"] == model) & (df["place"] == place)].copy()
            if sub.empty:
                continue
            sub = sub.set_index("strength").reindex(strengths_sorted)
            y = sub[metric].values
            ax.plot(x_positions, y, marker="o", label=f"{model}/{place}")

    if is_categorical:
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(s) for s in strengths_sorted])

    xlabel = "Mode" if is_categorical else "Multiplier"
    ylabel_final = ylabel if ylabel is not None else metric
    title = f"{metric} vs {xlabel.lower()} ({style_name}, {dataset_name})" if include_title else None
    _format_axis(ax, xlabel=xlabel, ylabel=ylabel_final, title=title)

    if not is_categorical:
        try:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=False))
        except Exception:
            pass

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


# =========================================================================
# make_all_structuredness_metric_plots  (replaces make_all_plots_from_csvs
#   from surface_plots for structured experiments)
# =========================================================================

def make_all_structuredness_metric_plots(
        *,
        plot_inputs: List[str],
        out_dir: str,
        strengths: Optional[list] = None,
        places_filter: Optional[List[str]] = None,
        models_filter: Optional[List[str]] = None,
        dataset_name: str,
        style_name: str = "structuredness",
        save_pdf: bool = False,
        include_title: bool = False,
        legend_outside: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CSVs, compute means, save combined means, and make NeurIPS-friendly
    metric-vs-strength line plots for structured style experiments.

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
        plot_metric_lines(
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


# =========================================================================
# Length-ratio boxplot
# =========================================================================

def plot_length_ratio_boxplot(
        df: pd.DataFrame,
        out_path: str,
        *,
        dataset_name: str = "",
        models_filter: Optional[List[str]] = None,
        include_title: bool = False,
        save_pdf: bool = False,
) -> None:
    """
    NeurIPS-friendly boxplot: for each length multiplier, show the
    distribution of actual word-count ratio (styled / original).

    Red × markers show the requested (ideal) multiplier for reference.

    Args:
        df: Experiment results DataFrame.
            Expected columns: strength, prompt_orig, prompt_pert
            Optional: model (used for filtering)
        out_path: Path to save the PNG.
        dataset_name: For the plot title (only shown when include_title=True).
        models_filter: If given, keep only these models.
        include_title: Whether to add a title (NeurIPS papers typically
            omit titles and rely on LaTeX captions instead).
        save_pdf: Also save a PDF alongside the PNG.
    """
    apply_neurips_style()

    d = df.copy()

    # Filter models if requested
    if models_filter and "model" in d.columns:
        d = d[d["model"].isin(models_filter)]

    # Need these columns
    for col in ("strength", "prompt_orig", "prompt_pert"):
        if col not in d.columns:
            print(f"⚠️  plot_length_ratio_boxplot: missing column '{col}', skipping.")
            return

    d["strength"] = pd.to_numeric(d["strength"], errors="coerce")
    d = d.dropna(subset=["strength", "prompt_orig", "prompt_pert"])
    if d.empty:
        return

    # Compute word-count ratio
    d["orig_words"] = d["prompt_orig"].apply(lambda s: len(str(s).split()))
    d["styled_words"] = d["prompt_pert"].apply(lambda s: len(str(s).split()))
    d["length_ratio"] = d["styled_words"] / d["orig_words"].clip(lower=1)

    multipliers_sorted = sorted(d["strength"].unique())
    if not multipliers_sorted:
        return

    data = [d[d["strength"] == m]["length_ratio"].values for m in multipliers_sorted]
    labels = [str(m) for m in multipliers_sorted]

    # NeurIPS double-column friendly figure size
    fig, ax = plt.subplots(figsize=(6.8, 2.8))

    lw = 0.8  # match axes.linewidth from NeurIPS style

    bp = ax.boxplot(
        data,
        positions=list(range(len(multipliers_sorted))),
        widths=0.4,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(
            marker="D", markerfacecolor="white",
            markeredgecolor="black", markersize=4,
        ),
        boxprops=dict(linewidth=lw),
        whiskerprops=dict(linewidth=lw),
        capprops=dict(linewidth=lw),
        medianprops=dict(linewidth=lw, color="black"),
        flierprops=dict(marker=".", markersize=3, alpha=0.5),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.35)

    # Reference: requested multiplier (ideal target)
    ax.scatter(
        list(range(len(multipliers_sorted))),
        multipliers_sorted,
        marker="x", color="#d62728", s=40, linewidths=1.0, zorder=5,
        label="Requested multiplier",
    )

    ax.set_xticks(list(range(len(multipliers_sorted))))
    ax.set_xticklabels(labels)

    title_str = None
    if include_title:
        title_str = "Actual vs. Requested Length Ratio"
        if dataset_name:
            title_str += f" ({dataset_name})"
    _format_axis(ax, xlabel="Requested Length Multiplier",
                 ylabel="Actual Length Ratio (styled / original)",
                 title=title_str)

    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    _save_figure(fig, out_path, save_pdf=save_pdf)
    plt.close(fig)
    print(f"✓ Length ratio boxplot saved: {out_path}")


def plot_length_ratio_boxplot_per_model(
        df: pd.DataFrame,
        out_dir: str,
        *,
        dataset_name: str = "",
        models_filter: Optional[List[str]] = None,
        include_title: bool = False,
        save_pdf: bool = False,
) -> None:
    """
    One boxplot per model, so you can compare how well the rewrite LLM
    preserved the target length across different tested models' prompts.

    Also generates a combined plot with all models side-by-side.
    """
    apply_neurips_style()

    d = df.copy()
    if models_filter and "model" in d.columns:
        d = d[d["model"].isin(models_filter)]

    if "model" not in d.columns:
        # No model column — just make the single combined plot
        plot_length_ratio_boxplot(
            d, os.path.join(out_dir, "length_ratio_boxplot.png"),
            dataset_name=dataset_name, include_title=include_title,
            save_pdf=save_pdf,
        )
        return

    models = sorted(d["model"].unique().tolist())

    # Per-model plots
    for model_name in models:
        dm = d[d["model"] == model_name]
        out_path = os.path.join(out_dir, f"length_ratio_boxplot_{model_name}.png")
        plot_length_ratio_boxplot(
            dm, out_path,
            dataset_name=f"{dataset_name}, {model_name}" if dataset_name else model_name,
            include_title=include_title,
            save_pdf=save_pdf,
        )

    # Combined plot (all models together)
    plot_length_ratio_boxplot(
        d, os.path.join(out_dir, "length_ratio_boxplot_all.png"),
        dataset_name=dataset_name, include_title=include_title,
        save_pdf=save_pdf,
    )


# =========================================================================
# Entry point: generate all structuredness plots from a results DataFrame
# =========================================================================

def make_structuredness_plots(
        df: pd.DataFrame,
        out_dir: str,
        *,
        dataset_name: str = "",
        models_filter: Optional[List[str]] = None,
        include_title: bool = False,
        save_pdf: bool = False,
) -> None:
    """
    Generate all structuredness-specific plots from an experiment DataFrame.

    Called by experiment files after the run, or standalone via CLI.
    Currently produces length-ratio boxplots (only meaningful for
    length_variation; silently skipped for inter_vs_imper).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Length-ratio boxplots (only for numeric strengths / length variation)
    plot_length_ratio_boxplot_per_model(
        df, out_dir,
        dataset_name=dataset_name,
        models_filter=models_filter,
        include_title=include_title,
        save_pdf=save_pdf,
    )


# =========================================================================
# Standalone CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate structuredness-specific plots from experiment CSVs."
    )
    parser.add_argument("--plot_inputs", nargs="+", required=True,
                        help="Run dirs or CSV paths to load.")
    parser.add_argument("--plot_out_dir", type=str, required=True,
                        help="Output directory for plots.")
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset name for plot titles.")
    parser.add_argument("--style_name", type=str, default="structuredness",
                        help="Style name for plot titles (length_variation, inter_vs_imper, etc.)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter to these models.")
    parser.add_argument("--places", nargs="+", default=None,
                        help="Filter to these places.")
    parser.add_argument("--save_pdf", action="store_true",
                        help="Also save PDF for each figure.")
    parser.add_argument("--include_title", action="store_true",
                        help="Add titles on figures.")

    args = parser.parse_args()

    df = load_results_csvs(args.plot_inputs)
    if df.empty:
        raise SystemExit("No data loaded from plot inputs.")

    # Metric line plots
    make_all_structuredness_metric_plots(
        plot_inputs=args.plot_inputs,
        out_dir=args.plot_out_dir,
        dataset_name=args.dataset,
        style_name=args.style_name,
        places_filter=args.places,
        models_filter=args.models,
        save_pdf=args.save_pdf,
        include_title=args.include_title,
    )

    # Extra structuredness plots (length-ratio boxplots, etc.)
    make_structuredness_plots(
        df, args.plot_out_dir,
        dataset_name=args.dataset,
        models_filter=args.models,
        save_pdf=args.save_pdf,
    )


if __name__ == "__main__":
    main()

