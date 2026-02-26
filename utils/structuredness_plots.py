"""
utils/structuredness_plots.py

Plotting utilities for structuredness-style experiments
(length variation, declarative/interrogative).

Designed to be:
  1. Imported by experiment files (e.g. experiments/length_variation.py)
     after the run to generate plots from the results DataFrame.
  2. Run standalone on saved CSVs / run directories.

Standalone example:
  python utils/structuredness_plots.py \
    --plot_inputs results/length_variation/run_multi_truthful_qa_20260225_120000 \
    --plot_out_dir results/length_variation/combined_plots \
    --dataset truthful_qa
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# =========================================================================
# Style
# =========================================================================

def apply_neurips_style() -> None:
    """NeurIPS-friendly matplotlib defaults."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.autolayout": False,
    })


# =========================================================================
# I/O helpers
# =========================================================================

def load_results_csvs(paths_or_dirs: List[str]) -> pd.DataFrame:
    """
    Load experiment results from run directories or CSV file paths.

    Accepts:
      - run directories containing full_results_all_models.csv or *_results.csv
      - direct CSV file paths

    Returns merged per-example DataFrame.
    """
    dfs = []
    for p in paths_or_dirs:
        p = os.path.expanduser(p)

        if os.path.isfile(p) and p.endswith(".csv"):
            df = pd.read_csv(p)
            df["run_id"] = os.path.basename(os.path.dirname(p)) or "run"
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
            for fn in sorted(os.listdir(p)):
                if fn.endswith("_results.csv"):
                    df = pd.read_csv(os.path.join(p, fn))
                    df["run_id"] = run_id
                    dfs.append(df)
            continue

        raise FileNotFoundError(f"Plot input not found: {p}")

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _save_figure(fig, out_png: str, *, save_pdf: bool = False, dpi: int = 200) -> None:
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(os.path.splitext(out_png)[0] + ".pdf", bbox_inches="tight")


# =========================================================================
# Length-ratio boxplot
# =========================================================================

def plot_length_ratio_boxplot(
        df: pd.DataFrame,
        out_path: str,
        *,
        dataset_name: str = "",
        models_filter: Optional[List[str]] = None,
        save_pdf: bool = False,
) -> None:
    """
    Boxplot: for each length multiplier, show the distribution of
    actual word-count ratio (styled / original) across all prompts.

    Red × markers show the requested (ideal) multiplier for reference.

    Args:
        df: Experiment results DataFrame.
            Expected columns: strength, prompt_orig, prompt_pert
            Optional: model (used for filtering)
        out_path: Path to save the PNG.
        dataset_name: For the plot title.
        models_filter: If given, keep only these models.
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

    fig, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(
        data,
        positions=list(range(len(multipliers_sorted))),
        widths=0.45,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(
            marker="D", markerfacecolor="white",
            markeredgecolor="black", markersize=6,
        ),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.4)

    # Reference: requested multiplier (ideal target)
    ax.scatter(
        list(range(len(multipliers_sorted))),
        multipliers_sorted,
        marker="x", color="red", s=80, zorder=5,
        label="Requested multiplier",
    )

    ax.set_xticks(list(range(len(multipliers_sorted))))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Requested Length Multiplier")
    ax.set_ylabel("Actual Length Ratio (styled / original)")

    title = "Actual vs. Requested Length Ratio"
    if dataset_name:
        title += f" ({dataset_name})"
    ax.set_title(title)
    ax.legend(loc="upper left")

    plt.tight_layout()
    _save_figure(fig, out_path, save_pdf=save_pdf)
    plt.close(fig)
    print(f"✓ Length ratio boxplot saved: {out_path}")


def plot_length_ratio_boxplot_per_model(
        df: pd.DataFrame,
        out_dir: str,
        *,
        dataset_name: str = "",
        models_filter: Optional[List[str]] = None,
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
            dataset_name=dataset_name, save_pdf=save_pdf,
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
            save_pdf=save_pdf,
        )

    # Combined plot (all models together)
    plot_length_ratio_boxplot(
        d, os.path.join(out_dir, "length_ratio_boxplot_all.png"),
        dataset_name=dataset_name, save_pdf=save_pdf,
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
        save_pdf: bool = False,
) -> None:
    """
    Generate all structuredness-specific plots from an experiment DataFrame.

    Called by experiment files after the run, or standalone via CLI.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Length-ratio boxplots
    plot_length_ratio_boxplot_per_model(
        df, out_dir,
        dataset_name=dataset_name,
        models_filter=models_filter,
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
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter to these models.")
    parser.add_argument("--save_pdf", action="store_true",
                        help="Also save PDF for each figure.")

    args = parser.parse_args()

    df = load_results_csvs(args.plot_inputs)
    if df.empty:
        raise SystemExit("No data loaded from plot inputs.")

    make_structuredness_plots(
        df, args.plot_out_dir,
        dataset_name=args.dataset,
        models_filter=args.models,
        save_pdf=args.save_pdf,
    )


if __name__ == "__main__":
    main()

