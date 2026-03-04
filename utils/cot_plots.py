# utils/cot_plots.py
"""
NeurIPS-style plotting utilities for CoT reasoning trace experiments.

Generates publication-ready plots showing:
- Number of reasoning steps vs style strength
- Parse success rate vs style strength
- Average step length vs style strength

Used in two ways:
1) Imported by experiments/cot_reasoning.py to plot run outputs
2) Run standalone to combine multiple runs

Standalone example:
  python utils/cot_plots.py \
    --plot_inputs results/cot_reasoning/run_multi_gsm8k_20260101_120000 \
    --plot_out_dir results/cot_reasoning/combined_plots \
    --dataset gsm8k \
    --style_name spacing \
    --save_pdf
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from utils.plot_utils import (
    apply_neurips_style,
    _coerce_numeric,
    load_results_csvs,
    _format_axis,
    _save_figure,
)


PLOT_METRICS_DEFAULT = [
    "num_steps_original",
    "num_steps_styled",
    "steps_diff",
    "parse_success_original",
    "parse_success_styled",
    "avg_step_length_original",
    "avg_step_length_styled",
]


# --------------------------
# Aggregation helpers
# --------------------------
def build_means_from_rows(df_rows: pd.DataFrame) -> pd.DataFrame:
    """Build df_mean from per-example rows."""
    if df_rows.empty:
        return pd.DataFrame()

    needed = {"model", "place", "strength"}
    missing = [c for c in needed if c not in df_rows.columns]
    if missing:
        raise ValueError(f"Cannot build means: missing columns {missing}")

    metric_cols = [c for c in PLOT_METRICS_DEFAULT if c in df_rows.columns]
    
    # Convert boolean to numeric for averaging
    for col in df_rows.columns:
        if col.startswith("parse_success"):
            df_rows[col] = df_rows[col].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})

    df_rows = _coerce_numeric(df_rows, ["strength"] + metric_cols)

    df_mean = (
        df_rows.groupby(["model", "place", "strength"], dropna=False)[metric_cols]
        .mean()
        .reset_index()
        .sort_values(["model", "place", "strength"])
    )

    return df_mean


# --------------------------
# Plotting
# --------------------------
def plot_metric_lines_general(
        df_mean: pd.DataFrame,
        metric: str,
        strengths: Optional[List[int]],
        out_path_png: str,
        *,
        dataset_name: str,
        style_name: str = "unknown",
        places: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        include_title: bool = False,
        legend_outside: bool = True,
        save_pdf: bool = False,
) -> None:
    """NeurIPS-friendly line plot: metric vs strength."""
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
        style_name: str = "unknown",
        save_pdf: bool = False,
        include_title: bool = False,
        legend_outside: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSVs, compute means, save combined means, and make plots."""
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
    return None


def main():
    parser = argparse.ArgumentParser(description="Plot CoT reasoning trace results (NeurIPS-style).")
    parser.add_argument("--plot_inputs", nargs="+", required=True,
                        help="Run dirs or CSV paths to combine.")
    parser.add_argument("--plot_out_dir", type=str, required=True,
                        help="Output directory for combined plots + combined means CSV.")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--style_name", type=str, default="unknown",
                        help="Style name for plot titles (spacing, punctuation, letter_case, politeness)")

    parser.add_argument("--places", nargs="+", default=None,
                        help="Filter places to plot.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter models to plot.")

    parser.add_argument("--strengths", nargs="+", type=int, default=None)
    parser.add_argument("--strength_range", nargs=2, type=int, default=None, metavar=("LO", "HI"))
    parser.add_argument("--strength_step", type=int, default=1)

    parser.add_argument("--save_pdf", action="store_true")
    parser.add_argument("--include_title", action="store_true")
    parser.add_argument("--legend_inside", action="store_true")

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