# utils/politeness_plots.py
"""
NeurIPS-style plotting utilities for politeness experiment.

Goals (NeurIPS 2025 template-friendly figures):
- Clean, legible, publication-ready.
- No tiny text; consistent font sizes.
- No clutter; legend outside or compact.
- White background, thin grid (optional), clear axes labels.
- Suitable DPI and tight bounding boxes for LaTeX \includegraphics[width=...].

Used in two ways:
1) Imported by experiments/politeness.py to plot the run outputs (by loading saved CSVs).
2) Run standalone to combine multiple runs / multiple CSVs into one set of plots.

Standalone example:
  python experiments/politeness_plots.py \
    --plot_inputs results/politeness/run_multi_truthful_qa_20260101_120000 \
                 results/politeness/run_multi_truthful_qa_20260101_121000 \
    --plot_out_dir results/politeness/combined_plots \
    --dataset truthful_qa \
    --places prefix suffix global \
    --strength_range -10 10 --strength_step 2 \
    --save_pdf
"""
import os
from typing import Dict, List, Optional, Sequence, Tuple

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
    "bertscore_prompt",
    "bleu",
    "bertscore_response",
    "delta_log_prob",
    "entropy_shift",
    "jsd_drift",
    "activation_similarity",
    "mirroring_rate",
]


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
def plot_metric_lines_general(
        df_mean: pd.DataFrame,
        metric: str,
        strengths: Optional[List[int]],
        out_path_png: str,
        *,
        dataset_name: str,
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

    title = f"{metric} vs strength ({dataset_name})" if include_title else None
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
            places=places_filter,
            models=models_filter,
            include_title=include_title,
            legend_outside=legend_outside,
            save_pdf=save_pdf,
        )
        print(f"✓ Plot saved: {out_png}" + (" (+pdf)" if save_pdf else ""))

    return df_rows, df_mean







def _safe_name(x: Optional[str]) -> str:
    if x is None:
        return "none"
    x = str(x)
    return "".join([c if c.isalnum() or c in ("-", "_", ".", "=") else "_" for c in x])


def _coords_cache_path(
        *,
        cache_dir: str,
        dataset_name: str,
        style_name: str,
        model: str,
        place: str,
        strength: int,
        method: str,
) -> str:
    return os.path.join(
        os.path.expanduser(cache_dir),
        "coords2d",
        _safe_name(dataset_name),
        _safe_name(style_name),
        _safe_name(model),
        _safe_name(place),
        f"strength_{int(strength)}",
        f"{_safe_name(method)}.csv",
    )


def save_coords2d(
        *,
        cache_dir: str,
        dataset_name: str,
        style_name: str,
        model: str,
        place: str,
        strength: int,
        method: str,
        prompt_ids: List[int],
        coords: np.ndarray,
        meta: Optional[Dict[str, str]] = None,
) -> str:
    """
    Save coords to CSV with schema:
      prompt_id, x, y, model, place, strength, method, (optional meta cols)
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be shape (N,2). got {coords.shape}")
    if len(prompt_ids) != coords.shape[0]:
        raise ValueError(f"prompt_ids length {len(prompt_ids)} != coords rows {coords.shape[0]}")

    path = _coords_cache_path(
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        style_name=style_name,
        model=model,
        place=place,
        strength=int(strength),
        method=method,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.DataFrame({
        "prompt_id": prompt_ids,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "model": str(model),
        "place": str(place),
        "strength": int(strength),
        "method": str(method),
    })
    if meta:
        for k, v in meta.items():
            df[str(k)] = str(v)

    df.to_csv(path, index=False)
    return path


def load_coords2d(
        *,
        cache_dir: str,
        dataset_name: str,
        style_name: str,
        model: str,
        place: str,
        strength: int,
        method: str,
) -> Optional[pd.DataFrame]:
    path = _coords_cache_path(
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        style_name=style_name,
        model=model,
        place=place,
        strength=int(strength),
        method=method,
    )
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# --------------------------
# 2D scatter: two clusters
# --------------------------
def plot_2d_scatter_two_clusters(
        *,
        coords: np.ndarray,
        labels: np.ndarray,
        out_path_png: str,
        title: Optional[str],
        xlabel: str,
        ylabel: str,
        legend_labels: Tuple[str, str] = ("harmless", "harmful"),
        jitter: float = 0.0,   # <-- add this
) -> None:
    """
    Square 2D scatter with:
      - equal aspect ratio
      - no axes
      - black border
      - clean legend
    """
    apply_neurips_style()

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    coords = np.asarray(coords, dtype=np.float32)
    labels = np.asarray(labels)

    if coords.ndim != 2 or coords.shape[1] != 2:
        return
    if labels.shape[0] != coords.shape[0]:
        return

    # Optional tiny jitter for visibility when points overlap
    if jitter and jitter > 0:
        rng = np.random.default_rng(0)
        coords = coords + jitter * rng.standard_normal(coords.shape).astype(np.float32)

    os.makedirs(os.path.dirname(out_path_png), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))

    m0 = labels == 0
    m1 = labels == 1

    ax.scatter(coords[m0, 0], coords[m0, 1], s=8, alpha=0.35, color="royalblue", edgecolors="none",
               label=legend_labels[0], rasterized=True)
    ax.scatter(coords[m1, 0], coords[m1, 1], s=8, alpha=0.35, color="crimson", edgecolors="none",
               label=legend_labels[1], rasterized=True)

    ax.set_aspect("equal", adjustable="datalim")

    # Keep your square border logic
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_frame_on(False)

    border = Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                       edgecolor="black", linewidth=1.6, clip_on=False)
    ax.add_patch(border)

    ax.legend(frameon=False, loc="upper right", fontsize=9)

    if title:
        ax.set_title(title, pad=12)

    fig.tight_layout()
    fig.savefig(out_path_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def plot_metric_vs_strength(
        *,
        df_summary: pd.DataFrame,
        metric_col: str,
        out_path_png: str,
        group_by: List[str],
        title: Optional[str],
        xlabel: str = "Strength",
        ylabel: Optional[str] = None,
        legend_outside: bool = True,
        save_pdf: bool = False,
) -> None:
    """
    Generic NeurIPS-style line plot:
      x = strength
      y = df_summary[metric_col]
      lines = group_by categories (e.g., ["place"] or ["model","place"])
    """
    apply_neurips_style()

    if df_summary is None or df_summary.empty:
        return
    if "strength" not in df_summary.columns or metric_col not in df_summary.columns:
        return

    df = df_summary.copy()
    df["strength"] = pd.to_numeric(df["strength"], errors="coerce")
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=["strength"])

    strengths_sorted = sorted(df["strength"].unique().tolist())
    if len(strengths_sorted) == 0:
        return

    fig_w, fig_h = 6.8, 3.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # one line per group
    if not group_by:
        group_by = []

    if group_by:
        grouped = df.groupby(group_by, dropna=False)
        for keys, sub in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            label = "/".join(str(k) for k in keys)
            sub2 = sub.set_index("strength").reindex(strengths_sorted)
            ax.plot(
                strengths_sorted,
                sub2[metric_col].values,
                marker="o",
                label=label
            )
    else:
        sub2 = df.set_index("strength").reindex(strengths_sorted)
        ax.plot(strengths_sorted, sub2[metric_col].values, marker="o")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else metric_col)
    if title:
        ax.set_title(title)

    ax.set_axisbelow(True)
    try:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    except Exception:
        pass

    if legend_outside and group_by:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=False,
            ncol=1,
        )
    elif group_by:
        ax.legend(frameon=False)

    fig.tight_layout()
    _save_figure(fig, out_png=out_path_png, save_pdf=save_pdf)
    plt.close(fig)


def plot_silhouette_vs_strength(
        *,
        df_summary: pd.DataFrame,
        out_path_png: str,
        group_by: List[str],
        title: Optional[str],
        legend_outside: bool = True,
        save_pdf: bool = False,
) -> None:
    # Keep your existing API, but implement via the generic function:
    plot_metric_vs_strength(
        df_summary=df_summary,
        metric_col="silhouette_cosine",
        out_path_png=out_path_png,
        group_by=group_by,
        title=title,
        ylabel="Silhouette (cosine)",
        legend_outside=legend_outside,
        save_pdf=save_pdf,
    )

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
    parser = argparse.ArgumentParser(description="Plot politeness experiment results by loading saved CSVs (NeurIPS-style).")
    parser.add_argument("--plot_inputs", nargs="+", required=True,
                        help="Run dirs or CSV paths to combine.")
    parser.add_argument("--plot_out_dir", type=str, required=True,
                        help="Output directory for combined plots + combined means CSV.")
    parser.add_argument("--dataset", type=str, default="unknown_dataset")

    parser.add_argument("--places", nargs="+", default=None,
                        help="Filter places to plot (e.g. prefix suffix global).")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter models to plot (e.g. L3-8B).")

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
        save_pdf=bool(args.save_pdf),
        include_title=bool(args.include_title),
        legend_outside=not bool(args.legend_inside),
    )


if __name__ == "__main__":
    main()