# utils/plots.py
"""
Unified plotting utilities for all styled-prompt experiments.

Inputs:
    - Per-example CSV files or run directories produced by experiment scripts.
    - Metric columns: activation_similarity, bleu, bertscore_response/prompt,
      delta_log_prob, entropy_shift, mirroring_rate, silhouette, asr.

Outputs:
    - Line plots (metric vs. style strength, grouped by model or placement).
    - Radar plots (multi-metric comparison across models or placements).
    - Ridge plots (per-layer activation distribution across strength levels).
    - PCA / t-SNE / UMAP scatter plots for 2-cluster activation visualisation.
    - BERTScore prompt-preservation line plots across datasets.

Assumptions:
    - CSVs follow the standardised schema produced by the experiment scripts
      (columns: model, place, strength, and one or more metric columns).
    - Matplotlib is available; umap-learn is optional (UMAP plots will be skipped if absent).
    - apply_neurips_style() should be called before plotting to set consistent aesthetics.
"""

import os
import re
import argparse
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D


# ============================================================
# Section 1: Shared NeurIPS-style helpers (formerly plot_utils.py)
# ============================================================

def apply_neurips_style() -> None:
    """
    NeurIPS-paper style optimised for \includegraphics[width=0.24\textwidth]{...}.
    All settings applied globally via rcParams so every subsequent plot inherits them.
    Axes fill most of the figure via subplots.left/right/top/bottom to avoid squishing.
    """
    plt.rcParams.update({
        # ── Figure & background ───────────────────────────────────────────
        "figure.figsize":       (6, 4),
        "figure.facecolor":     "white",
        "figure.autolayout":    False,
        "axes.facecolor":       "white",
        "savefig.facecolor":    "white",

        # ── Axes padding — right margin leaves room for outside legend ───────
        "figure.subplot.left":   0.15,
        "figure.subplot.right":  0.78,
        "figure.subplot.top":    0.88,
        "figure.subplot.bottom": 0.15,

        # ── Font — serif to match NeurIPS LaTeX ───────────────────────────
        "font.family":          "serif",
        "font.weight":          "bold",
        "font.size":            18,

        "axes.titlesize":       13,
        "axes.titleweight":     "bold",
        "axes.labelsize":       18,
        "axes.labelweight":     "bold",
        "xtick.labelsize":      16,
        "ytick.labelsize":      16,
        "legend.fontsize":      16,
        "legend.frameon":       True,
        "legend.framealpha":    0.9,

        # ── Lines & markers ───────────────────────────────────────────────
        "lines.linewidth":      2.5,
        "lines.markersize":     8,

        # ── Axes & grid ───────────────────────────────────────────────────
        "axes.linewidth":       1.0,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            True,
        "grid.alpha":           0.25,
        "grid.linewidth":       0.7,

        # ── Ticks ─────────────────────────────────────────────────────────
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.major.width":    1.0,
        "ytick.major.width":    1.0,

        # ── Save defaults ─────────────────────────────────────────────────
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.1,
    })


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Coerce the named columns to numeric, replacing non-parseable values with NaN."""
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

            cand = []
            for fn in sorted(os.listdir(p)):
                if fn.endswith("_results.csv"):
                    cand.append(os.path.join(p, fn))
            for fp in cand:
                df = pd.read_csv(fp)
                df["run_id"] = run_id
                dfs.append(df)
            continue

        raise FileNotFoundError(f"plot input not found: {p}")

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _format_axis(ax, xlabel: str, ylabel: str, title: str = None) -> None:
    """Apply standard axis labels and optional title; enable grid-below."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_axisbelow(True)


def _save_figure(fig, out_png: str, *, save_pdf: bool, pdf_only: bool = False, dpi: int = 300) -> None:
    """Save a matplotlib Figure to PNG and optionally to PDF."""
    out_pdf = os.path.splitext(out_png)[0] + ".pdf"
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    if not pdf_only:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if save_pdf or pdf_only:
        fig.savefig(out_pdf, bbox_inches="tight")


def _save_plt_figure(out_png: str, *, save_pdf: bool, pdf_only: bool = False, dpi: int = 300) -> None:
    """Save the current plt figure to PNG and optionally to PDF."""
    out_pdf = os.path.splitext(out_png)[0] + ".pdf"
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    if not pdf_only:
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if save_pdf or pdf_only:
        plt.savefig(out_pdf, bbox_inches="tight")


# ============================================================
# Section 2: Politeness experiment plots (formerly politeness_plots.py)
# ============================================================

PLOT_METRICS_DEFAULT_POLITENESS = [
    "bertscore_prompt",
    "bleu",
    "bertscore_response",
    "delta_log_prob",
    "entropy_shift",
    "activation_similarity",
    "mirroring_rate",
]


def build_means_from_rows_politeness(df_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Build df_mean from per-example rows (politeness variant).
    Also recompute mirroring_rate from verdicts if available.
    Always coerces strength to numeric.
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



def plot_metric_lines_general_politeness(
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
        pdf_only: bool = False,
) -> None:
    """
    NeurIPS-friendly line plot: metric vs strength (politeness variant).
    Title uses only dataset_name (no style_name).
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

    fig_w, fig_h = 6.8, 2.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

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

    if legend_outside:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            frameon=True,
            framealpha=0.9,
            ncol=1,
        )
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True, framealpha=0.9, fontsize=16)

    fig.tight_layout()
    _save_figure(fig, out_png=out_path_png, save_pdf=save_pdf, pdf_only=pdf_only)
    plt.close(fig)


def make_all_plots_from_csvs_politeness(
        *,
        plot_inputs: List[str],
        out_dir: str,
        strengths: Optional[List[int]],
        places_filter: Optional[List[str]],
        models_filter: Optional[List[str]],
        dataset_name: str,
        save_pdf: bool = False,
        pdf_only: bool = False,
        include_title: bool = False,
        legend_outside: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CSVs, compute means, save combined means, and make NeurIPS-friendly plots
    (politeness variant).
    Returns (df_rows, df_mean).
    """
    apply_neurips_style()
    os.makedirs(out_dir, exist_ok=True)

    df_rows = load_results_csvs(plot_inputs)
    if df_rows.empty:
        raise SystemExit("Plotting: no rows loaded from plot inputs.")

    df_mean = build_means_from_rows_politeness(df_rows)

    mean_path = os.path.join(out_dir, "combined_means_by_model_place_strength.csv")
    df_mean.to_csv(mean_path, index=False)
    print(f"✓ Saved combined means table: {mean_path}")

    for metric in PLOT_METRICS_DEFAULT_POLITENESS:
        if metric not in df_mean.columns:
            continue
        out_png = os.path.join(out_dir, f"{metric}_vs_strength.png")
        plot_metric_lines_general_politeness(
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
            pdf_only=pdf_only,
        )
        print(f"✓ Plot saved: {out_png}" + (" (+pdf)" if save_pdf else ""))

    return df_rows, df_mean


def _safe_name(x: Optional[str]) -> str:
    """Sanitise a value for use in file/directory names."""
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
    """Return the cache file path for 2D projection coordinates."""
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


def plot_2d_scatter_two_clusters(
        *,
        coords: np.ndarray,
        labels: np.ndarray,
        out_path_png: str,
        title: Optional[str],
        xlabel: str,
        ylabel: str,
        legend_labels: Tuple[str, str] = ("harmless", "harmful"),
        jitter: float = 0.0,
) -> None:
    """
    Square 2D scatter with:
      - equal aspect ratio
      - no axes
      - black border
      - clean legend
    """
    apply_neurips_style()

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

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_frame_on(False)

    border = Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                       edgecolor="black", linewidth=1.6, clip_on=False)
    ax.add_patch(border)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True, framealpha=0.9, fontsize=24)

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
        pdf_only: bool = False,
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
    _x_min_r = int(round(ax.get_xlim()[0]))
    _x_max_r = int(round(ax.get_xlim()[1]))
    _step = max(1, int(round((_x_max_r - _x_min_r) / 5)))
    _start = (_x_min_r // _step) * _step
    _ticks = np.arange(_start, _x_max_r + _step, _step, dtype=int)
    ax.set_xticks(_ticks)
    ax.set_xticklabels([str(x) for x in _ticks], fontsize=24)

    if legend_outside and group_by:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            frameon=True,
            framealpha=0.9,
            ncol=1,
        )
    elif group_by:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True, framealpha=0.9, fontsize=16)

    fig.tight_layout()
    _save_figure(fig, out_png=out_path_png, save_pdf=save_pdf, pdf_only=pdf_only)
    plt.close(fig)


def plot_silhouette_vs_strength(
        *,
        df_summary: pd.DataFrame,
        out_path_png: str,
        group_by: List[str],
        title: Optional[str],
        legend_outside: bool = True,
        save_pdf: bool = False,
        pdf_only: bool = False,
) -> None:
    plot_metric_vs_strength(
        df_summary=df_summary,
        metric_col="silhouette_cosine",
        out_path_png=out_path_png,
        group_by=group_by,
        title=title,
        ylabel="Silhouette (cosine)",
        legend_outside=legend_outside,
        save_pdf=save_pdf,
        pdf_only=pdf_only,
    )


def _select_strengths_for_plot(
        strengths: Optional[List[int]],
        strength_range: Optional[Tuple[int, int]],
        strength_step: int,
) -> Optional[List[int]]:
    """Resolve the explicit list of strengths to plot, or return None to auto-detect from CSV."""
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


# ============================================================
# Section 3: Surface noise experiment plots (formerly surface_plots.py)
# ============================================================

PLOT_METRICS_DEFAULT_SURFACE = [
    "bertscore_prompt",
    "bleu",
    "bertscore_response",
    "delta_log_prob",
    "entropy_shift",
    "activation_similarity",
    "mirroring_rate"
]


def build_means_from_rows_surface(df_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Build df_mean from per-example rows (surface noise variant).
    Also recompute mirroring_rate from verdicts if available.
    Always coerces strength to numeric.
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


def plot_metric_lines_general_surface(
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
        pdf_only: bool = False,
) -> None:
    """
    NeurIPS-friendly line plot: metric vs strength (surface noise variant).
    Title includes both style_name and dataset_name.
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

    fig_w, fig_h = 6.8, 2.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

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
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            frameon=True,
            framealpha=0.9,
            ncol=1,
        )
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True, framealpha=0.9, fontsize=16)

    fig.tight_layout()
    _save_figure(fig, out_png=out_path_png, save_pdf=save_pdf, pdf_only=pdf_only)
    plt.close(fig)


def make_all_plots_from_csvs_surface(
        *,
        plot_inputs: List[str],
        out_dir: str,
        strengths: Optional[List[int]],
        places_filter: Optional[List[str]],
        models_filter: Optional[List[str]],
        dataset_name: str,
        style_name: str = "surface_noise",
        save_pdf: bool = False,
        pdf_only: bool = False,
        include_title: bool = False,
        legend_outside: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CSVs, compute means, save combined means, and make NeurIPS-friendly plots
    (surface noise variant).
    Returns (df_rows, df_mean).
    """
    apply_neurips_style()
    os.makedirs(out_dir, exist_ok=True)

    df_rows = load_results_csvs(plot_inputs)
    if df_rows.empty:
        raise SystemExit("Plotting: no rows loaded from plot inputs.")

    df_mean = build_means_from_rows_surface(df_rows)

    mean_path = os.path.join(out_dir, "combined_means_by_model_place_strength.csv")
    df_mean.to_csv(mean_path, index=False)
    print(f"✓ Saved combined means table: {mean_path}")

    for metric in PLOT_METRICS_DEFAULT_SURFACE:
        if metric not in df_mean.columns:
            continue
        out_png = os.path.join(out_dir, f"{metric}_vs_strength.png")
        plot_metric_lines_general_surface(
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
            pdf_only=pdf_only,
        )
        print(f"✓ Plot saved: {out_png}" + (" (+pdf)" if save_pdf else ""))

    return df_rows, df_mean


# Backward-compatible alias: surface_plots imported make_all_plots_from_csvs
make_all_plots_from_csvs = make_all_plots_from_csvs_surface


# ============================================================
# Section 4: Structuredness experiment plots (formerly structuredness_plots.py)
# ============================================================

PLOT_METRICS_DEFAULT_STRUCTUREDNESS = [
    "bertscore_prompt",
    "bleu",
    "bertscore_response",
    "delta_log_prob",
    "entropy_shift",
    "activation_similarity",
    "mirroring_rate",
]


def _is_numeric(val) -> bool:
    """Return True if val can be interpreted as a number."""
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def build_means_from_rows_structuredness(df_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Build df_mean from per-example rows (structuredness variant).
    Also recompute mirroring_rate from verdicts if available.

    Works for both numeric strengths (length variation multipliers)
    and categorical strengths (inter_vs_imper modes).
    Only coerces strength to numeric if all values are numeric.
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
        pdf_only: bool = False,
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

    if metric == "bertscore_prompt":
        for place in sorted(df["place"].unique().tolist()):
            sub = df[df["place"] == place].copy()
            if sub.empty:
                continue
            sub = sub.drop_duplicates(subset=["strength", "place"])
            sub = sub.set_index("strength").reindex(strengths_sorted)
            y = sub[metric].values
            ax.plot(x_positions, y, marker="o", label=f"{place}")
    else:
        for model in sorted(df["model"].unique().tolist()):
            for place in sorted(df["place"].unique().tolist()):
                sub = df[(df["model"] == model) & (df["place"] == place)].copy()
                if sub.empty:
                    continue
                sub = sub.set_index("strength").reindex(strengths_sorted)
                y = sub[metric].values
                ax.plot(x_positions, y, marker="o", label=f"{model}/{place}")

    if metric == "bertscore_prompt":
        ax.axhline(
            y=0.85,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label="Threshold (0.85)",
        )

    if is_categorical:
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(s) for s in strengths_sorted])

    xlabel = "Mode" if is_categorical else "Multiplier"
    ylabel_final = ylabel if ylabel is not None else metric
    title = f"{metric} vs {xlabel.lower()} ({style_name}, {dataset_name})" if include_title else None

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel_final)
    if title:
        ax.set_title(title)
    ax.set_axisbelow(True)

    if not is_categorical:
        _x_min_r = int(round(ax.get_xlim()[0]))
        _x_max_r = int(round(ax.get_xlim()[1]))
        _step = max(1, int(round((_x_max_r - _x_min_r) / 5)))
        _start = (_x_min_r // _step) * _step
        _ticks = np.arange(_start, _x_max_r + _step, _step, dtype=int)
        ax.set_xticks(_ticks)
        ax.set_xticklabels([str(x) for x in _ticks], fontsize=24)

    # Legend
    if legend_outside:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            frameon=True,
            framealpha=0.9,
            ncol=1,
        )
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True, framealpha=0.9, fontsize=16)

    fig.tight_layout()
    _save_figure(fig, out_png=out_path_png, save_pdf=save_pdf, pdf_only=pdf_only)
    plt.close(fig)


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
        pdf_only: bool = False,
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

    df_mean = build_means_from_rows_structuredness(df_rows)

    mean_path = os.path.join(out_dir, "combined_means_by_model_place_strength.csv")
    df_mean.to_csv(mean_path, index=False)
    print(f"✓ Saved combined means table: {mean_path}")

    for metric in PLOT_METRICS_DEFAULT_STRUCTUREDNESS:
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
            pdf_only=pdf_only,
        )
        print(f"✓ Plot saved: {out_png}" + (" (+pdf)" if save_pdf else ""))

    return df_rows, df_mean


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
    Black-outlined white diamonds show the mean actual ratio.
    """
    apply_neurips_style()

    d = df.copy()

    if models_filter and "model" in d.columns:
        d = d[d["model"].isin(models_filter)]

    for col in ("strength", "prompt_orig", "prompt_pert"):
        if col not in d.columns:
            print(f"⚠️  plot_length_ratio_boxplot: missing column '{col}', skipping.")
            return

    d["strength"] = pd.to_numeric(d["strength"], errors="coerce")
    d = d.dropna(subset=["strength", "prompt_orig", "prompt_pert"])
    if d.empty:
        return

    # De-duplicate first: styled prompts are identical across models.
    dedup_cols = ["prompt_orig", "prompt_pert", "strength"]
    d = d.drop_duplicates(subset=dedup_cols)

    d["orig_words"] = d["prompt_orig"].apply(lambda s: len(str(s).split()))
    d["styled_words"] = d["prompt_pert"].apply(lambda s: len(str(s).split()))
    d["length_ratio"] = d["styled_words"] / d["orig_words"].clip(lower=1)

    multipliers_sorted = sorted(d["strength"].unique())
    if not multipliers_sorted:
        return

    data = [d[d["strength"] == m]["length_ratio"].values for m in multipliers_sorted]
    labels = [str(m) for m in multipliers_sorted]

    fig, ax = plt.subplots(figsize=(6.8, 5))
    lw = 0.8  # match axes.linewidth from NeurIPS style

    bp = ax.boxplot(
        data,
        positions=list(range(len(multipliers_sorted))),
        widths=0.4,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(
            marker="D",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=4,
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

    ax.scatter(
        list(range(len(multipliers_sorted))),
        multipliers_sorted,
        marker="x",
        color="#d62728",
        s=40,
        linewidths=1.0,
        zorder=5,
    )

    ax.set_xticks(list(range(len(multipliers_sorted))))
    ax.set_xticklabels(labels)

    title_str = None
    if include_title:
        title_str = "Actual vs. Requested Length Ratio"
        if dataset_name:
            title_str += f" ({dataset_name})"

    ax.set_xlabel("Requested Multiplier")
    ax.set_ylabel("Actual Ratio")
    if title_str:
        ax.set_title(title_str)

    ax.set_axisbelow(True)

    # Manual legend handles so each item appears only once.
    mean_handle = Line2D(
        [0], [0],
        marker="D",
        markerfacecolor="white",
        markeredgecolor="black",
        linestyle="None",
        markersize=4,
        label="Mean",
    )

    requested_handle = Line2D(
        [0], [0],
        marker="x",
        color="#d62728",
        linestyle="None",
        markersize=6,
        label="Requested multiplier",
    )

    ax.legend(
        handles=[mean_handle, requested_handle],
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=True,
        framealpha=0.9,
    )

    fig.tight_layout()
    _save_figure(fig, out_path, save_pdf=save_pdf, pdf_only=pdf_only)
    plt.close(fig)
    print(f"[PLOT] Length ratio")


def make_structuredness_plots(
        df: pd.DataFrame,
        out_dir: str,
        *,
        dataset_name: str = "",
        models_filter: Optional[List[str]] = None,
        include_title: bool = False,
        save_pdf: bool = False,
        pdf_only: bool = False,
) -> None:
    """
    Generate all structuredness-specific plots from an experiment DataFrame.

    Called by experiment files after the run, or standalone via CLI.
    Currently produces length-ratio boxplots (only meaningful for
    length_variation; silently skipped for inter_vs_imper).
    """
    apply_neurips_style()
    os.makedirs(out_dir, exist_ok=True)

    plot_length_ratio_boxplot(
        df, os.path.join(out_dir, "length_ratio_boxplot.png"),
        dataset_name=dataset_name,
        models_filter=models_filter,
        include_title=include_title,
        save_pdf=save_pdf,
        pdf_only=pdf_only,
    )


# ============================================================
# Section 5: CoT reasoning trace plots (formerly cot_plots.py)
# ============================================================

PLOT_METRICS_DEFAULT_COT = [
    "num_steps_original",
    "num_steps_styled",
    "steps_diff",
    "parse_success_original",
    "parse_success_styled",
    "avg_step_length_original",
    "avg_step_length_styled",
]


def build_means_from_rows_cot(df_rows: pd.DataFrame) -> pd.DataFrame:
    """Build df_mean from per-example rows (CoT variant)."""
    if df_rows.empty:
        return pd.DataFrame()

    needed = {"model", "place", "strength"}
    missing = [c for c in needed if c not in df_rows.columns]
    if missing:
        raise ValueError(f"Cannot build means: missing columns {missing}")

    metric_cols = [c for c in PLOT_METRICS_DEFAULT_COT if c in df_rows.columns]

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


def plot_metric_lines_general_cot(
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
        pdf_only: bool = False,
) -> None:
    """NeurIPS-friendly line plot: metric vs strength (CoT variant)."""
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
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            frameon=True,
            framealpha=0.9,
            ncol=1,
        )
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True, framealpha=0.9, fontsize=16)

    fig.tight_layout()
    _save_figure(fig, out_png=out_path_png, save_pdf=save_pdf, pdf_only=pdf_only)
    plt.close(fig)


def make_all_plots_from_csvs_cot(
        *,
        plot_inputs: List[str],
        out_dir: str,
        strengths: Optional[List[int]],
        places_filter: Optional[List[str]],
        models_filter: Optional[List[str]],
        dataset_name: str,
        style_name: str = "unknown",
        save_pdf: bool = False,
        pdf_only: bool = False,
        include_title: bool = False,
        legend_outside: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSVs, compute means, save combined means, and make plots (CoT variant)."""
    apply_neurips_style()
    os.makedirs(out_dir, exist_ok=True)

    df_rows = load_results_csvs(plot_inputs)
    if df_rows.empty:
        raise SystemExit("Plotting: no rows loaded from plot inputs.")

    df_mean = build_means_from_rows_cot(df_rows)

    mean_path = os.path.join(out_dir, "combined_means_by_model_place_strength.csv")
    df_mean.to_csv(mean_path, index=False)
    print(f"✓ Saved combined means table: {mean_path}")

    for metric in PLOT_METRICS_DEFAULT_COT:
        if metric not in df_mean.columns:
            continue
        out_png = os.path.join(out_dir, f"{metric}_vs_strength.png")
        plot_metric_lines_general_cot(
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
            pdf_only=pdf_only,
        )
        print(f"✓ Plot saved: {out_png}" + (" (+pdf)" if save_pdf else ""))

    return df_rows, df_mean


# ============================================================
# Section 6: Aggregate plots (formerly aggregate_plots.py)
# ============================================================

# Metrics we want for all models and places
ALLOWED_METRICS = [
    "bleu",
    "bertscore_response",
    "delta_log_prob",
    "entropy_shift",
    "activation_similarity",
    "mirroring_rate",
    "asr",
    "silhouette",
]

# Optional special-case metric (not for all models)
SPECIAL_SINGLE_MODEL_METRIC = "bertscore_prompt"

METRIC_DISPLAY = {
    "bleu": "BLEU",
    "bertscore_response": "BERTScore",
    "delta_log_prob": "Δ Log Prob",
    "entropy_shift": "Entropy Shift",
    "activation_similarity": "Activation Similarity",
    "mirroring_rate": "Mirroring Rate",
    "bertscore_prompt": "BERTScore (Prompt)",
    "asr": "ASR",
    "silhouette": "Silhouette Score",
}


def metric_display_name(metric: str) -> str:
    """Return a human-readable display name for a metric column."""
    return METRIC_DISPLAY.get(metric, metric)


# Strength value to exclude from radar averages, keyed by style name.
# For these styles the "identity" / baseline is not strength=0 or 1 but a
# specific numeric value that should not bias the aggregate.
STYLE_BASELINE_STRENGTH: Dict[str, float] = {
    "politeness":     0.0,
    "spacing":        0.0,
    "letter_case":    0.0,
    "punctuation":    0.0,
    "length_variation": 1.0,
}


STYLE_DISPLAY_NAMES: Dict[str, str] = {
    "politeness": "Social-Tone",
}


def _style_display_name(style_name: Optional[str]) -> str:
    """Return the display name for a style, with overrides for renamed styles."""
    if not style_name:
        return "Style"
    return STYLE_DISPLAY_NAMES.get(style_name, style_name.replace("_", " ").title())


def _style_suffix(style_name: Optional[str]) -> str:
    """Return ' — Style Name' for use in plot titles, or '' if not given."""
    if not style_name:
        return ""
    return " — " + _style_display_name(style_name)


_DATASET_CANON = {
    "truthfulqa": "TruthfulQA",
    "truthful_qa": "TruthfulQA",
    "natural_questions": "Natural Questions",
    "naturalquestions": "Natural Questions",
    "nq": "Natural Questions",
}


def _canon_dataset_name(raw: str) -> Optional[str]:
    """Normalise a raw dataset name string to a canonical display name, or None if unrecognised."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    if s in _DATASET_CANON:
        return _DATASET_CANON[s]
    for k, v in _DATASET_CANON.items():
        if k in s:
            return v
    return None


def dataset_title_suffix(
        d: pd.DataFrame,
        dataset_name: Optional[str] = None
) -> str:
    """
    Returns:
      " (TruthfulQA)" or " (Natural Questions)" or
      " (TruthfulQA + Natural Questions)" or ""
    Priority:
      1) Explicit --dataset_name argument
      2) dataset / benchmark / task column
      3) run_dir inference
    """
    if dataset_name is not None:
        return f" ({dataset_name})"

    names: List[str] = []

    for col in ["dataset", "benchmark", "task"]:
        if col in d.columns:
            vals = [x for x in d[col].dropna().unique().tolist()]
            for v in vals:
                cn = _canon_dataset_name(v)
                if cn and cn not in names:
                    names.append(cn)

    if not names and "run_dir" in d.columns:
        vals = [x for x in d["run_dir"].dropna().unique().tolist()]
        for v in vals:
            cn = _canon_dataset_name(v)
            if cn and cn not in names:
                names.append(cn)

    if not names:
        return ""

    ordered = []
    for v in ["TruthfulQA", "Natural Questions"]:
        if v in names:
            ordered.append(v)
    for v in sorted(set(names)):
        if v not in ordered:
            ordered.append(v)

    if len(ordered) == 1:
        return f" ({ordered[0]})"
    return " (" + " + ".join(ordered) + ")"


def metric_dataset_suffix(metric: str, d: pd.DataFrame, dataset_name: Optional[str] = None) -> str:
    """
    Metric-specific dataset suffix rule:
      - if metric == "asr"       -> " (HarmBench)"
      - if metric == "silhouette"-> " (HarmBench + Alpaca)"
      - else                     -> dataset_title_suffix(..., dataset_name)
    """
    if metric == "asr":
        return " (HarmBench)"
    if metric == "silhouette":
        return " (HarmBench + Alpaca)"
    return dataset_title_suffix(d, dataset_name=dataset_name)


def apply_style():
    """
    Same NeurIPS-paper settings as apply_neurips_style(), used for aggregate
    (ridge / multi-style line) plots.
    """
    apply_neurips_style()


def _find_means_csv(run_dir: str) -> Optional[str]:
    """Return run_dir if it exists as a CSV path, else None."""
    cand = run_dir
    if os.path.exists(cand):
        return cand
    return None


def load_all_runs(run_dirs: List[str]) -> pd.DataFrame:
    """Load and concatenate combined_means CSVs from all provided run paths."""
    dfs = []
    for rd in run_dirs:
        rd = os.path.expanduser(rd)
        csv_path = _find_means_csv(rd)
        if csv_path is None:
            print(f"[WARN] Missing combined means CSV in: {rd}")
            continue
        df = pd.read_csv(csv_path)
        df["run_dir"] = os.path.basename(os.path.normpath(rd))
        dfs.append(df)

    if not dfs:
        raise ValueError("No combined_means_by_model_place_strength.csv found in provided --runs.")

    out = pd.concat(dfs, ignore_index=True)

    if "strength" in out.columns and _strength_is_numeric_series(out["strength"]):
        out["strength"] = pd.to_numeric(out["strength"], errors="coerce")
    else:
        out["strength"] = out["strength"].astype(str)
        out.loc[out["strength"].isin(["nan", "None"]), "strength"] = np.nan
    return out


def infer_metric_columns(df: pd.DataFrame) -> List[str]:
    """Return columns that look like numeric metrics (not key columns)."""
    base = {"model", "place", "strength", "run_dir"}
    metrics = [c for c in df.columns if c not in base]
    keep = []
    for c in metrics:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            keep.append(c)
    return keep


def select_metrics(df: pd.DataFrame) -> List[str]:
    """Return the allowed metrics that are present as numeric columns in df."""
    present = set(infer_metric_columns(df))
    chosen = [m for m in ALLOWED_METRICS if m in present]
    if SPECIAL_SINGLE_MODEL_METRIC in present:
        chosen.append(SPECIAL_SINGLE_MODEL_METRIC)
    return chosen


DARK_MODEL_BASE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

DARK_PLACE_BASE = {
    "prefix": "#1f77b4",
    "middle": "#2ca02c",
    "suffix": "#ff7f0e",
    "global": "#d62728",
}


def _mix_with_white(hex_color: str, alpha: float) -> Tuple[float, float, float]:
    """Blend a hex colour with white at weight alpha (0=original, 1=white)."""
    rgb = np.array(to_rgb(hex_color))
    return tuple((1 - alpha) * rgb + alpha * np.ones_like(rgb))


def build_model_color_map(models: List[str]) -> Dict[str, str]:
    """Return a stable colour mapping for the given model names."""
    models_sorted = sorted(models)
    return {m: DARK_MODEL_BASE[i % len(DARK_MODEL_BASE)] for i, m in enumerate(models_sorted)}


def build_place_color_map(places: List[str]) -> Dict[str, str]:
    """Return a stable colour mapping for the given placement names."""
    fallback = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    out = {}
    for i, p in enumerate(sorted(places)):
        out[p] = DARK_PLACE_BASE.get(p, fallback[i % len(fallback)])
    return out


def shade_for_place(base_color: str, place: str, places_sorted: List[str]) -> Tuple[float, float, float]:
    """Return a lighter shade of base_color for a placement, offset by its rank in the sorted list."""
    idx = places_sorted.index(place)
    alpha = 0.05 + 0.18 * idx
    return _mix_with_white(base_color, alpha)


PLACE_LINESTYLE = {
    "prefix": "-",
    "middle": "--",
    "suffix": "-.",
    "global": ":",
}


def _sanitize_filename(s: str) -> str:
    """Sanitise a string for use as a filename (replace whitespace and special chars)."""
    s = str(s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s[:200]


def _filter_df(df: pd.DataFrame, models: Optional[List[str]], places: Optional[List[str]]) -> pd.DataFrame:
    """Filter a DataFrame to only rows matching the requested models/places, if specified."""
    d = df.copy()
    if models is not None:
        d = d[d["model"].isin(models)]
    if places is not None:
        d = d[d["place"].isin(places)]
    return d


def _strength_is_numeric_series(s: pd.Series) -> bool:
    """Return True if all non-NaN values in s can be parsed as numbers."""
    vals = [x for x in s.dropna().tolist()]
    return len(vals) > 0 and all(_is_numeric(x) for x in vals)


def _sorted_strength_values(values: Sequence[Any]) -> List[Any]:
    """Sort strength values numerically if possible, otherwise lexicographically."""
    vals = [x for x in values if pd.notna(x)]
    if not vals:
        return []
    if all(_is_numeric(x) for x in vals):
        return sorted(vals, key=lambda x: float(x))
    return sorted(vals, key=lambda x: (0, str(x).lower()) if str(x).lower() == "original" else (1, str(x).lower()))


def _prepare_strength_axis(d: pd.DataFrame) -> Tuple[pd.DataFrame, List[Any], List[Any], bool]:
    """
    Prepare a DataFrame for plotting the strength axis.

    Returns (df_with_pos_col, sorted_strengths, x_tick_values, is_categorical).
    For numeric strengths, _strength_pos = strength value.
    For categorical strengths, _strength_pos = rank index.
    """
    d = d.copy()
    strengths_sorted = _sorted_strength_values(d["strength"].unique().tolist())
    is_categorical = not _strength_is_numeric_series(d["strength"])
    if is_categorical:
        pos_map = {s: i for i, s in enumerate(strengths_sorted)}
        d["_strength_pos"] = d["strength"].map(pos_map)
        x_values = list(range(len(strengths_sorted)))
    else:
        d["strength"] = pd.to_numeric(d["strength"], errors="coerce")
        strengths_sorted = _sorted_strength_values(d["strength"].dropna().unique().tolist())
        d["_strength_pos"] = d["strength"]
        x_values = [float(s) for s in strengths_sorted]
    return d, strengths_sorted, x_values, is_categorical


def aggregate_plot_metric_lines(
        df: pd.DataFrame,
        metric: str,
        out_path: str,
        models: Optional[List[str]] = None,
        places: Optional[List[str]] = None,
        save_pdf: bool = False,
        pdf_only: bool = False,
        dataset_name: Optional[str] = None,
        style_name: Optional[str] = None,
):
    """Line plot TYPE 1: all models + all places (aggregate variant)."""
    apply_style()

    d = _filter_df(df, models, places)

    if d.empty or metric not in d.columns:
        return

    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=["strength", metric])
    if d.empty:
        return

    d, strengths_sorted, x_values, is_categorical = _prepare_strength_axis(d)

    ds_suffix = metric_dataset_suffix(metric, d, dataset_name=dataset_name)

    models_u = sorted(d["model"].unique().tolist())
    places_u = sorted(d["place"].unique().tolist())

    single_model_mode = (metric == SPECIAL_SINGLE_MODEL_METRIC and len(models_u) > 0)
    if single_model_mode:
        keep_model = models_u[0]
        d = d[d["model"] == keep_model].copy()
        models_u = [keep_model]

    model_colors = build_model_color_map(models_u)

    fig, ax = plt.subplots(figsize=(12, 7))

    for model in models_u:
        base = model_colors[model]
        for place in places_u:
            sub = d[(d["model"] == model) & (d["place"] == place)].copy()
            if sub.empty:
                continue
            sub = sub.drop_duplicates(subset=["strength"]).set_index("strength").reindex(strengths_sorted).reset_index()

            series_label = f"{place}" if single_model_mode else f"{model} · {place}"

            ax.plot(
                x_values,
                sub[metric].values,
                marker="o",
                markersize=4,
                linewidth=1.8,
                color=shade_for_place(base, place, places_u),
                linestyle=PLACE_LINESTYLE.get(place, "-"),
                alpha=0.95,
                label=series_label,
            )

    ax.set_xlabel("Style Strength" if not is_categorical else "Style Mode", fontsize=28)
    ax.set_ylabel(metric_display_name(metric), fontsize=28)
    if not is_categorical:
        _x_min_r = int(round(ax.get_xlim()[0]))
        _x_max_r = int(round(ax.get_xlim()[1]))
        _step = max(1, int(round((_x_max_r - _x_min_r) / 5)))
        _start = (_x_min_r // _step) * _step
        _ticks = np.arange(_start, _x_max_r + _step, _step, dtype=int)
        ax.set_xticks(_ticks)
        ax.set_xticklabels([str(x) for x in _ticks], fontsize=24)
    else:
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(s) for s in strengths_sorted], fontsize=24)
    ax.tick_params(axis="y", labelsize=24)

    _sn = _style_display_name(style_name)
    _ds = dataset_name or ""
    _scope = f"{_ds} — All Models & Places" if _ds else "All Models & Places"
    ax.set_title(f"{metric_display_name(metric)} vs. {_sn} Strength\n{_scope}", fontsize=26)

    ymin = float(d[metric].min())
    ymax = float(d[metric].max())
    if ymin != ymax:
        pad = 0.12 * (ymax - ymin)
        ylo, yhi = ymin - pad, ymax + pad
    else:
        ylo, yhi = ymin - 0.1, ymax + 0.1

    if metric == "bertscore_prompt":
        thr = 0.85
        ylo = min(ylo, thr - 0.02)
        yhi = max(yhi, thr + 0.02)
        ax.axhline(
            y=thr,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label="Threshold (0.85)",
            zorder=3,
        )

    ax.set_ylim(ylo, yhi)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True, framealpha=0.9, fontsize=24)
    plt.subplots_adjust(left=0.10, right=0.78, top=0.90, bottom=0.12)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _save_plt_figure(out_path, save_pdf=save_pdf, pdf_only=pdf_only)
    plt.close()


def plot_metric_lines_per_model(
        df: pd.DataFrame,
        metric: str,
        out_dir: str,
        models: Optional[List[str]] = None,
        places: Optional[List[str]] = None,
        save_pdf: bool = False,
        pdf_only: bool = False,
        dataset_name: Optional[str] = None,
        style_name: Optional[str] = None,
):
    """Line plot TYPE 2: per-model (combine all places). One figure per model."""
    apply_style()

    d = _filter_df(df, models, places)
    if d.empty or metric not in d.columns:
        return

    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=["strength", metric])
    if d.empty:
        return

    d, strengths_sorted, x_values, is_categorical = _prepare_strength_axis(d)

    ds_suffix = metric_dataset_suffix(metric, d, dataset_name=dataset_name)

    models_u = sorted(d["model"].unique().tolist())
    places_u = sorted(d["place"].unique().tolist())

    single_model_mode = (metric == SPECIAL_SINGLE_MODEL_METRIC and len(models_u) > 0)
    if single_model_mode:
        models_u = [models_u[0]]
        d = d[d["model"] == models_u[0]].copy()

    model_colors = build_model_color_map(models_u)

    os.makedirs(out_dir, exist_ok=True)

    for model in models_u:
        fig, ax = plt.subplots(figsize=(12, 7))
        base = model_colors[model]

        for place in places_u:
            sub = d[(d["model"] == model) & (d["place"] == place)].copy()
            if sub.empty:
                continue
            sub = sub.drop_duplicates(subset=["strength"]).set_index("strength").reindex(strengths_sorted).reset_index()

            ax.plot(
                x_values,
                sub[metric].values,
                marker="o",
                markersize=4,
                linewidth=1.8,
                color=shade_for_place(base, place, places_u),
                linestyle=PLACE_LINESTYLE.get(place, "-"),
                alpha=0.95,
                label=str(place),
            )

        ax.set_xlabel("Style Strength" if not is_categorical else "Style Mode", fontsize=28)
        ax.set_ylabel(metric_display_name(metric), fontsize=28)
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(s) for s in strengths_sorted], fontsize=24)
        ax.tick_params(axis="y", labelsize=24)
        _sn = _style_display_name(style_name)
        _ds = dataset_name or ""
        _scope = f"{_ds} — {model}" if _ds else model
        ax.set_title(f"{metric_display_name(metric)} vs. {_sn} Strength\n{_scope}", fontsize=26)

        ymin = float(d[d["model"] == model][metric].min())
        ymax = float(d[d["model"] == model][metric].max())
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
            pad = 0.12 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

        if metric == "bertscore_prompt":
            thr = 0.85
            ax.axhline(y=thr, color="black", linestyle="--", linewidth=1.5, alpha=0.8, zorder=3)

        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True, framealpha=0.9, fontsize=24)
        plt.subplots_adjust(left=0.10, right=0.78, top=0.90, bottom=0.12)

        out_path = os.path.join(out_dir, f"{_sanitize_filename(metric)}_line_per_model__{_sanitize_filename(model)}.png")
        _save_plt_figure(out_path, save_pdf=save_pdf, pdf_only=pdf_only)
        plt.close()


def plot_metric_lines_per_place(
        df: pd.DataFrame,
        metric: str,
        out_dir: str,
        models: Optional[List[str]] = None,
        places: Optional[List[str]] = None,
        save_pdf: bool = False,
        pdf_only: bool = False,
        dataset_name: Optional[str] = None,
        style_name: Optional[str] = None,
):
    """Line plot TYPE 3: per-place (combine all models). One figure per place."""
    apply_style()

    d = _filter_df(df, models, places)
    if d.empty or metric not in d.columns:
        return

    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=["strength", metric])
    if d.empty:
        return

    d, strengths_sorted, x_values, is_categorical = _prepare_strength_axis(d)

    ds_suffix = metric_dataset_suffix(metric, d, dataset_name=dataset_name)

    models_u = sorted(d["model"].unique().tolist())
    places_u = sorted(d["place"].unique().tolist())

    single_model_mode = (metric == SPECIAL_SINGLE_MODEL_METRIC and len(models_u) > 0)
    if single_model_mode:
        keep_model = models_u[0]
        d = d[d["model"] == keep_model].copy()
        models_u = [keep_model]

    model_colors = build_model_color_map(models_u)

    os.makedirs(out_dir, exist_ok=True)

    for place in places_u:
        fig, ax = plt.subplots(figsize=(12, 7))

        for model in models_u:
            sub = d[(d["model"] == model) & (d["place"] == place)].copy()
            if sub.empty:
                continue
            sub = sub.drop_duplicates(subset=["strength"]).set_index("strength").reindex(strengths_sorted).reset_index()

            ax.plot(
                x_values,
                sub[metric].values,
                marker="o",
                markersize=4,
                linewidth=1.8,
                color=model_colors[model],
                linestyle="-",
                alpha=0.95,
                label=str(model),
            )

        ax.set_xlabel("Style Strength" if not is_categorical else "Style Mode", fontsize=28)
        ax.set_ylabel(metric_display_name(metric), fontsize=28)
        if not is_categorical:
            _x_min_r = int(round(ax.get_xlim()[0]))
            _x_max_r = int(round(ax.get_xlim()[1]))
            _step = max(1, int(round((_x_max_r - _x_min_r) / 5)))
            _start = (_x_min_r // _step) * _step
            _ticks = np.arange(_start, _x_max_r + _step, _step, dtype=int)
            ax.set_xticks(_ticks)
            ax.set_xticklabels([str(x) for x in _ticks], fontsize=24)
        else:
            ax.set_xticks(x_values)
            ax.set_xticklabels([str(s) for s in strengths_sorted], fontsize=24)
        ax.tick_params(axis="y", labelsize=24)
        _sn = _style_display_name(style_name)
        _ds = dataset_name or ""
        #upper case placement for better title formatting, e.g. "Prefix" instead of "prefix"
        _place= place.title() if isinstance(place, str) else str(place)
        _scope = f"{_ds} — {_place} Placement" if _ds else place
        ax.set_title(f"{metric_display_name(metric)} vs. {_sn} Strength\n{_scope}", fontsize=26)

        ymin = float(d[d["place"] == place][metric].min())
        ymax = float(d[d["place"] == place][metric].max())
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
            pad = 0.12 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

        if metric == "bertscore_prompt":
            thr = 0.85
            ax.axhline(y=thr, color="black", linestyle="--", linewidth=1.5, alpha=0.8, zorder=3)

        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True, framealpha=0.9, fontsize=24)
        plt.subplots_adjust(left=0.10, right=0.78, top=0.90, bottom=0.12)

        out_path = os.path.join(out_dir, f"{_sanitize_filename(metric)}_line_per_place__{_sanitize_filename(place)}.png")
        _save_plt_figure(out_path, save_pdf=save_pdf, pdf_only=pdf_only)
        plt.close()


def fix_radar_ax(ax) -> None:
    """
    Fix overlapping labels/numbers on any polar (radar/spider) axes.
    Call this after all plotting on `ax` is done.
    """
    ax.set_rlabel_position(330)                     # place numbers midway between Form and Polite. spokes
    ax.tick_params(pad=18)                           # push spoke labels outward
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])           # 4 clean gridlines
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=22, color="grey")
    ax.tick_params(axis="x", labelsize=24)           # spoke labels readable


def _radar_setup(ax, labels: List[str], title: Optional[str] = None):
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=16)

    ax.grid(True, alpha=0.45, linewidth=0.9)

    if title:
        ax.set_title(title, pad=18, fontsize=18)

    fix_radar_ax(ax)
    return angles


def _aggregate_for_radar(df: pd.DataFrame, metric: str,
                         style_name: Optional[str] = None) -> pd.DataFrame:
    d = df.copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    if style_name is not None and "strength" in d.columns:
        baseline = STYLE_BASELINE_STRENGTH.get(style_name.lower())
        if baseline is not None:
            d = d[pd.to_numeric(d["strength"], errors="coerce") != baseline]
    agg = d.groupby(["model", "place"], dropna=False)[metric].mean().reset_index()
    return agg


def _normalize_table(values: pd.DataFrame, cols: List[str], mode: str) -> pd.DataFrame:
    if mode == "none":
        return values

    out = values.copy()
    arr = out[cols].to_numpy(dtype=float)

    if mode == "minmax":
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return out
        out[cols] = (arr - mn) / (mx - mn)
        return out

    if mode == "zscore":
        mu = np.nanmean(arr)
        sd = np.nanstd(arr)
        if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 1e-12:
            return out
        z = (arr - mu) / sd
        out[cols] = 1 / (1 + np.exp(-z))
        return out

    return out


def _set_rgrid(ax, data_min: float, data_max: float):
    span = data_max - data_min
    if not np.isfinite(span) or span <= 0:
        span = 1.0
        data_min = 0.0
    rings_shifted = np.linspace(0, span, 6)[1:]
    ax.set_yticks(rings_shifted)
    ax.set_yticklabels([f"{r:.2f}" for r in rings_shifted + data_min], fontsize=12)
    ax.set_ylim(0, span)


def plot_radar_places_axes(df: pd.DataFrame, metric: str, out_path: str,
                           models: Optional[List[str]] = None,
                           places: Optional[List[str]] = None,
                           radar_norm: str = "none",
                           save_pdf: bool = False,
        pdf_only: bool = False,
                           dataset_name: Optional[str] = None,
                           style_name: Optional[str] = None):
    """Radar Type A: axes=places, colors=models."""
    apply_style()

    df_f = _filter_df(df, models, places)
    if df_f.empty:
        return
    if metric not in df_f.columns:
        return

    ds_suffix = metric_dataset_suffix(metric, df_f, dataset_name=dataset_name)

    agg = _aggregate_for_radar(df_f, metric, style_name=style_name)
    if agg.empty:
        return

    models_u = sorted(agg["model"].unique().tolist())
    places_u = sorted(agg["place"].unique().tolist())

    single_model_mode = (metric == SPECIAL_SINGLE_MODEL_METRIC and len(models_u) > 0)
    if single_model_mode:
        keep_model = models_u[0]
        agg = agg[agg["model"] == keep_model].copy()
        models_u = [keep_model]

    tab = agg.pivot(index="model", columns="place", values=metric).reindex(index=models_u, columns=places_u)
    # Always use actual metric values for ring labels (no normalization for A/B).
    arr = tab.to_numpy(dtype=float)
    finite_vals = arr[np.isfinite(arr)]
    vmin = float(finite_vals.min()) if finite_vals.size > 0 else 0.0
    vmax = float(finite_vals.max()) if finite_vals.size > 0 else 1.0

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_position([0.0, 0.0, 0.80, 0.95])

    _sn = _style_display_name(style_name)
    _ds = dataset_name or ""
    _ds_part = f" — {_ds}" if _ds else ""
    title = f"{metric_display_name(metric)} by Model and Place{_ds_part}\n{_sn} (Avg. over Style Strength)"
    angles = _radar_setup(ax, places_u, title=title)

    model_colors = build_model_color_map(models_u)

    _set_rgrid(ax, vmin, vmax)

    for m in models_u:
        row = tab.loc[m].to_numpy(dtype=float)
        if np.all(np.isnan(row)):
            continue
        vals = (row - vmin).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=model_colors[m], label=m, linewidth=2.6)
        ax.fill(angles, vals, color=model_colors[m], alpha=0.12)

    ax.legend(loc="upper left", bbox_to_anchor=(1.25, 1.15), borderaxespad=0, frameon=True, framealpha=0.9, fontsize=16)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _save_plt_figure(out_path, save_pdf=save_pdf, pdf_only=pdf_only)
    plt.close()


def plot_radar_models_axes(df: pd.DataFrame, metric: str, out_path: str,
                           models: Optional[List[str]] = None,
                           places: Optional[List[str]] = None,
                           radar_norm: str = "none",
                           save_pdf: bool = False,
        pdf_only: bool = False,
                           dataset_name: Optional[str] = None,
                           style_name: Optional[str] = None):
    """Radar Type B: axes=models, colors=places."""
    apply_style()

    df_f = _filter_df(df, models, places)
    if df_f.empty:
        return
    if metric not in df_f.columns:
        return

    ds_suffix = metric_dataset_suffix(metric, df_f, dataset_name=dataset_name)

    agg = _aggregate_for_radar(df_f, metric, style_name=style_name)
    if agg.empty:
        return

    models_u = sorted(agg["model"].unique().tolist())
    places_u = sorted(agg["place"].unique().tolist())

    if metric == SPECIAL_SINGLE_MODEL_METRIC and len(models_u) <= 1:
        return

    tab = agg.pivot(index="place", columns="model", values=metric).reindex(index=places_u, columns=models_u)
    # Always use actual metric values for ring labels (no normalization for A/B).
    arr = tab.to_numpy(dtype=float)
    finite_vals = arr[np.isfinite(arr)]
    vmin = float(finite_vals.min()) if finite_vals.size > 0 else 0.0
    vmax = float(finite_vals.max()) if finite_vals.size > 0 else 1.0

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_position([0.0, 0.0, 0.80, 0.95])

    _sn = _style_display_name(style_name)
    _ds = dataset_name or ""
    _ds_part = f" — {_ds}" if _ds else ""
    title = f"{metric_display_name(metric)} by Model and Place{_ds_part}\n{_sn} (Avg. over Style Strength)"
    angles = _radar_setup(ax, models_u, title=title)

    place_colors = build_place_color_map(places_u)

    _set_rgrid(ax, vmin, vmax)

    for p in places_u:
        row = tab.loc[p].to_numpy(dtype=float)
        if np.all(np.isnan(row)):
            continue
        vals = (row - vmin).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=place_colors[p], label=p, linewidth=2.6)
        ax.fill(angles, vals, color=place_colors[p], alpha=0.10)

    ax.legend(loc="upper left", bbox_to_anchor=(1.25, 1.15), borderaxespad=0, frameon=True, framealpha=0.9, fontsize=16)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _save_plt_figure(out_path, save_pdf=save_pdf, pdf_only=pdf_only)
    plt.close()


def plot_radar_metrics_axes(df: pd.DataFrame, metrics: List[str], out_path: str,
                            models: Optional[List[str]] = None,
                            places: Optional[List[str]] = None,
                            radar_norm: str = "minmax",
                            save_pdf: bool = False,
        pdf_only: bool = False,
                            dataset_name: Optional[str] = None,
                            style_name: Optional[str] = None):
    """Radar Type C: axes=metrics, color=model, linestyle=place."""
    apply_style()

    d = _filter_df(df, models, places)
    if d.empty:
        return

    ds_suffix = dataset_title_suffix(d, dataset_name=dataset_name)

    keep_metrics = [m for m in metrics if m in d.columns and m in ALLOWED_METRICS]
    if not keep_metrics:
        return

    # Exclude baseline strength for this style before averaging.
    baseline = STYLE_BASELINE_STRENGTH.get((style_name or "").lower())

    rows = []
    for m in keep_metrics:
        tmp = d[["model", "place", "strength", m]].copy()
        tmp[m] = pd.to_numeric(tmp[m], errors="coerce")
        if baseline is not None and "strength" in tmp.columns:
            tmp = tmp[pd.to_numeric(tmp["strength"], errors="coerce") != baseline]
        agg = tmp.groupby(["model", "place"], dropna=False)[m].mean().reset_index()
        agg["metric"] = m
        agg = agg.rename(columns={m: "value"})
        rows.append(agg)

    long = pd.concat(rows, ignore_index=True)
    long = long.dropna(subset=["value"])
    if long.empty:
        return

    models_u = sorted(long["model"].unique().tolist())
    metrics_u = keep_metrics[:]

    tab = long.pivot_table(index=["model", "place"], columns="metric", values="value", aggfunc="mean")
    tab = tab.reindex(columns=metrics_u)
    tab = _normalize_table(tab, cols=metrics_u, mode=radar_norm)

    metric_labels = [metric_display_name(m) for m in metrics_u]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_position([0.0, 0.0, 0.80, 0.95])
    _sn = _style_display_name(style_name)
    _ds = dataset_name or ""
    _ds_part = f" — {_ds}" if _ds else ""
    title = f"Metric Profile by Model and Place{_ds_part}\n{_sn} (Avg. over Style Strength)"
    angles = _radar_setup(ax, metric_labels, title=title)

    model_colors = build_model_color_map(models_u)

    arr = tab.to_numpy(dtype=float)
    vmin = 0.0 if radar_norm != "none" else float(np.nanmin(arr)) if np.any(np.isfinite(arr)) else 0.0
    vmax = 1.0 if radar_norm != "none" else float(np.nanmax(arr)) if np.any(np.isfinite(arr)) else 1.0
    _set_rgrid(ax, vmin, vmax)

    for (model, place), row in tab.iterrows():
        vals = row.to_numpy(dtype=float)
        if np.all(np.isnan(vals)):
            continue
        vals = (vals - vmin).tolist()
        vals += vals[:1]
        ax.plot(
            angles,
            vals,
            color=model_colors[str(model)],
            linestyle=PLACE_LINESTYLE.get(str(place), "-"),
            linewidth=2.3,
            alpha=0.95,
            label=f"{model} · {place}",
        )

    ax.legend(loc="upper left", bbox_to_anchor=(1.25, 1.15), borderaxespad=0, frameon=True, framealpha=0.9, fontsize=16, ncol=1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _save_plt_figure(out_path, save_pdf=save_pdf, pdf_only=pdf_only)
    plt.close()


def _kde_1d(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Simple Gaussian KDE without scipy.
    Uses Silverman's rule for bandwidth, with safe fallbacks.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros_like(grid)

    if x.size == 1:
        bw = 1e-3 if np.isfinite(x[0]) else 1e-3
    else:
        sd = np.std(x, ddof=1)
        if not np.isfinite(sd) or sd <= 1e-12:
            sd = np.std(x, ddof=0)
        if not np.isfinite(sd) or sd <= 1e-12:
            sd = 1e-3
        bw = 1.06 * sd * (x.size ** (-1 / 5))

    bw = float(max(bw, 1e-6))
    diffs = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * diffs * diffs).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
    return dens


def plot_metric_ridge(
        df: pd.DataFrame,
        metric: str,
        out_path: str,
        models: Optional[List[str]] = None,
        places: Optional[List[str]] = None,
        save_pdf: bool = False,
        pdf_only: bool = False,
        max_strengths: int = 30,
        dataset_name: Optional[str] = None,
        style_name: Optional[str] = None,
):
    """
    Ridge plot: for each strength, show the distribution of metric values across (model,place).
    This visualizes strength effects without collapsing to mean.
    """
    apply_style()

    d = _filter_df(df, models, places)
    if d.empty or metric not in d.columns:
        return

    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=["strength", metric])
    if d.empty:
        return

    ds_suffix = metric_dataset_suffix(metric, d, dataset_name=dataset_name)

    strengths = _sorted_strength_values(d["strength"].dropna().unique().tolist())
    if not strengths:
        return

    if len(strengths) > max_strengths:
        idx = np.linspace(0, len(strengths) - 1, max_strengths).round().astype(int)
        strengths = [strengths[i] for i in idx]

    x_min = float(np.nanmin(d[metric].to_numpy(dtype=float)))
    x_max = float(np.nanmax(d[metric].to_numpy(dtype=float)))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        return
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5

    x_grid = np.linspace(x_min, x_max, 300)

    fig, ax = plt.subplots(figsize=(12, 7))

    offset = 1.0
    y_ticks = []
    y_ticklabels = []

    for i, s in enumerate(strengths):
        vals = d.loc[d["strength"] == s, metric].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        dens = _kde_1d(vals, x_grid)
        if np.nanmax(dens) > 0:
            dens = dens / (np.nanmax(dens) + 1e-12)

        base_y = i * offset
        ax.fill_between(x_grid, base_y, base_y + dens * 0.85, alpha=0.35)
        ax.plot(x_grid, base_y + dens * 0.85, linewidth=1.2)

        y_ticks.append(base_y + 0.15)
        y_ticklabels.append(str(s))

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels, fontsize=24)
    ax.set_xlabel(metric_display_name(metric), fontsize=28)
    ax.set_ylabel("Style Strength", fontsize=28)
    ax.tick_params(axis="x", labelsize=24)

    models_u = sorted(d["model"].unique().tolist()) if "model" in d.columns else []
    places_u = sorted(d["place"].unique().tolist()) if "place" in d.columns else []
    _sn = _style_display_name(style_name)
    _ds = dataset_name or ""
    _ds_part = f" — {_ds}" if _ds else ""
    ax.set_title(
        f"Mirroring Rate Distribution vs. {_sn} Strength{_ds_part}\n"
        f"Each ridge = strength; density over {len(models_u)} model(s) × {len(places_u)} place(s)",
        fontsize=26,
    )

    plt.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.10)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _save_plt_figure(out_path, save_pdf=save_pdf, pdf_only=pdf_only)
    plt.close()


# ============================================================
# Section 7: BERTScore prompt preservation plot
# (formerly at bottom of politeness_plots.py, used by polite_prompt_check.py)
# ============================================================

def build_dataset_color_map(datasets: List[str]) -> Dict[str, str]:
    """Map dataset names to distinct colors (uses the same palette as aggregate_plots)."""
    ds_sorted = sorted(datasets)
    return {d: DARK_MODEL_BASE[i % len(DARK_MODEL_BASE)] for i, d in enumerate(ds_sorted)}


def plot_bertscore_prompt_lines(
        df: pd.DataFrame,
        out_path_png: str,
        *,
        threshold: float = 0.85,
        save_pdf: bool = False,
        pdf_only: bool = False,
        style_name: Optional[str] = None,
) -> None:
    """
    Line plot of BERTScore(prompt) vs style strength across all datasets and places.
    Used by experiments/prompt_check.py and utils/run_plots.py --prompt_check.

    Args:
        df:            DataFrame with columns dataset, place, strength, bertscore_prompt.
        out_path_png:  Output PNG path.
        threshold:     Semantic-preservation threshold line drawn on the plot.
        save_pdf:      Also save a .pdf alongside the .png.
        style_name:    Human-readable style label (e.g. "politeness", "spacing") added
                       to the plot title.  Pass None to omit.
    """
    apply_style()

    needed = {"dataset", "place", "strength", "bertscore_prompt"}
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns for plotting: {miss}")

    d = df.copy()
    d["strength"] = pd.to_numeric(d["strength"], errors="coerce")
    d["bertscore_prompt"] = pd.to_numeric(d["bertscore_prompt"], errors="coerce")
    d = d.dropna(subset=["strength", "bertscore_prompt"])
    if d.empty:
        return

    datasets_u = sorted(d["dataset"].unique().tolist())
    places_u = sorted(d["place"].unique().tolist())
    color_map = build_dataset_color_map(datasets_u)

    fig, ax = plt.subplots(figsize=(12, 7))

    for dataset in datasets_u:
        base = color_map[dataset]
        for place in places_u:
            sub = d[(d["dataset"] == dataset) & (d["place"] == place)].sort_values("strength")
            if sub.empty:
                continue
            ax.plot(
                sub["strength"].values,
                sub["bertscore_prompt"].values,
                marker="o",
                markersize=4,
                linewidth=1.7,
                color=shade_for_place(base, place, places_u),
                linestyle=PLACE_LINESTYLE.get(place, "-"),
                alpha=0.95,
                label=f"{dataset} · {place}",
            )

    ax.set_xlabel("Style Strength", fontsize=28)
    ax.set_ylabel("BERTScore (Prompt)", fontsize=28)
    _x_min_r = int(round(ax.get_xlim()[0]))
    _x_max_r = int(round(ax.get_xlim()[1]))
    _step = max(1, int(round((_x_max_r - _x_min_r) / 5)))
    _start = (_x_min_r // _step) * _step
    _ticks = np.arange(_start, _x_max_r + _step, _step, dtype=int)
    ax.set_xticks(_ticks)
    ax.set_xticklabels([str(x) for x in _ticks], fontsize=24)
    ax.tick_params(axis="y", labelsize=24)

    style_suffix = _style_suffix(style_name)
    ax.set_title(
        f"BERTScore (Prompt) vs. Style Strength (All Datasets and Places){style_suffix}",
        fontsize=26,
    )

    ax.axhline(
        y=float(threshold),
        color="black",
        linestyle="--",
        linewidth=1.6,
        alpha=0.9,
        label=f"Threshold ({threshold:.2f})",
        zorder=3,
    )

    ymin = float(d["bertscore_prompt"].min())
    ymax = float(d["bertscore_prompt"].max())
    ax.set_ylim(min(ymin, threshold) - 0.03, max(ymax, threshold) + 0.03)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True, framealpha=0.9, ncol=1, fontsize=24)
    plt.subplots_adjust(left=0.10, right=0.78, top=0.90, bottom=0.12)
    os.makedirs(os.path.dirname(out_path_png) or ".", exist_ok=True)
    _save_plt_figure(out_path_png, save_pdf=save_pdf, pdf_only=pdf_only)
    plt.close()


# ============================================================
# Section 8: Multi-style subplot BERTScore prompt preservation
# For spacing/letter_case/punctuation (3 subplots) and
# length_variation/inter_vs_imper (2 subplots)
# ============================================================

def save_bertscore_legend_image(
        all_datasets: List[str],
        all_places: List[str],
        color_map: Dict[str, str],
        out_path: str,
        threshold: float = 0.85,
        save_pdf: bool = False,
) -> None:
    """Save a standalone legend image: one row per place, datasets as columns."""
    apply_style()
    places   = sorted(all_places)
    datasets = sorted(all_datasets)
    n_ds     = len(datasets)

    n_rows = len(places) + 1   # one row per place + one row for threshold
    fig_w  = max(10.0, 2.4 * n_ds)
    fig_h  = 1.4 * n_rows
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h))

    for ax, place in zip(axes, places):
        ax.axis("off")
        handles = []
        labels  = []
        for ds in datasets:
            base  = color_map[ds]
            color = shade_for_place(base, place, places)
            ls    = PLACE_LINESTYLE.get(place, "-")
            handles.append(Line2D([0], [0], color=color, linestyle=ls, linewidth=4.0))
            labels.append(ds)
        ax.legend(
            handles, labels,
            loc="center", ncol=len(handles),
            title=place.capitalize(),
            fontsize=22, title_fontsize=22,
            frameon=True, framealpha=0.95,
            borderpad=0.6, handletextpad=0.5, columnspacing=1.2,
            handlelength=3.5,
        )

    # Threshold row — separate axis
    ax_thr = axes[-1]
    ax_thr.axis("off")
    ax_thr.legend(
        [Line2D([0], [0], color="black", linestyle="--", linewidth=3.0)],
        [f"Threshold ({threshold:.2f})"],
        loc="center", ncol=1,
        fontsize=22, frameon=True, framealpha=0.95,
        borderpad=0.6, handletextpad=0.5, handlelength=2.5,
    )

    plt.tight_layout(h_pad=0.4)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


def plot_bertscore_prompt_subplots(
        dfs: List[pd.DataFrame],
        style_titles: List[str],
        out_path_png: str,
        *,
        threshold: float = 0.85,
        save_pdf: bool = False,
        pdf_only: bool = False,
        suptitle: str = "",
        force_xmin_zero: bool = False,
) -> None:
    """
    One figure with N subplots (one per style), all sharing the same visual theme
    as plot_bertscore_prompt_lines.  Each subplot is a line plot of
    BERTScore(prompt) vs style strength, with lines per dataset × place.

    Used for:
      - 3 subplots: spacing | letter_case | punctuation
      - 2 subplots: length_variation | inter_vs_imper

    Args:
        dfs:           List of DataFrames, one per style.  Each must have columns
                       dataset, place, strength, bertscore_prompt.
        style_titles:  Human-readable title for each subplot (same length as dfs).
        out_path_png:  Output PNG path.
        threshold:     Horizontal dashed threshold line drawn on every subplot.
        save_pdf:      Also save a .pdf alongside the .png.
        suptitle:      Optional overall figure title (suptitle).
    """
    apply_style()

    n = len(dfs)
    if n == 0:
        return

    # Gather a unified dataset/place set for a consistent legend across all subplots
    all_datasets: List[str] = []
    all_places: List[str] = []
    for df in dfs:
        for ds in df["dataset"].dropna().unique():
            if ds not in all_datasets:
                all_datasets.append(ds)
        for pl in df["place"].dropna().unique():
            if pl not in all_places:
                all_places.append(pl)
    all_datasets = sorted(all_datasets)
    all_places = sorted(all_places)
    color_map = build_dataset_color_map(all_datasets)

    # Unified y-limits across all subplots
    global_ymin = min(
        float(df["bertscore_prompt"].min()) for df in dfs
        if not df.empty and "bertscore_prompt" in df.columns
    )
    global_ymax = max(
        float(df["bertscore_prompt"].max()) for df in dfs
        if not df.empty and "bertscore_prompt" in df.columns
    )
    ylo = min(global_ymin, threshold) - 0.03
    yhi = max(global_ymax, threshold) + 0.03

    # Match per-place plot scale: axes 8.16 x 5.46 in (12*(0.78-0.10) x 7*(0.90-0.12))
    _ax_w = 12.0 * (0.78 - 0.10)   # 8.16 in — axes width of a single per-place plot
    _ax_h = 7.0  * (0.90 - 0.12)   # 5.46 in — axes height of a single per-place plot
    _wspace, _left, _right = 0.30, 0.08, 0.97
    fig_w = _ax_w * (n + _wspace * (n - 1)) / (_right - _left)
    fig_h = _ax_h / (0.90 - 0.12)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, df, title in zip(axes, dfs, style_titles):
        d = df.copy()
        d["bertscore_prompt"] = pd.to_numeric(d["bertscore_prompt"], errors="coerce")
        d = d.dropna(subset=["bertscore_prompt"])

        # Determine whether strength is numeric or categorical (inter_vs_imper)
        strength_vals = d["strength"].dropna().unique().tolist()
        numeric_strength = all(
            isinstance(v, (int, float)) or (isinstance(v, str) and _is_numeric(v))
            for v in strength_vals
        )

        if numeric_strength:
            d["strength"] = pd.to_numeric(d["strength"], errors="coerce")
            d = d.dropna(subset=["strength"])
            x_key = "strength"
        else:
            # Categorical (e.g. "interrogative", "imperative") — map to positions
            cats = sorted(d["strength"].unique().tolist())
            cat_pos = {c: i for i, c in enumerate(cats)}
            d = d.copy()
            d["_x"] = d["strength"].map(cat_pos)
            x_key = "_x"

        for dataset in all_datasets:
            base = color_map[dataset]
            for place in all_places:
                sub = d[(d["dataset"] == dataset) & (d["place"] == place)].sort_values(x_key)
                if sub.empty:
                    continue
                ax.plot(
                    sub[x_key].values,
                    sub["bertscore_prompt"].values,
                    marker="o",
                    markersize=6,
                    linewidth=2.5,
                    color=shade_for_place(base, place, all_places),
                    linestyle=PLACE_LINESTYLE.get(place, "-"),
                    alpha=0.95,
                    label=f"{dataset} · {place}",
                )

        ax.axhline(
            y=float(threshold),
            color="black",
            linestyle="--",
            linewidth=1.4,
            alpha=0.85,
            label=f"Threshold ({threshold:.2f})",
            zorder=3,
        )

        ax.set_title(title, fontsize=26, pad=6)
        ax.set_xlabel("Style Strength", fontsize=28)
        ax.tick_params(axis="y", labelsize=24)
        ax.set_ylim(ylo, yhi)

        if not numeric_strength:
            cats = sorted(d["strength"].unique().tolist())
            cat_pos_final = {c: i for i, c in enumerate(cats)}
            ax.set_xticks(list(cat_pos_final.values()))
            ax.set_xticklabels(list(cat_pos_final.keys()), ha="right", fontsize=24)
        else:
            if force_xmin_zero:
                ax.set_xlim(left=0)
            _x_min_r = 0 if force_xmin_zero else int(round(ax.get_xlim()[0]))
            _x_max_r = int(round(ax.get_xlim()[1]))
            _step = max(1, int(round((_x_max_r - _x_min_r) / 5)))
            _ticks = np.arange(_x_min_r, _x_max_r + _step, _step, dtype=int)
            ax.set_xticks(_ticks)
            ax.set_xticklabels([str(x) for x in _ticks], fontsize=24)

    # Y label only on the leftmost subplot
    axes[0].set_ylabel("BERTScore", fontsize=28)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.12, wspace=0.30)
    os.makedirs(os.path.dirname(out_path_png) or ".", exist_ok=True)
    _save_plt_figure(out_path_png, save_pdf=save_pdf, pdf_only=pdf_only)
    plt.close()


