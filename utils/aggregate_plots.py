# utils/aggregate_plots.py
"""
Global plotting utilities for styled-prompt experiments.

What this script does
---------------------
1) Load and merge run summaries from multiple run directories.
   Expected file per run:
     <RUN_DIR>/plots_metrics/combined_means_by_model_place_strength.csv

2) For EACH metric column (restricted to a selected set):
   - Line plot TYPE 1 : all models+places together
       color = model, shade = place
   - Line plot TYPE 2 : per-model (one figure per model), combine all places
       color shade = place, single base color for the model
   - Line plot TYPE 3 : per-place (one figure per place), combine all models
       color = model (no shading)

   - Radar plot Type A: axes=places, lines=models (colors=models).
   - Radar plot Type B: axes=models, lines=places (colors=places).
   - Radar plot Type C: axes=metrics, lines=(model,place) with:
        color=model, linestyle=place.

"""

import os
import re
import argparse
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


# ============================================================
# Metric selection + display name mapping
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
    "bertscore_response": "BERTScore (Response)",
    "delta_log_prob": "Δ Log Prob",
    "entropy_shift": "Entropy Shift",
    "activation_similarity": "Activation Similarity",
    "mirroring_rate": "Mirroring Rate",
    "bertscore_prompt": "BERTScore (Prompt)",
    # NEW:
    "asr": "ASR",
    "silhouette": "Silhouette Score",
}


def metric_display_name(metric: str) -> str:
    return METRIC_DISPLAY.get(metric, metric)


# ============================================================
# Dataset name (TruthfulQA / Natural Questions) for titles
# ============================================================

_DATASET_CANON = {
    "truthfulqa": "TruthfulQA",
    "truthful_qa": "TruthfulQA",
    "natural_questions": "Natural Questions",
    "naturalquestions": "Natural Questions",
    "nq": "Natural Questions",
}


def _canon_dataset_name(raw: str) -> Optional[str]:
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


# ============================================================
# Style (bigger fonts + visible polar grids)
# ============================================================

def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,

        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.9,

        "lines.linewidth": 2.4,
        "lines.markersize": 5,
    })


# ============================================================
# Loading
# ============================================================

def _find_means_csv(run_dir: str) -> Optional[str]:
    # User requested: treat --runs as direct CSV path(s)
    cand = run_dir
    if os.path.exists(cand):
        return cand
    return None


def load_all_runs(run_dirs: List[str]) -> pd.DataFrame:
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

    for c in ["strength"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def infer_metric_columns(df: pd.DataFrame) -> List[str]:
    base = {"model", "place", "strength", "run_dir"}
    metrics = [c for c in df.columns if c not in base]
    keep = []
    for c in metrics:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            keep.append(c)
    return keep


def select_metrics(df: pd.DataFrame) -> List[str]:
    present = set(infer_metric_columns(df))
    chosen = [m for m in ALLOWED_METRICS if m in present]
    if SPECIAL_SINGLE_MODEL_METRIC in present:
        chosen.append(SPECIAL_SINGLE_MODEL_METRIC)
    return chosen


# ============================================================
# Colors: darker palette + per-place shade
# ============================================================

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
    rgb = np.array(to_rgb(hex_color))
    return tuple((1 - alpha) * rgb + alpha * np.ones_like(rgb))


def build_model_color_map(models: List[str]) -> Dict[str, str]:
    models_sorted = sorted(models)
    return {m: DARK_MODEL_BASE[i % len(DARK_MODEL_BASE)] for i, m in enumerate(models_sorted)}


def build_place_color_map(places: List[str]) -> Dict[str, str]:
    fallback = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    out = {}
    for i, p in enumerate(sorted(places)):
        out[p] = DARK_PLACE_BASE.get(p, fallback[i % len(fallback)])
    return out


def shade_for_place(base_color: str, place: str, places_sorted: List[str]) -> Tuple[float, float, float]:
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
    s = str(s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s[:200]


def _filter_df(df: pd.DataFrame, models: Optional[List[str]], places: Optional[List[str]]) -> pd.DataFrame:
    d = df.copy()
    if models is not None:
        d = d[d["model"].isin(models)]
    if places is not None:
        d = d[d["place"].isin(places)]
    return d


# ============================================================
# Line plot TYPE 1: all models + all places (existing)
# ============================================================

def plot_metric_lines(
        df: pd.DataFrame,
        metric: str,
        out_path: str,
        models: Optional[List[str]] = None,
        places: Optional[List[str]] = None,
        save_pdf: bool = False,
        dataset_name: Optional[str] = None,
):
    apply_style()

    d = _filter_df(df, models, places)

    if d.empty or metric not in d.columns:
        return

    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=["strength", metric])
    if d.empty:
        return

    ds_suffix = metric_dataset_suffix(metric, d, dataset_name=dataset_name)

    models_u = sorted(d["model"].unique().tolist())
    places_u = sorted(d["place"].unique().tolist())

    single_model_mode = (metric == SPECIAL_SINGLE_MODEL_METRIC and len(models_u) > 0)
    if single_model_mode:
        keep_model = models_u[0]
        d = d[d["model"] == keep_model].copy()
        models_u = [keep_model]

    model_colors = build_model_color_map(models_u)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    for model in models_u:
        base = model_colors[model]
        for place in places_u:
            sub = d[(d["model"] == model) & (d["place"] == place)].sort_values("strength")
            if sub.empty:
                continue

            series_label = f"{place}" if single_model_mode else f"{model} · {place}"

            ax.plot(
                sub["strength"].values,
                sub[metric].values,
                marker="o",
                markersize=4,
                linewidth=1.6,
                color=shade_for_place(base, place, places_u),
                linestyle=PLACE_LINESTYLE.get(place, "-"),
                alpha=0.9,
                label=series_label,
            )

    ax.set_xlabel("Style Strength")
    ax.set_ylabel(metric_display_name(metric))
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if single_model_mode:
        ax.set_title(f"{metric_display_name(metric)} vs. Style Strength{ds_suffix} (All Places)")
    else:
        ax.set_title(f"{metric_display_name(metric)} vs. Style Strength{ds_suffix} (All Models and Places)")

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
    ax.margins(y=0.05)

    ax.legend(
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False,
        ncol=1,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if save_pdf:
        plt.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# Line plot TYPE 2 (NEW): per-model (combine all places)
#   One figure per model, shade lines by place.
# ============================================================

def plot_metric_lines_per_model(
        df: pd.DataFrame,
        metric: str,
        out_dir: str,
        models: Optional[List[str]] = None,
        places: Optional[List[str]] = None,
        save_pdf: bool = False,
        dataset_name: Optional[str] = None,
):
    apply_style()

    d = _filter_df(df, models, places)
    if d.empty or metric not in d.columns:
        return

    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=["strength", metric])
    if d.empty:
        return

    ds_suffix = metric_dataset_suffix(metric, d, dataset_name=dataset_name)

    models_u = sorted(d["model"].unique().tolist())
    places_u = sorted(d["place"].unique().tolist())

    single_model_mode = (metric == SPECIAL_SINGLE_MODEL_METRIC and len(models_u) > 0)
    if single_model_mode:
        # keep existing single-model behavior: just one model anyway
        models_u = [models_u[0]]
        d = d[d["model"] == models_u[0]].copy()

    model_colors = build_model_color_map(models_u)

    os.makedirs(out_dir, exist_ok=True)

    for model in models_u:
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        base = model_colors[model]

        for place in places_u:
            sub = d[(d["model"] == model) & (d["place"] == place)].sort_values("strength")
            if sub.empty:
                continue

            ax.plot(
                sub["strength"].values,
                sub[metric].values,
                marker="o",
                markersize=4,
                linewidth=1.8,
                color=shade_for_place(base, place, places_u),
                linestyle=PLACE_LINESTYLE.get(place, "-"),
                alpha=0.95,
                label=str(place),
            )

        ax.set_xlabel("Style Strength")
        ax.set_ylabel(metric_display_name(metric))
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_title(f"{metric_display_name(metric)} vs. Style Strength{ds_suffix} ({model})")

        ymin = float(d[d["model"] == model][metric].min())
        ymax = float(d[d["model"] == model][metric].max())
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
            pad = 0.12 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

        if metric == "bertscore_prompt":
            thr = 0.85
            ax.axhline(y=thr, color="black", linestyle="--", linewidth=1.5, alpha=0.8, zorder=3)

        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{_sanitize_filename(metric)}_line_per_model__{_sanitize_filename(model)}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        if save_pdf:
            plt.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
        plt.close()


# ============================================================
# Line plot TYPE 3 (NEW): per-place (combine all models)
#   One figure per place, color lines by model.
# ============================================================

def plot_metric_lines_per_place(
        df: pd.DataFrame,
        metric: str,
        out_dir: str,
        models: Optional[List[str]] = None,
        places: Optional[List[str]] = None,
        save_pdf: bool = False,
        dataset_name: Optional[str] = None,
):
    apply_style()

    d = _filter_df(df, models, places)
    if d.empty or metric not in d.columns:
        return

    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=["strength", metric])
    if d.empty:
        return

    ds_suffix = metric_dataset_suffix(metric, d, dataset_name=dataset_name)

    models_u = sorted(d["model"].unique().tolist())
    places_u = sorted(d["place"].unique().tolist())

    single_model_mode = (metric == SPECIAL_SINGLE_MODEL_METRIC and len(models_u) > 0)
    if single_model_mode:
        # in special single-model metric, per-place plot is trivial but still valid
        keep_model = models_u[0]
        d = d[d["model"] == keep_model].copy()
        models_u = [keep_model]

    model_colors = build_model_color_map(models_u)

    os.makedirs(out_dir, exist_ok=True)

    for place in places_u:
        fig, ax = plt.subplots(figsize=(7.6, 4.4))

        for model in models_u:
            sub = d[(d["model"] == model) & (d["place"] == place)].sort_values("strength")
            if sub.empty:
                continue

            ax.plot(
                sub["strength"].values,
                sub[metric].values,
                marker="o",
                markersize=4,
                linewidth=1.8,
                color=model_colors[model],
                linestyle="-",
                alpha=0.95,
                label=str(model),
            )

        ax.set_xlabel("Style Strength")
        ax.set_ylabel(metric_display_name(metric))
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_title(f"{metric_display_name(metric)} vs. Style Strength{ds_suffix} ({place})")

        ymin = float(d[d["place"] == place][metric].min())
        ymax = float(d[d["place"] == place][metric].max())
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
            pad = 0.12 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

        if metric == "bertscore_prompt":
            thr = 0.85
            ax.axhline(y=thr, color="black", linestyle="--", linewidth=1.5, alpha=0.8, zorder=3)

        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{_sanitize_filename(metric)}_line_per_place__{_sanitize_filename(place)}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        if save_pdf:
            plt.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
        plt.close()


# ============================================================
# Radar helpers
# ============================================================

def _radar_setup(ax, labels: List[str], title: Optional[str] = None):
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)

    ax.set_rlabel_position(0)
    ax.grid(True, alpha=0.45, linewidth=0.9)

    if title:
        ax.set_title(title, pad=18, fontsize=14)

    return angles


def _aggregate_for_radar(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    d = df.copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
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


def _set_rgrid(ax, data_max: float):
    if not np.isfinite(data_max) or data_max <= 0:
        data_max = 1.0
    rings = np.linspace(0, data_max, 6)[1:]
    ax.set_yticks(rings)
    ax.set_yticklabels([f"{r:.2f}" for r in rings], fontsize=10)
    ax.set_ylim(0, data_max)


# ============================================================
# Radar Type A: axes=places, colors=models
# ============================================================

def plot_radar_places_axes(df: pd.DataFrame, metric: str, out_path: str,
                           models: Optional[List[str]] = None,
                           places: Optional[List[str]] = None,
                           radar_norm: str = "none",
                           save_pdf: bool = False,
                           dataset_name: Optional[str] = None):
    apply_style()

    df_f = _filter_df(df, models, places)
    if df_f.empty:
        return
    if metric not in df_f.columns:
        return

    ds_suffix = metric_dataset_suffix(metric, df_f, dataset_name=dataset_name)

    agg = _aggregate_for_radar(df_f, metric)
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
    tab = _normalize_table(tab, cols=places_u, mode=radar_norm)

    fig, ax = plt.subplots(figsize=(6.2, 6.2), subplot_kw=dict(polar=True))

    if single_model_mode:
        title = f"{metric_display_name(metric)} by Place{ds_suffix} (Avg. over Style Strength)"
    else:
        title = f"{metric_display_name(metric)} by Place and Model{ds_suffix} (Avg. over Style Strength)"
    angles = _radar_setup(ax, places_u, title=title)

    model_colors = build_model_color_map(models_u)

    vmax = np.nanmax(tab.to_numpy(dtype=float))
    if radar_norm != "none":
        vmax = 1.0
    _set_rgrid(ax, float(vmax) if np.isfinite(vmax) else 1.0)

    for m in models_u:
        row = tab.loc[m].to_numpy(dtype=float)
        if np.all(np.isnan(row)):
            continue
        vals = row.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=model_colors[m], label=m, linewidth=2.6)
        ax.fill(angles, vals, color=model_colors[m], alpha=0.12)

    ax.legend(bbox_to_anchor=(1.15, 1.1), loc="upper left", frameon=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if save_pdf:
        plt.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# Radar Type B: axes=models, colors=places
# ============================================================

def plot_radar_models_axes(df: pd.DataFrame, metric: str, out_path: str,
                           models: Optional[List[str]] = None,
                           places: Optional[List[str]] = None,
                           radar_norm: str = "none",
                           save_pdf: bool = False,
                           dataset_name: Optional[str] = None):
    apply_style()

    df_f = _filter_df(df, models, places)
    if df_f.empty:
        return
    if metric not in df_f.columns:
        return

    ds_suffix = metric_dataset_suffix(metric, df_f, dataset_name=dataset_name)

    agg = _aggregate_for_radar(df_f, metric)
    if agg.empty:
        return

    models_u = sorted(agg["model"].unique().tolist())
    places_u = sorted(agg["place"].unique().tolist())

    if metric == SPECIAL_SINGLE_MODEL_METRIC and len(models_u) <= 1:
        return

    tab = agg.pivot(index="place", columns="model", values=metric).reindex(index=places_u, columns=models_u)
    tab = _normalize_table(tab, cols=models_u, mode=radar_norm)

    fig, ax = plt.subplots(figsize=(6.6, 6.6), subplot_kw=dict(polar=True))

    title = f"{metric_display_name(metric)} by Model and Place{ds_suffix} (Avg. over Style Strength)"
    angles = _radar_setup(ax, models_u, title=title)

    place_colors = build_place_color_map(places_u)

    vmax = np.nanmax(tab.to_numpy(dtype=float))
    if radar_norm != "none":
        vmax = 1.0
    _set_rgrid(ax, float(vmax) if np.isfinite(vmax) else 1.0)

    for p in places_u:
        row = tab.loc[p].to_numpy(dtype=float)
        if np.all(np.isnan(row)):
            continue
        vals = row.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=place_colors[p], label=p, linewidth=2.6)
        ax.fill(angles, vals, color=place_colors[p], alpha=0.10)

    ax.legend(bbox_to_anchor=(1.15, 1.1), loc="upper left", frameon=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if save_pdf:
        plt.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# Radar Type C: axes=metrics, color=model, linestyle=place
# ============================================================

def plot_radar_metrics_axes(df: pd.DataFrame, metrics: List[str], out_path: str,
                            models: Optional[List[str]] = None,
                            places: Optional[List[str]] = None,
                            radar_norm: str = "minmax",
                            save_pdf: bool = False,
                            dataset_name: Optional[str] = None):
    apply_style()

    d = _filter_df(df, models, places)
    if d.empty:
        return

    ds_suffix = dataset_title_suffix(d, dataset_name=dataset_name)

    keep_metrics = [m for m in metrics if m in d.columns and m in ALLOWED_METRICS]
    if not keep_metrics:
        return

    rows = []
    for m in keep_metrics:
        tmp = d[["model", "place", "strength", m]].copy()
        tmp[m] = pd.to_numeric(tmp[m], errors="coerce")
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

    fig, ax = plt.subplots(figsize=(7.0, 7.0), subplot_kw=dict(polar=True))
    title = (
        f"Metric Profile by Model and Place{ds_suffix}\n"
        f"(Normalized, Avg. over Style Strength)"
    )
    angles = _radar_setup(ax, metric_labels, title=title)

    model_colors = build_model_color_map(models_u)

    vmax = np.nanmax(tab.to_numpy(dtype=float))
    if radar_norm != "none":
        vmax = 1.0
    _set_rgrid(ax, float(vmax) if np.isfinite(vmax) else 1.0)

    for (model, place), row in tab.iterrows():
        vals = row.to_numpy(dtype=float)
        if np.all(np.isnan(vals)):
            continue
        vals = vals.tolist()
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

    ax.legend(bbox_to_anchor=(1.25, 1.08), loc="upper left", frameon=False, ncol=1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if save_pdf:
        plt.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# Ridge plot (distribution per strength)
# ============================================================

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
        max_strengths: int = 30,
        dataset_name: Optional[str] = None,
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

    strengths = sorted(d["strength"].dropna().unique().tolist())
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

    fig, ax = plt.subplots(figsize=(9.0, 6.2))

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
    ax.set_yticklabels(y_ticklabels)
    ax.set_xlabel(metric_display_name(metric))
    ax.set_ylabel("Style Strength")

    models_u = sorted(d["model"].unique().tolist()) if "model" in d.columns else []
    places_u = sorted(d["place"].unique().tolist()) if "place" in d.columns else []
    ax.set_title(
        f"{metric_display_name(metric)} Distribution over Style Strength{ds_suffix}\n"
        f"Each ridge = strength; density over {len(models_u)} model(s) × {len(places_u)} place(s)"
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if save_pdf:
        plt.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories (CSV paths).")
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--places", nargs="+", default=None)

    parser.add_argument("--metrics", nargs="+", default=None,
                        help="(Ignored except for intersection with allowed metrics.)")
    parser.add_argument("--save_pdf", action="store_true")

    parser.add_argument("--radar_norm", type=str, default="minmax",
                        choices=["none", "minmax", "zscore"],
                        help="Normalization for radar plots. Recommended: minmax.")
    parser.add_argument("--dataset_name", type=str, default=None,
                        choices=["TruthfulQA", "Natural Questions"])

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_all_runs(args.runs)

    if args.models is not None:
        df = df[df["model"].isin(args.models)]
    if args.places is not None:
        df = df[df["place"].isin(args.places)]

    if df.empty:
        raise SystemExit("No data left after filtering.")

    metrics = select_metrics(df)

    if args.metrics is not None:
        metrics = [m for m in metrics if m in set(args.metrics)]

    # --- Existing plots: Line + Radar A+B ---
    for metric in metrics:
        print(f"[PLOT] {metric}")

        # TYPE 1: all models + all places
        plot_metric_lines(
            df, metric,
            out_path=os.path.join(args.out_dir, f"{metric}_line.png"),
            models=args.models,
            places=args.places,
            save_pdf=bool(args.save_pdf),
            dataset_name=args.dataset_name,
        )

        # TYPE 2: per-model (combine all places)
        plot_metric_lines_per_model(
            df, metric,
            out_dir=os.path.join(args.out_dir, "line_per_model"),
            models=args.models,
            places=args.places,
            save_pdf=bool(args.save_pdf),
            dataset_name=args.dataset_name,
        )

        # TYPE 3: per-place (combine all models)
        plot_metric_lines_per_place(
            df, metric,
            out_dir=os.path.join(args.out_dir, "line_per_place"),
            models=args.models,
            places=args.places,
            save_pdf=bool(args.save_pdf),
            dataset_name=args.dataset_name,
        )

        plot_radar_places_axes(
            df, metric,
            out_path=os.path.join(args.out_dir, f"{metric}_radar_axes_places.png"),
            models=args.models,
            places=args.places,
            radar_norm=args.radar_norm,
            save_pdf=bool(args.save_pdf),
            dataset_name=args.dataset_name,
        )

        plot_radar_models_axes(
            df, metric,
            out_path=os.path.join(args.out_dir, f"{metric}_radar_axes_models.png"),
            models=args.models,
            places=args.places,
            radar_norm=args.radar_norm,
            save_pdf=bool(args.save_pdf),
            dataset_name=args.dataset_name,
        )

    # --- Existing plot: Radar Type C ---
    metrics_for_type_c = [
        m for m in metrics
        if m in ALLOWED_METRICS and m not in ["bleu", "delta_log_prob"]
    ]

    plot_radar_metrics_axes(
        df,
        metrics=metrics_for_type_c,
        out_path=os.path.join(args.out_dir, "radar_axes_metrics_all.png"),
        models=args.models,
        places=args.places,
        radar_norm=args.radar_norm,
        save_pdf=bool(args.save_pdf),
        dataset_name=args.dataset_name,
    )

    # ============================================================
    # Ridge plots (only; nothing else added/removed)
    # ============================================================

    ridge_dir = os.path.join(args.out_dir, "ridge_plots")
    os.makedirs(ridge_dir, exist_ok=True)
    for metric in metrics:
        plot_metric_ridge(
            df,
            metric=metric,
            out_path=os.path.join(ridge_dir, f"{metric}_ridge.png"),
            models=args.models,
            places=args.places,
            save_pdf=bool(args.save_pdf),
            dataset_name=args.dataset_name,
        )

    print(f"\n✓ Done. Plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()