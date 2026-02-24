# plots.py
"""
Global plotting utilities for styled-prompt experiments.

What this script does
---------------------
1) Load and merge run summaries from multiple run directories.
   Expected file per run:
     <RUN_DIR>/plots_metrics/combined_means_by_model_place_strength.csv

2) For EACH metric column (restricted to a selected set):
   - Line plot: x=strength (interpreted as), y=metric,
     one color per model, shades per place.
   - Radar plot Type A: axes=places, lines=models (colors=models).
   - Radar plot Type B: axes=models, lines=places (colors=places).
   - Radar plot Type C: axes=metrics, lines=(model,place) with:
        color=model, linestyle=place.

Notes
-----
- This file intentionally keeps your plotting logic the same, except:
  * Titles and axis labels are fixed as requested.
  * Metric display names are mapped (y-axis + titles).
  * Only a specific metric subset is plotted (plus optional bertscore_prompt special case).
  * bertscore_prompt is plotted for only ONE model across 3 places, and title omits model.
"""

import os
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
}


def metric_display_name(metric: str) -> str:
    return METRIC_DISPLAY.get(metric, metric)


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
    cand = os.path.join(run_dir, "plots_metrics", "combined_means_by_model_place_strength.csv")
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

    # Coerce main keys
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
    """
    Keep ONLY the allowed metrics, plus bertscore_prompt if present (special-case).
    """
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
    """
    alpha in [0,1]: 0 -> original color, 1 -> white
    """
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


# ============================================================
# Line plot: metric vs "Style Strength" (strength)
# ============================================================
def plot_metric_lines(
        df: pd.DataFrame,
        metric: str,
        out_path: str,
        models: Optional[List[str]] = None,
        places: Optional[List[str]] = None,
        save_pdf: bool = False,
):
    apply_style()

    d = df.copy()

    if models is not None:
        d = d[d["model"].isin(models)]
    if places is not None:
        d = d[d["place"].isin(places)]

    if d.empty or metric not in d.columns:
        return

    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=["strength", metric])
    if d.empty:
        return

    models_u = sorted(d["model"].unique().tolist())
    places_u = sorted(d["place"].unique().tolist())

    # Special case: bertscore_prompt -> show ONLY ONE model (since same across models)
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

            # Special case: labels omit model for bertscore_prompt
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

    # Axis labels + title fixes
    ax.set_xlabel("Style Strength")
    ax.set_ylabel(metric_display_name(metric))
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if single_model_mode:
        ax.set_title(f"{metric_display_name(metric)} vs. Style Strength (All Places)")
    else:
        ax.set_title(f"{metric_display_name(metric)} vs. Style Strength (All Models and Places)")

    # ---- Dynamic y padding (prevents stiff look) ----
    ymin = float(d[metric].min())
    ymax = float(d[metric].max())
    if ymin != ymax:
        pad = 0.12 * (ymax - ymin)
        ylo, yhi = ymin - pad, ymax + pad
    else:
        ylo, yhi = ymin - 0.1, ymax + 0.1

    # Add threshold line ONLY for prompt plot, and ensure it's within ylim
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
                           save_pdf: bool = False):
    apply_style()

    agg = _aggregate_for_radar(df, metric)
    if models is not None:
        agg = agg[agg["model"].isin(models)]
    if places is not None:
        agg = agg[agg["place"].isin(places)]
    if agg.empty:
        return

    models_u = sorted(agg["model"].unique().tolist())
    places_u = sorted(agg["place"].unique().tolist())

    # Special case: bertscore_prompt -> show ONLY ONE model
    if metric == SPECIAL_SINGLE_MODEL_METRIC and len(models_u) > 0:
        keep_model = models_u[0]
        agg = agg[agg["model"] == keep_model].copy()
        models_u = [keep_model]

    tab = agg.pivot(index="model", columns="place", values=metric).reindex(index=models_u, columns=places_u)
    tab = _normalize_table(tab, cols=places_u, mode=radar_norm)

    fig, ax = plt.subplots(figsize=(6.2, 6.2), subplot_kw=dict(polar=True))

    if metric == SPECIAL_SINGLE_MODEL_METRIC:
        title = f"{metric_display_name(metric)} by Place (Averaged over Style Strength)"
    else:
        title = f"{metric_display_name(metric)} by Place and Model (Averaged over Style Strength)"

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
                           save_pdf: bool = False):
    apply_style()

    agg = _aggregate_for_radar(df, metric)
    if models is not None:
        agg = agg[agg["model"].isin(models)]
    if places is not None:
        agg = agg[agg["place"].isin(places)]
    if agg.empty:
        return

    models_u = sorted(agg["model"].unique().tolist())
    places_u = sorted(agg["place"].unique().tolist())

    # Special case: bertscore_prompt -> show ONLY ONE model, but this plot is axes=models.
    # With one model, the radar becomes degenerate; we keep behavior safe by returning.
    if metric == SPECIAL_SINGLE_MODEL_METRIC:
        if len(models_u) <= 1:
            return

    tab = agg.pivot(index="place", columns="model", values=metric).reindex(index=places_u, columns=models_u)
    tab = _normalize_table(tab, cols=models_u, mode=radar_norm)

    fig, ax = plt.subplots(figsize=(6.6, 6.6), subplot_kw=dict(polar=True))
    title = f"{metric_display_name(metric)} by Model and Place (Averaged over Style Strength)"
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
                            save_pdf: bool = False):
    apply_style()

    d = df.copy()
    if models is not None:
        d = d[d["model"].isin(models)]
    if places is not None:
        d = d[d["place"].isin(places)]
    if d.empty:
        return

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
    places_u = sorted(long["place"].unique().tolist())
    metrics_u = keep_metrics[:]  # keep order

    tab = long.pivot_table(index=["model", "place"], columns="metric", values="value", aggfunc="mean")
    tab = tab.reindex(columns=metrics_u)
    tab = _normalize_table(tab, cols=metrics_u, mode=radar_norm)

    metric_labels = [metric_display_name(m) for m in metrics_u]

    fig, ax = plt.subplots(figsize=(7.0, 7.0), subplot_kw=dict(polar=True))
    title = "Metric Profile by Model and Place (Normalized, Averaged over Style Strength)"
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
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories.")
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--places", nargs="+", default=None)

    # Kept for compatibility, but we will still enforce allowed metrics + optional bertscore_prompt.
    parser.add_argument("--metrics", nargs="+", default=None, help="(Ignored except for intersection with allowed metrics.)")
    parser.add_argument("--save_pdf", action="store_true")

    parser.add_argument("--radar_norm", type=str, default="minmax",
                        choices=["none", "minmax", "zscore"],
                        help="Normalization for radar plots. Recommended: minmax.")

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

    # If user provided --metrics, intersect with our selected set (no other changes)
    if args.metrics is not None:
        metrics = [m for m in metrics if m in set(args.metrics)]

    # --- Line + per-metric radars (A+B) ---
    for metric in metrics:
        print(f"[PLOT] {metric}")

        plot_metric_lines(
            df, metric,
            out_path=os.path.join(args.out_dir, f"{metric}_line.png"),
            models=args.models,
            places=args.places,
            save_pdf=bool(args.save_pdf),
        )

        plot_radar_places_axes(
            df, metric,
            out_path=os.path.join(args.out_dir, f"{metric}_radar_axes_places.png"),
            models=args.models,
            places=args.places,
            radar_norm=args.radar_norm,
            save_pdf=bool(args.save_pdf),
        )

        plot_radar_models_axes(
            df, metric,
            out_path=os.path.join(args.out_dir, f"{metric}_radar_axes_models.png"),
            models=args.models,
            places=args.places,
            radar_norm=args.radar_norm,
            save_pdf=bool(args.save_pdf),
        )

    # --- Radar Type C: one radar over metrics (only the allowed metrics) ---
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
    )

    print(f"\n✓ Done. Plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()