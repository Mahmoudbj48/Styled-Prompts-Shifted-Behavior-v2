"""
Multi-style radar plot generator.

For each model produces two figures: Type A (metrics on spokes, one polygon
per style-place pair) and Type B (styles on spokes, one polygon per metric-place
pair), each with three sub-plots for baseline, median, and maximum strength.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation
import matplotlib.colors as mcolors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plots.plots import apply_neurips_style, fix_radar_ax

# ── Metric definitions ────────────────────────────────────────────────────────

# (candidate_columns, display_label, direction)
RADAR_METRICS: List[Tuple[List[str], str, str]] = [
    (["activation_similarity"],                                  "Cos.Sim",    "inv"),
    (["bleu"],                                                   "BLEU",       "inv"),
    (["delta_log_prob"],                                         "ΔLogProb",   "abs"),
    (["mirroring", "mirroring_verdict", "mirroring_rate_batch"], "MR",         "fwd"),
    (["cot_steps"],                                              "CoT Steps",  "abs"),
    (["unsafe_score", "asr"],                                    "ASR",        "fwd"),
]

# Maps metric display_label → axis/category name shown in the legend
METRIC_AXIS_NAME: Dict[str, str] = {
    "Cos.Sim":   "Activations G.",
    "BLEU":      "Generation Q.",
    "ΔLogProb":  "Confidence",
    "MR":        "Mirroring",
    "CoT Steps": "CoT",
    "ASR":       "Safety",
}

METRIC_SHORT: Dict[str, str] = {
    "Cos.Sim":   "Cos.Sim",
    "BLEU":      "BLEU",
    "ΔLogProb":  "ΔLogProb",
    "MR":        "MR",
    "CoT Steps": "CoT",
    "ASR":       "ASR",
}

# Known constant baseline values — only for metrics that are definitionally
# zero at baseline (differences/deltas).  Everything else is data-derived
# from the rows whose strength == STYLE_BASELINE[style].
BASELINE_CONST: Dict[str, float] = {
    "delta_log_prob": 0.0,
    "entropy_shift":  0.0,
}

# ── Style configuration ───────────────────────────────────────────────────────

STYLE_LEVELS: Dict[str, list] = {
    "spacing":         [0, 1, 5, 20, 50, 100],
    "punctuation":     [0, 1, 3, 5, 10, 20],
    "letter_case":     [0, 10, 25, 50, 75, 100],
    "politeness":      [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10],
    "inter_vs_imper":  ["interrogative", "imperative"],
    "length_variation":[0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
}

STYLE_BASELINE: Dict[str, object] = {
    "spacing":         0,
    "punctuation":     0,
    "letter_case":     0,
    "politeness":      0,
    "inter_vs_imper":  "interrogative",
    "length_variation":1.0,
}

STYLE_DISPLAY: Dict[str, str] = {
    "politeness":      "Polite.",
    "punctuation":     "Punct.",
    "spacing":         "Spacing",
    "letter_case":     "Casing",
    "length_variation":"Length",
    "inter_vs_imper":  "Form",
}

STYLE_COLORS: Dict[str, str] = {
    "politeness":      "#e41a1c",
    "punctuation":     "#377eb8",
    "spacing":         "#4daf4a",
    "letter_case":     "#984ea3",
    "length_variation":"#ff7f00",
    "inter_vs_imper":  "#a65628",
}

METRIC_COLORS = [
    "#e6194b",  # vivid red
    "#4363d8",  # vivid blue
    "#3cb44b",  # vivid green
    "#ffe119",  # vivid yellow
    "#911eb4",  # vivid purple
    "#42d4f4",  # vivid cyan
    "#f032e6",  # vivid magenta
    "#f58231",  # vivid orange
    "#469990",  # dark teal
    "#9a6324",  # dark gold
]

# Styles that only apply to global placement (no prefix/suffix variant)
GLOBAL_ONLY_STYLES = {"length_variation", "inter_vs_imper"}

# Per-label nudge in points (dx: positive=right, dy: positive=up)
LABEL_NUDGE: Dict[str, Tuple[float, float]] = {
    "Spacing": (12, -10),
    "Length":  (-12,  0),
    "Punct.":  (12,   0),
}

PLACE_LINESTYLE: Dict[str, str] = {"global": "-",  "prefix": "--", "suffix": ":"}
PLACE_LINEWIDTH: Dict[str, float] = {"global": 2.5, "prefix": 2.0, "suffix": 2.0}
PLACE_ALPHA:     Dict[str, float] = {"global": 0.10,"prefix": 0.06,"suffix": 0.06}

MODEL_NORMALIZE: Dict[str, str] = {
    "G-2B": "G-2B", "G-7B": "G-7B",
    "L3.2-3B": "L-3B", "L3.1-8B": "L-8B",
    "Q2.5-1.5B": "Q-1.5B", "Q2.5-7B": "Q-7B",
    "L-3B": "L-3B", "L-8B": "L-8B", "Q-1.5B": "Q-1.5B", "Q-7B": "Q-7B",
}


# ── Strength selection ────────────────────────────────────────────────────────

def _get_strengths(style: str) -> Tuple[str, str, str]:
    """
    Return (baseline_str, median_strength_str, max_strength_str) for a style.
    For politeness: use positive side only (median=6, max=10).
    For inter_vs_imper: median = max = 'imperative'.
    """
    base = STYLE_BASELINE[style]

    if style == "inter_vs_imper":
        return "interrogative", "imperative", "imperative"

    levels   = STYLE_LEVELS[style]
    non_base = [l for l in levels if l != base]

    if style == "politeness":
        non_base = sorted([l for l in non_base if l > 0])
    else:
        non_base = sorted(non_base, key=float)

    n        = len(non_base)
    median_s = non_base[n // 2]
    max_s    = non_base[-1]
    return str(base), str(median_s), str(max_s)


def _get_max_dev_strength(dev_df: pd.DataFrame, style: str, place: str, metric: str) -> str:
    """Return the strength with the highest absolute deviation from baseline for (style, place, metric)."""
    rows = dev_df[
        (dev_df["style"]  == style) &
        (dev_df["place"]  == place) &
        (dev_df["metric"] == metric)
    ]
    if rows.empty:
        return _get_strengths(style)[2]   # fallback to max strength
    idx = rows["deviation"].abs().idxmax()
    return str(rows.loc[idx, "strength"])


# ── Data loading ──────────────────────────────────────────────────────────────

def _preprocess_mirroring(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a numeric 'mirroring' column exists, converting from raw verdict columns if needed."""
    df = df.copy()
    if "mirroring" not in df.columns:
        if "mirroring_judge_raw" in df.columns:
            df["mirroring"] = (
                df["mirroring_judge_raw"].astype(str).str.strip().str.lower() == "yes"
            ).astype(float)
        elif "mirroring_verdict" in df.columns:
            df["mirroring"] = (
                df["mirroring_verdict"].astype(str).str.strip().str.upper() == "YES"
            ).astype(float)
    return df


_DIR_CSV_NAMES = (
    "results_cleaned.csv",
    "full_results_all_models.csv",
    "results_with_mirroring.csv",
    "combined_means_by_model_place_strength.csv",
    "summary.csv",
)


def _resolve_path(path: str) -> Optional[str]:
    """Resolve a path to a CSV file; if it's a directory, look for a known CSV inside."""
    path = path.strip()
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for fname in _DIR_CSV_NAMES:
            fpath = os.path.join(path, fname)
            if os.path.isfile(fpath):
                return fpath
        print(f"  [WARN] No CSV found in directory: {path}")
        return None
    print(f"  [WARN] Path not found: {path}")
    return None


def load_style_csvs(paths: List[str], style: str) -> pd.DataFrame:
    """Load + concatenate CSVs (or directories) for one style and tag each row."""
    parts = []
    for raw_path in paths:
        resolved = _resolve_path(raw_path)
        if resolved is None:
            continue
        df = pd.read_csv(resolved)
        df["style"] = style
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = _preprocess_mirroring(df)
    if "model" in df.columns:
        df["model"] = df["model"].map(lambda m: MODEL_NORMALIZE.get(str(m).strip(), str(m).strip()))
    if "problem_id" in df.columns and "prompt_id" not in df.columns:
        df = df.rename(columns={"problem_id": "prompt_id"})
    df["strength"] = df["strength"].astype(str)
    return df


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column name from candidates that exists in df, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ── Deviation table ───────────────────────────────────────────────────────────

def _transform(value: float, direction: str, baseline: float) -> float:
    """Apply a direction transform (inv/fwd/abs) to compute deviation from baseline."""
    if direction == "inv":
        return baseline - value
    elif direction == "fwd":
        return value - baseline
    else:  # "abs"
        return abs(value - baseline)


def build_deviation_table(
    all_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-(style, model, place, strength, metric) mean deviation from baseline.

    Returns a DataFrame with columns:
      style, model, place, strength, metric, deviation, norm
    where `norm` ∈ [0, 1] and higher = more perturbed.
    """
    rows = []
    for candidates, label, direction in RADAR_METRICS:
        col = _find_col(all_data, candidates)
        if col is None:
            continue

        known_base = next(
            (BASELINE_CONST[k] for k in BASELINE_CONST if k in col), None
        )

        for (style, model, place), grp in all_data.groupby(
                ["style", "model", "place"], sort=True):
            base_str = STYLE_BASELINE.get(style, 0)

            if known_base is not None:
                bval = known_base
            else:
                base_rows = grp[grp["strength"] == str(base_str)][col].dropna()
                if base_rows.empty:
                    continue
                bval = float(base_rows.mean())

            for strength_val, sgrp in grp.groupby("strength"):
                vals = sgrp[col].dropna()
                if vals.empty:
                    continue
                mean_val = float(vals.mean())
                dev = _transform(mean_val, direction, bval)
                rows.append({
                    "style":    style,
                    "model":    model,
                    "place":    place,
                    "strength": str(strength_val),
                    "metric":   label,
                    "deviation":dev,
                })

    if not rows:
        return pd.DataFrame()

    dev_df = pd.DataFrame(rows)

    # Normalise each metric to [0, 1] using global max deviation
    norm_vals = np.zeros(len(dev_df))
    for metric, grp in dev_df.groupby("metric"):
        max_dev = grp["deviation"].clip(lower=0).max()
        if max_dev > 0:
            norm_vals[grp.index] = grp["deviation"].clip(lower=0).values / max_dev
    dev_df["norm"] = norm_vals
    return dev_df


# ── Lookup helper ─────────────────────────────────────────────────────────────

def _lookup(dev_df: pd.DataFrame,
            style: str, place: str, strength: str, metric: str) -> float:
    """Look up the mean normalised deviation for a given (style, place, strength, metric) combination."""
    rows = dev_df[
        (dev_df["style"]    == style)  &
        (dev_df["place"]    == place)  &
        (dev_df["strength"] == str(strength)) &
        (dev_df["metric"]   == metric)
    ]
    return float(rows["norm"].mean()) if not rows.empty else 0.0


# ── Radar drawing ─────────────────────────────────────────────────────────────

def _draw_radar(ax, spoke_labels, polygons, title=""):
    """
    Draw a radar chart on polar axes `ax`.

    polygons: list of (values_array, color, linestyle, linewidth, fill_alpha)
    """
    N      = len(spoke_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    closed = angles + [angles[0]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(spoke_labels, size=20)
    ax.set_ylim(0, 1)
    ax.grid(color="grey", linestyle="--", linewidth=0.4, alpha=0.5)
    if title:
        ax.set_title(title, pad=14, fontsize=19, fontweight="bold")

    for vals, color, ls, lw, alpha in polygons:
        v_c = list(vals) + [vals[0]]
        ax.plot(closed, v_c, color=color, linestyle=ls, linewidth=lw, alpha=0.95)
        ax.fill(closed, v_c, color=color, alpha=alpha)
    fix_radar_ax(ax)
    fig = ax.get_figure()
    for label in ax.get_xticklabels():
        nudge = LABEL_NUDGE.get(label.get_text())
        if nudge:
            dx, dy = nudge
            label.set_transform(
                label.get_transform() + ScaledTranslation(dx / 72.0, dy / 72.0, fig.dpi_scale_trans)
            )


# ── Per-model figure builder ──────────────────────────────────────────────────

def _make_figure(
    model:         str,
    dev_df:        pd.DataFrame,
    all_styles:    List[str],
    all_places:    List[str],
    metric_labels: List[str],
    out_dir:       str,
    plot_type:     str,   # "A" or "B"
    save_pdf:      bool,
    pdf_only:      bool = False,
) -> None:
    """
    Produce and save a 3-panel radar figure (no/moderate/peak perturbation) for one model.

    plot_type 'A': metrics on spokes, one polygon per (style × place).
    plot_type 'B': styles on spokes,  one polygon per (metric × place).
    """
    apply_neurips_style()
    style_labels = [STYLE_DISPLAY.get(s, s) for s in all_styles]
    subplot_titles = ["(a) No Perturbation", "(b) Moderate Perturbation", "(c) Peak Perturbation"]
    col_indices    = [0, 1, 2]          # index into (base, median, max) tuple

    fig, axes = plt.subplots(
        1, 3,
        figsize=(20, 6.5),
        subplot_kw={"projection": "polar"},
    )
    type_desc = (
        "Behavioral Deviation per Style across Metrics"
        if plot_type == "A" else
        "Behavioral Deviation per Metric across Styles"
    )
    #make place capitalised for title

    _place_str = ", ".join(p.capitalize() for p in all_places) if len(all_places) == 1 else "All Placements"
    fig.suptitle(f"{type_desc}\n{model}  —  {_place_str} Placement", fontsize=21, fontweight="bold", y=1.01)

    legend_items: List[Tuple] = []

    for col_idx, (ax, sub_title) in enumerate(zip(axes, subplot_titles)):
        polygons = []

        if plot_type == "A":
            # Spokes = metrics; polygons = styles × places
            for s_idx, style in enumerate(all_styles):
                base_s, med_s, max_s = _get_strengths(style)
                strength    = [base_s, med_s, max_s][col_idx]
                color       = STYLE_COLORS.get(style, f"C{s_idx}")
                style_places = (
                    [p for p in all_places if p == "global"]
                    if style in GLOBAL_ONLY_STYLES else all_places
                )
                for place in style_places:
                    ls    = PLACE_LINESTYLE.get(place, "-")
                    lw    = PLACE_LINEWIDTH.get(place, 1.5)
                    alpha = PLACE_ALPHA.get(place, 0.07)
                    vals  = np.array([
                        _lookup(dev_df, style, place, strength, m)
                        for m in metric_labels
                    ])
                    polygons.append((vals, color, ls, lw, alpha))
                    if col_idx == 0:
                        legend_items.append((
                            f"{STYLE_DISPLAY.get(style, style)} / {place}",
                            color, ls, lw,
                        ))

        else:  # Type B — spokes = styles; polygons = metrics × places
            for m_idx, metric in enumerate(metric_labels):
                color = METRIC_COLORS[m_idx % len(METRIC_COLORS)]
                for place in all_places:
                    ls    = PLACE_LINESTYLE.get(place, "-")
                    lw    = PLACE_LINEWIDTH.get(place, 1.5)
                    alpha = PLACE_ALPHA.get(place, 0.07)
                    vals  = np.array([
                        _lookup(dev_df, style, place,
                                [_get_strengths(style)[0],
                                 _get_strengths(style)[1],
                                 _get_strengths(style)[2]][col_idx],
                                metric)
                        if place == "global" or style not in GLOBAL_ONLY_STYLES
                        else 0.0
                        for style in all_styles
                    ])
                    polygons.append((vals, color, ls, lw, alpha))
                    if col_idx == 0:
                        legend_items.append((
                            f"{metric} / {place}",
                            color, ls, lw,
                        ))

        spoke_labels = metric_labels if plot_type == "A" else style_labels
        _draw_radar(ax, spoke_labels, polygons, title=sub_title)

    # Legend outside right
    handles = [
        Line2D([0], [0], color=c, linestyle=ls, linewidth=lw, label=lbl)
        for lbl, c, ls, lw in legend_items
    ]
    leg_title = "Style / Place" if plot_type == "A" else "Metric / Place"
    fig.legend(
        handles=handles, title=leg_title,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=True, framealpha=0.9,
        fontsize=13, title_fontsize=14,
    )

    plt.tight_layout()
    tag        = "type_A_metrics_axes" if plot_type == "A" else "type_B_styles_axes"
    place_sfx  = "_global" if all_places == ["global"] else "_all_places"
    stem = os.path.join(out_dir, f"radar_{tag}{place_sfx}_{model.replace('-', '_')}")
    if not pdf_only:
        fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {stem}.png")


# ── Per-model figure: subplots = places, strength = max ───────────────────────

def _make_figure_by_place(
    model:         str,
    dev_df:        pd.DataFrame,
    all_styles:    List[str],
    all_places:    List[str],
    metric_labels: List[str],
    out_dir:       str,
    plot_type:     str,   # "C" (metrics on spokes) or "D" (styles on spokes)
    save_pdf:      bool,
    pdf_only:      bool = False,
) -> None:
    """
    3 subplots = 3 places, each showing MAX strength only.
    Type C: spokes = metrics, polygons = styles.
    Type D: spokes = styles,  polygons = metrics.
    """
    apply_neurips_style()
    style_labels = [STYLE_DISPLAY.get(s, s) for s in all_styles]
    n_subplots   = len(all_places)

    fig, axes = plt.subplots(
        1, n_subplots,
        figsize=(7 * n_subplots, 6.5),
        subplot_kw={"projection": "polar"},
    )
    if n_subplots == 1:
        axes = [axes]

    type_desc = (
        "Peak Perturbation Effects across Metrics by Placement"
        if plot_type == "C" else
        "Peak Perturbation Effects across Styles by Placement"
    )
    _place_str = ", ".join(p.capitalize() for p in all_places) if len(all_places) == 1 else "All Placements"
    if not (plot_type == "D" and len(all_places) == 1):
        fig.suptitle(f"{type_desc}\n{model}  —  {_place_str} Placement", fontsize=22, fontweight="bold", y=1.01)

    legend_items: List[Tuple] = []

    for ax, place in zip(axes, all_places):
        polygons = []

        if plot_type == "C":
            # Spokes = metrics; one polygon per style at max strength
            # Skip global-only styles for prefix/suffix subplots
            for s_idx, style in enumerate(all_styles):
                if place != "global" and style in GLOBAL_ONLY_STYLES:
                    continue
                _, _, max_s = _get_strengths(style)
                color = STYLE_COLORS.get(style, f"C{s_idx}")
                vals  = np.array([
                    _lookup(dev_df, style, place, max_s, m)
                    for m in metric_labels
                ])
                polygons.append((vals, color, "-", 2.5, 0.10))
                if place == all_places[0]:
                    legend_items.append((STYLE_DISPLAY.get(style, style), color, "-", 2.5))

        else:  # Type D — spokes = styles; one polygon per metric at max strength
            # For prefix/suffix: remove global-only styles from the spokes entirely
            if place == "global":
                active_styles = all_styles
            else:
                active_styles = [s for s in all_styles if s not in GLOBAL_ONLY_STYLES]
            active_style_labels = [STYLE_DISPLAY.get(s, s) for s in active_styles]

            for m_idx, metric in enumerate(metric_labels):
                color = METRIC_COLORS[m_idx % len(METRIC_COLORS)]
                vals  = np.array([
                    _lookup(dev_df, style, place, _get_max_dev_strength(dev_df, style, place, metric), metric)
                    for style in active_styles
                ])
                polygons.append((vals, color, "-", 2.5, 0.10))
                if place == all_places[0]:
                    axis_name = METRIC_AXIS_NAME.get(metric, metric)
                    legend_items.append((f"{axis_name} ({metric})", color, "-", 2.5))

        if plot_type == "C":
            spoke_labels = metric_labels
        elif place == "global":
            spoke_labels = [STYLE_DISPLAY.get(s, s) for s in all_styles]
        else:
            spoke_labels = active_style_labels
        place_titles = {"global": "Global Placement", "prefix": "Prefix Placement", "suffix": "Suffix Placement"}
        _sub_title = "" if len(all_places) == 1 else place_titles.get(place, place.capitalize())
        _draw_radar(ax, spoke_labels, polygons, title=_sub_title)

    leg_title = "Style" if plot_type == "C" else "Axes"
    if plot_type == "D":
        handles = [
            Patch(label=lbl,
                  edgecolor=c, linewidth=2,
                  facecolor=(*mcolors.to_rgb(c), 0.15))
            for lbl, c, ls, lw in legend_items
        ]
        fig.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            bbox_transform=fig.transFigure,
            borderaxespad=0,
            frameon=True, framealpha=0.9,
            fontsize=23,
            handletextpad=0.4,
            labelspacing=0.5,
        )
    else:
        handles = [
            Line2D([0], [0], color=c, linestyle=ls, linewidth=lw, label=lbl)
            for lbl, c, ls, lw in legend_items
        ]
        fig.legend(
            handles=handles, title=leg_title,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            bbox_transform=fig.transFigure,
            borderaxespad=0,
            frameon=True, framealpha=0.9,
            fontsize=22, title_fontsize=22,
        )

    plt.tight_layout()
    tag  = "type_C_metrics_axes" if plot_type == "C" else "type_D_styles_axes"
    stem = os.path.join(out_dir, f"radar_{tag}_by_place_{model.replace('-', '_')}")
    if not pdf_only:
        fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {stem}.png")


# ── Public entry point ────────────────────────────────────────────────────────

def make_multi_style_radar_plots(
    style_csvs:          Dict[str, List[str]],
    out_dir:             str,
    cot_style_dirs:      Optional[Dict[str, List[str]]] = None,
    asr_style_csvs:      Optional[Dict[str, List[str]]] = None,
    silhouette_style_csvs: Optional[Dict[str, List[str]]] = None,
    models:              Optional[List[str]] = None,
    places:              Optional[List[str]] = None,
    save_pdf:            bool = True,
    pdf_only:            bool = False,
) -> None:
    """
    Parameters
    ----------
    style_csvs : {style_name: [csv_path_or_dir, ...]}
        Per-style raw result CSVs (full_results_all_models.csv,
        results_with_mirroring.csv, etc.).
    cot_style_dirs : {style_name: [dir_or_csv, ...]}
        Per-style CoT run directories (containing results_cleaned.csv)
        or CSV paths with cot_correct / cot_steps columns.
    asr_style_csvs : {style_name: [csv_path_or_dir, ...]}
        Per-style ASR CSVs (combined_means_by_model_place_strength.csv or
        summary.csv files) with unsafe_score / asr columns.
    silhouette_style_csvs : {style_name: [csv_path_or_dir, ...]}
        Per-style silhouette CSVs (summary.csv files) with silhouette column.
    out_dir : str
        Directory where PNG (and optionally PDF) files are saved.
    models : list of str, optional
        Restrict to these (normalised) model names.
    places : list of str, optional
        Restrict to these placement positions.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load all data
    parts = []
    for style, paths in style_csvs.items():
        df = load_style_csvs(paths, style)
        if not df.empty:
            parts.append(df)

    if cot_style_dirs:
        print("[RADAR] Loading CoT data …")
        for style, paths in cot_style_dirs.items():
            df = load_style_csvs(paths, style)
            if not df.empty:
                parts.append(df)

    if asr_style_csvs:
        print("[RADAR] Loading ASR data …")
        for style, paths in asr_style_csvs.items():
            df = load_style_csvs(paths, style)
            if not df.empty:
                parts.append(df)

    if silhouette_style_csvs:
        print("[RADAR] Loading silhouette data …")
        for style, paths in silhouette_style_csvs.items():
            df = load_style_csvs(paths, style)
            if not df.empty:
                parts.append(df)
    if not parts:
        print("[ERROR] No data loaded — check paths.")
        return

    all_data = pd.concat(parts, ignore_index=True)

    required_cols = {"style", "model", "place", "strength"}
    missing = required_cols - set(all_data.columns)
    if missing:
        print(f"[ERROR] Missing columns after loading: {missing}")
        return

    if models:
        all_data = all_data[all_data["model"].isin(models)]
    if places:
        all_data = all_data[all_data["place"].isin(places)]
    if all_data.empty:
        print("[ERROR] No data after filtering.")
        return

    # Build deviation table
    print("[RADAR] Building deviation table …")
    dev_df = build_deviation_table(all_data)
    if dev_df.empty:
        print("[ERROR] Deviation table is empty — no matching metric columns found.")
        return

    all_models  = sorted(dev_df["model"].dropna().unique())
    all_places  = sorted(dev_df["place"].dropna().unique())
    all_styles  = [s for s in STYLE_DISPLAY if s in style_csvs]
    metric_labels = [
        lbl for _, lbl, _ in RADAR_METRICS
        if lbl in dev_df["metric"].unique()
    ]

    print(f"  Models : {all_models}")
    print(f"  Places : {all_places}")
    print(f"  Metrics: {metric_labels}")
    print(f"  Styles : {[STYLE_DISPLAY.get(s, s) for s in all_styles]}")

    place_variants = [all_places]
    if "global" in all_places and len(all_places) > 1:
        place_variants.append(["global"])

    for model in all_models:
        mdf = dev_df[dev_df["model"] == model]

        # Single global radar — Type D, global placement only
        _make_figure_by_place(
            model=model, dev_df=mdf,
            all_styles=all_styles, all_places=["global"],
            metric_labels=metric_labels,
            out_dir=out_dir, plot_type="D", save_pdf=save_pdf, pdf_only=pdf_only,
        )

    print(f"[DONE] Radar plots saved to: {out_dir}")
