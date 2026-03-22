"""
Individual publication-ready plots (no titles, big fonts, consistent sizes):
  1. BERTScore response – line per place  (politeness, TruthfulQA)
  2. Mirroring rate     – ridge plot       (politeness, TruthfulQA)
  3. ASR                – line per place  (letter_case, safety)
  4. Standalone model-legend image        (one horizontal line)

Usage:
    python Plots/plot_individual_figures.py
    python Plots/plot_individual_figures.py --out_dir results/individual_plots
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ── repo root on path ──────────────────────────────────────────────────────
_THIS = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS))
sys.path.insert(0, _ROOT)

from plots.plots import (
    apply_neurips_style,
    build_model_color_map,
    _kde_1d,
    _prepare_strength_axis,
    _filter_df,
    _sorted_strength_values,
    _aggregate_for_radar,
    _radar_setup,
    _set_rgrid,
    metric_display_name,
)

# ── Data paths ─────────────────────────────────────────────────────────────
POLITENESS_CSVS = [
    "results/politeness/run_multi_truthful_qa_20260221_112112/plots_metrics/combined_means_by_model_place_strength.csv",
    "results/politeness/run_multi_truthful_qa_20260221_145354/plots_metrics/combined_means_by_model_place_strength.csv",
    "results/politeness/run_multi_truthful_qa_20260222_113341/plots_metrics/combined_means_by_model_place_strength.csv",
    "results/politeness/run_multi_truthful_qa_20260222_132615/plots_metrics/combined_means_by_model_place_strength.csv",
    "results/politeness/run_multi_truthful_qa_20260222_150512/plots_metrics/combined_means_by_model_place_strength.csv",
    "results/politeness/run_multi_truthful_qa_20260223_120916/plots_metrics/combined_means_by_model_place_strength.csv",
]

LENGTH_VARIATION_CSVS = [
    "results/length_variation/run_multi_truthful_qa_20260304_055338/plots_metrics/combined_means_by_model_place_strength.csv",
]

COT_LENGTH_VARIATION_CSVS = [
    "results/cot_responses/run_gsm8k_length_variation_20260304_075450/plots_metrics/combined_means_by_model_place_strength.csv",
    "results/cot_responses/run_gsm8k_length_variation_20260304_103343/plots_metrics/combined_means_by_model_place_strength.csv",
    "results/cot_responses/run_gsm8k_length_variation_20260304_113317/plots_metrics/combined_means_by_model_place_strength.csv",
    "results/cot_responses/run_gsm8k_length_variation_20260304_132144/plots_metrics/combined_means_by_model_place_strength.csv",
    "results/cot_responses/run_gsm8k_length_variation_20260304_134359/plots_metrics/combined_means_by_model_place_strength.csv",
    "results/cot_responses/run_gsm8k_length_variation_20260304_154646/plots_metrics/combined_means_by_model_place_strength.csv",
]

LETTER_CASE_ASR_CSVS = [
    "results/safety/letter_case/L3.1-8B/combined_means_by_model_place_strength.csv",
    "results/safety/letter_case/L3.2-3B/combined_means_by_model_place_strength.csv",
    "results/safety/letter_case/Q2.5-7B/combined_means_by_model_place_strength.csv",
    "results/safety/letter_case/Q2.5-1.5B/combined_means_by_model_place_strength.csv",
    "results/safety/letter_case/G-7B/combined_means_by_model_place_strength.csv",
    "results/safety/letter_case/G-2B/combined_means_by_model_place_strength.csv",
]

# ── Shared style settings ──────────────────────────────────────────────────
FIG_W, FIG_H   = 12, 7
FONT_AXIS      = 34
FONT_TICK      = 30
FONT_LEGEND    = 28


def _apply_style():
    """Apply the NeurIPS rcParams and override with large-font publication settings."""
    apply_neurips_style()
    plt.rcParams.update({
        "figure.figsize":        (FIG_W, FIG_H),
        "font.size":             FONT_AXIS,
        "axes.labelsize":        FONT_AXIS,
        "xtick.labelsize":       FONT_TICK,
        "ytick.labelsize":       FONT_TICK,
        "legend.fontsize":       FONT_LEGEND,
        "lines.linewidth":       2.5,
        "lines.markersize":      8,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
    })


def _load_and_aggregate(csv_paths: list, metric: str) -> pd.DataFrame:
    """Load multiple CSVs, concatenate, aggregate by (model, place, strength)."""
    dfs = []
    for p in csv_paths:
        full = os.path.join(_ROOT, p)
        if not os.path.exists(full):
            print(f"[WARN] Missing: {full}")
            continue
        df = pd.read_csv(full)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No CSVs found.")
    combined = pd.concat(dfs, ignore_index=True)
    combined[metric] = pd.to_numeric(combined[metric], errors="coerce")
    combined["strength"] = pd.to_numeric(combined["strength"], errors="coerce")
    agg = (
        combined.dropna(subset=["model", "place", "strength", metric])
        .groupby(["model", "place", "strength"], as_index=False)[metric]
        .mean()
    )
    return agg


# ══════════════════════════════════════════════════════════════════════════
# 1. Line-per-place plot
# ══════════════════════════════════════════════════════════════════════════

def plot_line_per_place(df: pd.DataFrame, metric: str, out_dir: str, x_start=None, x_ticks=None, ylabel: str = None) -> None:
    """
    One PNG per place. Lines = models. No title. Big fonts.
    Legend is saved separately (see plot_legend_only).
    """
    _apply_style()

    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["strength", metric])
    if df.empty:
        return

    df, strengths_sorted, x_values, is_categorical = _prepare_strength_axis(df)

    models_u = sorted(df["model"].unique().tolist())
    places_u = sorted(df["place"].unique().tolist())
    model_colors = build_model_color_map(models_u)

    os.makedirs(out_dir, exist_ok=True)

    for place in places_u:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

        for model in models_u:
            sub = (
                df[(df["model"] == model) & (df["place"] == place)]
                .copy()
                .drop_duplicates(subset=["strength"])
                .set_index("strength")
                .reindex(strengths_sorted)
                .reset_index()
            )
            if sub.dropna(subset=[metric]).empty:
                continue
            ax.plot(
                x_values,
                sub[metric].values,
                marker="o",
                markersize=8,
                linewidth=2.5,
                color=model_colors[model],
                linestyle="-",
                alpha=0.95,
                label=str(model),
            )

        ax.set_xlabel("Style Strength", fontsize=FONT_AXIS)
        ax.set_ylabel(ylabel if ylabel is not None else metric_display_name(metric), fontsize=FONT_AXIS)

        if x_ticks is not None:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([str(t) for t in x_ticks], fontsize=FONT_TICK)
        elif not is_categorical:
            x0, x1 = ax.get_xlim()
            step = max(1, int(round((x1 - x0) / 5)))
            start = (int(round(x0)) // step) * step
            ticks = np.arange(start, int(round(x1)) + step, step, dtype=int)
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks], fontsize=FONT_TICK)
        else:
            ax.set_xticks(x_values)
            ax.set_xticklabels([str(s) for s in strengths_sorted], fontsize=FONT_TICK)

        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.set_title("")
        if x_start is not None:
            ax.set_xlim(left=x_start)

        ymin = float(df[df["place"] == place][metric].min())
        ymax = float(df[df["place"] == place][metric].max())
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
            pad = 0.12 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))
        ax.tick_params(axis="y", labelsize=FONT_TICK)

        # No legend on individual plots — saved separately
        ax.get_legend() and ax.get_legend().remove()

        plt.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.13)
        fname = f"{metric}_line_per_place__{place}.png"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close()
        print(f"  [SAVE] {out_path}")


# ══════════════════════════════════════════════════════════════════════════
# 2. Ridge plot
# ══════════════════════════════════════════════════════════════════════════

def plot_ridge(df: pd.DataFrame, metric: str, out_path: str) -> None:
    """Ridge plot – one ridge per strength value. No title. Big fonts."""
    _apply_style()

    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["strength", metric])
    if df.empty:
        return

    strengths = _sorted_strength_values(df["strength"].dropna().unique().tolist())
    if not strengths:
        return

    x_min = float(np.nanmin(df[metric].to_numpy(dtype=float)))
    x_max = float(np.nanmax(df[metric].to_numpy(dtype=float)))
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    x_grid = np.linspace(x_min, x_max, 300)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    y_ticks, y_labels = [], []
    for i, s in enumerate(strengths):
        vals = df.loc[df["strength"] == s, metric].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        dens = _kde_1d(vals, x_grid)
        if np.nanmax(dens) > 0:
            dens /= np.nanmax(dens) + 1e-12
        base_y = i * 1.0
        ax.fill_between(x_grid, base_y, base_y + dens * 0.85, alpha=0.35)
        ax.plot(x_grid, base_y + dens * 0.85, linewidth=1.5)
        y_ticks.append(base_y + 0.15)
        y_labels.append(str(s))

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=FONT_TICK)
    ax.set_xlabel(metric_display_name(metric), fontsize=FONT_AXIS)
    ax.set_ylabel("Style Strength", fontsize=FONT_AXIS)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.set_title("")

    plt.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.12)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] {out_path}")


# ══════════════════════════════════════════════════════════════════════════
# 3. Standalone legend image – models in one horizontal line
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# 4. Radar plot – axes=places, colors=models  (no title, big fonts)
# ══════════════════════════════════════════════════════════════════════════

def plot_radar_places(df: pd.DataFrame, metric: str, out_path: str,
                      style_name: str = None) -> None:
    """Radar Type A: axes = places, one line per model. No title. Big fonts."""
    _apply_style()

    df = df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])
    if df.empty or metric not in df.columns:
        return

    agg = _aggregate_for_radar(df, metric, style_name=style_name)
    if agg.empty:
        return

    models_u = sorted(agg["model"].unique().tolist())
    places_u = sorted(agg["place"].unique().tolist())

    tab = (
        agg.pivot(index="model", columns="place", values=metric)
        .reindex(index=models_u, columns=places_u)
    )
    arr = tab.to_numpy(dtype=float)
    finite_vals = arr[np.isfinite(arr)]
    vmin = float(finite_vals.min()) if finite_vals.size > 0 else 0.0
    vmax = float(finite_vals.max()) if finite_vals.size > 0 else 1.0

    fig, ax = plt.subplots(figsize=(FIG_H, FIG_H), subplot_kw=dict(polar=True))
    ax.set_position([0.1, 0.05, 0.85, 0.90])

    # spoke labels – big font, no title; nudge prefix/suffix in display space
    from matplotlib.transforms import ScaledTranslation
    angles = _radar_setup(ax, places_u, title=None)
    ax.set_xticklabels([p.title() for p in places_u], fontsize=FONT_AXIS)

    _set_rgrid(ax, vmin, vmax)
    ax.set_yticklabels(
        [f"{r:.2f}" for r in np.linspace(0, vmax - vmin, 6)[1:] + vmin],
        fontsize=FONT_TICK,
    )
    # Place ring numbers at midpoint between global (0°) and suffix (240°)
    ax.set_rlabel_position(320)

    model_colors = build_model_color_map(models_u)
    for m in models_u:
        row = tab.loc[m].to_numpy(dtype=float)
        if np.all(np.isnan(row)):
            continue
        vals = (row - vmin).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=model_colors[m], linewidth=3.0)
        ax.fill(angles, vals, color=model_colors[m], alpha=0.12)

    # Nudge specific spoke labels in display space (points)
    _NUDGE = {"suffix": (-8, -8), "prefix": (8, -8)}
    fig.canvas.draw()
    for label in ax.get_xticklabels():
        key = label.get_text().lower()
        if key in _NUDGE:
            dx_pt, dy_pt = _NUDGE[key]
            offset = ScaledTranslation(dx_pt / 72, dy_pt / 72, fig.dpi_scale_trans)
            label.set_transform(label.get_transform() + offset)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] {out_path}")


def plot_legend_only(models: list, out_path: str) -> None:
    """
    Save a legend-only PNG with all models in one horizontal line.
    No axes, no data – just the coloured handles + labels.
    """
    _apply_style()
    model_colors = build_model_color_map(models)

    handles = [
        mlines.Line2D(
            [], [],
            color=model_colors[m],
            marker="o",
            markersize=14,
            linewidth=3.0,
            label=m,
        )
        for m in sorted(models)
    ]

    n = len(handles)
    fig_w = max(4, n * 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, 0.9))
    ax.set_axis_off()

    leg = ax.legend(
        handles=handles,
        loc="center",
        ncol=n,
        frameon=True,
        framealpha=0.95,
        fontsize=FONT_LEGEND,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.2,
        borderpad=0.3,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] {out_path}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    """Parse CLI args and generate all publication-ready individual figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default=os.path.join(_ROOT, "results", "individual_plots"),
    )
    args = parser.parse_args()
    out = args.out_dir

    # ── 1. Politeness – BERTScore response (line per place) ───────────────
    print("\n[1] BERTScore response – line per place (politeness)")
    df_pol_bsr = _load_and_aggregate(POLITENESS_CSVS, "bertscore_response")
    plot_line_per_place(
        df_pol_bsr, "bertscore_response",
        os.path.join(out, "politeness_bertscore_response"),
        x_ticks=np.arange(-10, 11, 2),
    )

    # ── 2. Politeness – Mirroring rate (ridge) ────────────────────────────
    print("\n[2] Mirroring rate – ridge plot (politeness)")
    df_pol_mir = _load_and_aggregate(POLITENESS_CSVS, "mirroring_rate")
    plot_ridge(
        df_pol_mir, "mirroring_rate",
        os.path.join(out, "politeness_mirroring_rate", "mirroring_rate_ridge.png"),
    )

    # ── 3. Length variation – Activation similarity (line per place) ─────
    print("\n[3] Activation similarity – line per place (length_variation)")
    df_lv_act = _load_and_aggregate(LENGTH_VARIATION_CSVS, "activation_similarity")
    plot_line_per_place(
        df_lv_act, "activation_similarity",
        os.path.join(out, "length_variation_activation_similarity"),
    )

    # ── 4. Letter-case – ASR (line per place) ─────────────────────────────
    print("\n[4] ASR – line per place (letter_case)")
    df_lc_asr = _load_and_aggregate(LETTER_CASE_ASR_CSVS, "asr")
    plot_line_per_place(
        df_lc_asr, "asr",
        os.path.join(out, "letter_case_asr"),
        x_start=0,
    )

    # ── 5. CoT steps – line per place (length_variation) ──────────────────
    print("\n[5] CoT steps – line per place (length_variation)")
    df_cot_lv = _load_and_aggregate(COT_LENGTH_VARIATION_CSVS, "cot_steps")
    plot_line_per_place(
        df_cot_lv, "cot_steps",
        os.path.join(out, "cot_length_variation_steps"),
        ylabel="CoT Steps",
        x_ticks=np.arange(0.5, 3.5, 0.5),
    )

    # ── 6. Politeness – delta_log_prob radar (axes=places) ────────────────
    print("\n[6] Delta log prob – radar axes=places (politeness)")
    df_pol_dlp = _load_and_aggregate(POLITENESS_CSVS, "delta_log_prob")
    plot_radar_places(
        df_pol_dlp, "delta_log_prob",
        os.path.join(out, "politeness_delta_log_prob_radar_places.png"),
        style_name="politeness",
    )

    # ── 7. Standalone legend ──────────────────────────────────────────────
    print("\n[7] Standalone model legend")
    all_models = sorted(
        set(df_pol_bsr["model"].tolist())
        | set(df_lc_asr["model"].tolist())
        | set(df_lv_act["model"].tolist())
        | set(df_cot_lv["model"].tolist())
    )
    plot_legend_only(
        all_models,
        os.path.join(out, "legend_models.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
