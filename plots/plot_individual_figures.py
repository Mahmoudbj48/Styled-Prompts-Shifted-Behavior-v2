"""
SGS heatmap — publication-ready figure.

Usage:
    python plots/plot_individual_figures.py
    python plots/plot_individual_figures.py --out_dir results/individual_plots
"""

import argparse
import os
import re as _re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── repo root on path ──────────────────────────────────────────────────────
_THIS = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS))
sys.path.insert(0, _ROOT)

from plots.plots import apply_neurips_style


def _apply_style():
    """Apply the NeurIPS rcParams."""
    apply_neurips_style()


def _extract_row_vals(parts: list) -> list:
    """Extract 6 numeric values from a split LaTeX table row, ignoring text/ding cells."""
    vals = []
    for p in parts:
        p = p.strip()
        # bare number (dataset rows): starts with digit or is \ding
        if _re.match(r"^[\d]", p):
            clean = p.split("$")[0].strip()
            try:
                vals.append(float(clean))
            except ValueError:
                vals.append(float("nan"))
        # \textbf{<digit>...} (Total SGS rows)
        elif _re.search(r"\\textbf\{[\d.]", p):
            inner = _re.sub(r"\\textbf\{([^}]+)\}", r"\1", p).strip()
            clean = inner.split("$")[0].strip()
            try:
                vals.append(float(clean))
            except ValueError:
                vals.append(float("nan"))
        # \ding{55} = not applicable → NaN
        elif r"\ding" in p:
            vals.append(float("nan"))
    return vals


def _parse_sgs_tex(tex_path: str):
    """
    Parse sgs_table.tex and return (model_labels, data_matrix) for Total SGS rows only.
    One row per model (the bold \\rowcolor summary line), 6 metric columns.
    """
    with open(tex_path, encoding="utf-8") as f:
        text = f.read()

    model_labels, rows = [], []
    current_model = None

    for line in text.split("\n"):
        s = line.strip()
        m = _re.match(r"\\multirow\{\d+\}\{\*\}\{\\textbf\{(.+?)\}\}", s)
        if m:
            current_model = m.group(1)
            continue
        if "\\rowcolor" in s and current_model is not None:
            parts = s.rstrip("\\").split("&")
            vals = _extract_row_vals(parts)
            if len(vals) == 6:
                model_labels.append(current_model)
                rows.append(vals)

    return model_labels, (np.array(rows) if rows else np.empty((0, 6)))


def _parse_sgs_tex_full(tex_path: str):
    """
    Parse sgs_table.tex and return per-dataset SGS values.

    Returns
    -------
    model_labels  : list of model name strings
    dataset_labels: list of dataset name strings (cleaned of citations)
    data          : ndarray of shape (n_models, n_datasets, 6)
    """
    with open(tex_path, encoding="utf-8") as f:
        text = f.read()

    # strip \cite{...} for clean dataset labels
    def _clean_ds(s):
        return _re.sub(r"\\cite\{[^}]+\}", "", s).strip()

    model_labels = []
    dataset_labels_ordered = []
    # records: list of (model, dataset, [6 vals])
    records = []
    current_model = None

    for line in text.split("\n"):
        s = line.strip()
        # model heading
        m = _re.match(r"\\multirow\{\d+\}\{\*\}\{\\textbf\{(.+?)\}\}", s)
        if m:
            current_model = m.group(1)
            if current_model not in model_labels:
                model_labels.append(current_model)
            continue
        # skip Total SGS row
        if "\\rowcolor" in s:
            continue
        # dataset row: starts with & <dataset label> & ...
        if s.startswith("&") and current_model is not None:
            parts = s.rstrip("\\").split("&")
            if len(parts) < 3:
                continue
            ds_raw = parts[1].strip()
            ds = _clean_ds(ds_raw)
            if not ds:
                continue
            vals = _extract_row_vals(parts[2:])
            if len(vals) == 6:
                if ds not in dataset_labels_ordered:
                    dataset_labels_ordered.append(ds)
                records.append((current_model, ds, vals))

    n_models = len(model_labels)
    n_ds = len(dataset_labels_ordered)
    data = np.full((n_models, n_ds, 6), np.nan)
    for model, ds, vals in records:
        mi = model_labels.index(model)
        di = dataset_labels_ordered.index(ds)
        data[mi, di, :] = vals

    return model_labels, dataset_labels_ordered, data


def plot_sgs_heatmap(tex_path: str, out_path: str) -> None:
    """
    Plot a Total SGS heatmap (Oranges colormap) from sgs_table.tex.

    Rows = models (one row per model, Total SGS only), columns = 6 metrics + Mean.
    No per-dataset rows. No colored separators between models.
    """
    _apply_style()

    model_labels, data = _parse_sgs_tex(tex_path)
    if data.size == 0:
        print(f"  [SKIP] No data parsed from {tex_path}")
        return

    METRIC_LABELS = ["Δ-Cos", "Δ-BLEU", "Δ-BERT", "Δ-Prob", "Δ-Ent", "Δ-MR"]

    data_plot = data
    col_labels = METRIC_LABELS
    n_rows, n_cols = data_plot.shape

    # per-column min/max for independent color scales
    col_vmin = np.nanmin(data_plot, axis=0)
    col_vmax = np.nanmax(data_plot, axis=0)
    for j in range(n_cols):
        if col_vmin[j] == col_vmax[j]:
            col_vmax[j] = col_vmin[j] + 1e-6

    cmap = plt.get_cmap("Oranges")

    fig_w = max(10, n_cols * 1.8 + 3.0)
    fig_h = max(5, n_rows * 0.65 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor("gray")

    # draw each cell as a colored rectangle, normalized per column
    from matplotlib.patches import Rectangle
    for i in range(n_rows):
        for j in range(n_cols):
            v = data_plot[i, j]
            if np.isnan(v):
                color = "#f5f5f5"
                norm_v = 0.0
            else:
                norm_v = (v - col_vmin[j]) / (col_vmax[j] - col_vmin[j])
                color = cmap(0.1 + 0.85 * norm_v)  # avoid pure white at 0
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="gray", linewidth=0.5)
            ax.add_patch(rect)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    # Metric column labels on top
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=16, fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", which="both", length=0)

    # Metric family brackets on top
    metric_families = [
        ("Activation\nGeometry", 0, 0),  # Δ-Cos
        ("Generation Q.", 1, 2),          # Δ-BLEU, Δ-BERT
        ("Confidence", 3, 4),             # Δ-Prob, Δ-Ent
        ("Mirroring", 5, 5),              # Δ-MR
    ]
    y_brack = -1.1
    for mf_name, col_start, col_end in metric_families:
        mid = (col_start + col_end) / 2.0
        ax.plot(
            [col_start - 0.4, col_end + 0.4], [y_brack, y_brack],
            color="black", lw=1.5, clip_on=False,
        )
        # small ticks at left and right of bracket
        ax.plot([col_start - 0.4, col_start - 0.4], [y_brack, y_brack + 0.1],
                color="black", lw=1.5, clip_on=False)
        ax.plot([col_end + 0.4, col_end + 0.4], [y_brack, y_brack + 0.1],
                color="black", lw=1.5, clip_on=False)
        # family name
        ax.text(
            mid, y_brack - 0.15, mf_name,
            ha="center", va="bottom",
            fontsize=12, fontweight="bold", fontstyle="italic",
            clip_on=False,
        )

    # Model labels on y-axis
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(model_labels, fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", which="both", length=0)

    # Family brackets on the left side
    def _get_family(name):
        if name.startswith("L"):
            return "Llama"
        elif name.startswith("G"):
            return "Gemma"
        elif name.startswith("Q"):
            return "Qwen"
        return "Other"

    # group consecutive models by family
    families = []
    cur_fam, cur_start = _get_family(model_labels[0]), 0
    for idx in range(1, n_rows):
        fam = _get_family(model_labels[idx])
        if fam != cur_fam:
            families.append((cur_fam, cur_start, idx - 1))
            cur_fam, cur_start = fam, idx
    families.append((cur_fam, cur_start, n_rows - 1))

    for fam_name, row_start, row_end in families:
        mid = (row_start + row_end) / 2.0
        # draw bracket line
        x_brack = -1.1
        ax.plot(
            [x_brack, x_brack], [row_start - 0.4, row_end + 0.4],
            color="black", lw=1.5, clip_on=False,
        )
        # small ticks at top and bottom of bracket
        ax.plot([x_brack, x_brack + 0.1], [row_start - 0.4, row_start - 0.4],
                color="black", lw=1.5, clip_on=False)
        ax.plot([x_brack, x_brack + 0.1], [row_end + 0.4, row_end + 0.4],
                color="black", lw=1.5, clip_on=False)
        # family name
        ax.text(
            x_brack - 0.15, mid, fam_name,
            ha="right", va="center",
            fontsize=13, fontweight="bold", fontstyle="italic",
            rotation=90, clip_on=False,
        )

    # Cell value annotations
    for i in range(n_rows):
        for j in range(n_cols):
            v = data_plot[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=12, color="#aaa")
            else:
                norm_v = (v - col_vmin[j]) / (col_vmax[j] - col_vmin[j])
                text_color = "white" if norm_v > 0.55 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=12, color=text_color)

    # colorbar showing normalized scale (0 = column min, 1 = column max)
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Best\ngeneralization", "Worst\ngeneralization"])
    cbar.ax.tick_params(labelsize=14)

    plt.tight_layout()

    # place "SGS" manually between the heatmap and the colorbar (after layout)
    cbar_pos = cbar.ax.get_position()
    ax_pos = ax.get_position()
    x_mid = (ax_pos.x1 + cbar_pos.x0) / 2
    fig.text(
        x_mid, 0.5, "Total $SGS_X$",
        ha="center", va="center",
        fontsize=16, fontweight="bold", rotation=90,
    )

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] {out_path}")


def plot_sgs_barplot(tex_path: str, out_path: str) -> None:
    """
    Grouped bar chart: x = models, one bar group per metric.
    Bar height = mean SGS across datasets; error bar = std across datasets.
    One subplot per metric (2 rows × 3 cols).
    """
    _apply_style()

    model_labels, dataset_labels, data = _parse_sgs_tex_full(tex_path)
    if data.size == 0:
        print(f"  [SKIP] No data parsed from {tex_path}")
        return

    METRIC_LABELS = ["Δ-Cos", "Δ-BLEU", "Δ-BERT", "Δ-Prob", "Δ-Ent", "Δ-MR"]
    n_models = len(model_labels)
    x = np.arange(n_models)

    # mean and std per (model, metric) across datasets
    means = np.nanmean(data, axis=1)   # (n_models, 6)
    stds  = np.nanstd(data, axis=1, ddof=1)   # (n_models, 6)

    cmap = plt.get_cmap("Oranges")
    colors = [cmap(0.35 + 0.5 * i / max(n_models - 1, 1)) for i in range(n_models)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=False)
    axes = axes.flatten()

    for k, (metric_label, ax) in enumerate(zip(METRIC_LABELS, axes)):
        ax.grid(False)
        bars = ax.bar(
            x,
            means[:, k],
            yerr=stds[:, k],
            color=colors,
            edgecolor="gray",
            linewidth=0.8,
            capsize=4,
            error_kw=dict(elinewidth=1.2, ecolor="black"),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=12, fontweight="bold", rotation=30, ha="right")
        ax.set_ylabel("SGS", fontsize=13)
        ax.set_title(metric_label, fontsize=14, fontweight="bold", pad=6)
        ax.tick_params(axis="y", labelsize=11)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    plt.suptitle("SGS per metric — mean ± std across datasets", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] {out_path}")


def _build_sgs_total_data():
    """
    Return hardcoded Total SGS values and per-dataset values from the paper tables.

    Returns (model_names, family_map, metric_labels, totals_dict, per_dataset_dict)
    where totals_dict[model] = [val_per_metric] (NaN if unavailable)
    and   per_dataset_dict[model] = list of [6 vals] per dataset (for std computation).
    """
    METRICS = ["Δ-Cos", "Δ-BLEU", "Δ-BERT", "Δ-Prob", "Δ-Ent", "Δ-MR"]
    nan = float("nan")

    # ── open-source totals ──
    open_totals = {
        "L-3B":    [0.041, 0.147, 0.082, 0.017, 0.023, 0.089],
        "L-8B":    [0.050, 0.139, 0.076, 0.014, 0.023, 0.089],
        "G-2B":    [0.026, 0.160, 0.092, 0.024, 0.016, 0.081],
        "G-7B":    [0.026, 0.150, 0.085, 0.026, 0.018, 0.085],
        "G4-E4B":  [0.032, 0.189, 0.146, 0.023, 0.024, 0.087],
        "Q-1.5B":  [0.022, 0.170, 0.098, 0.014, 0.015, 0.084],
        "Q-7B":    [0.044, 0.150, 0.081, 0.026, 0.012, 0.098],
        "Q3.5-9B": [0.016, 0.120, 0.068, 0.028, 0.012, 0.117],
    }
    # ── per-dataset values (6 datasets × 6 metrics) for open-source ──
    open_datasets = {
        "L-3B": [
            [0.102, 0.158, 0.083, 0.017, 0.014, 0.042],
            [0.030, 0.156, 0.083, 0.014, 0.014, 0.070],
            [0.033, 0.146, 0.086, 0.021, 0.024, 0.101],
            [0.024, 0.128, 0.070, 0.015, 0.027, 0.098],
            [0.025, 0.144, 0.084, 0.016, 0.027, 0.116],
            [0.030, 0.150, 0.084, 0.018, 0.030, 0.107],
        ],
        "L-8B": [
            [0.097, 0.149, 0.072, 0.013, 0.014, 0.040],
            [0.096, 0.137, 0.068, 0.011, 0.023, 0.069],
            [0.032, 0.141, 0.082, 0.021, 0.022, 0.096],
            [0.022, 0.126, 0.067, 0.011, 0.020, 0.106],
            [0.026, 0.139, 0.083, 0.012, 0.028, 0.116],
            [0.028, 0.145, 0.082, 0.014, 0.031, 0.107],
        ],
        "G-2B": [
            [0.058, 0.174, 0.096, 0.020, 0.016, 0.052],
            [0.023, 0.169, 0.096, 0.021, 0.013, 0.050],
            [0.020, 0.151, 0.095, 0.029, 0.015, 0.105],
            [0.014, 0.146, 0.078, 0.020, 0.014, 0.084],
            [0.023, 0.155, 0.092, 0.026, 0.016, 0.102],
            [0.020, 0.164, 0.094, 0.027, 0.022, 0.091],
        ],
        "G-7B": [
            [0.070, 0.160, 0.081, 0.020, 0.022, 0.045],
            [0.047, 0.162, 0.082, 0.020, 0.020, 0.051],
            [0.004, 0.147, 0.092, 0.029, 0.012, 0.113],
            [0.008, 0.133, 0.074, 0.028, 0.015, 0.091],
            [0.009, 0.141, 0.085, 0.022, 0.017, 0.110],
            [0.014, 0.154, 0.092, 0.034, 0.020, 0.099],
        ],
        "G4-E4B": [
            [0.029, 0.193, 0.147, 0.022, 0.024, 0.085],
            [0.033, 0.188, 0.152, 0.022, 0.034, 0.069],
            [0.039, 0.181, 0.141, 0.029, 0.026, 0.096],
            [0.023, 0.190, 0.143, 0.017, 0.020, 0.084],
            [0.035, 0.186, 0.144, 0.026, 0.020, 0.100],
            [0.032, 0.194, 0.148, 0.020, 0.018, 0.087],
        ],
        "Q-1.5B": [
            [0.095, 0.176, 0.093, 0.011, 0.009, 0.044],
            [0.009, 0.181, 0.102, 0.010, 0.011, 0.041],
            [0.010, 0.154, 0.096, 0.018, 0.020, 0.123],
            [0.005, 0.169, 0.098, 0.013, 0.015, 0.095],
            [0.008, 0.161, 0.095, 0.014, 0.018, 0.105],
            [0.006, 0.180, 0.105, 0.017, 0.017, 0.097],
        ],
        "Q-7B": [
            [0.096, 0.158, 0.078, 0.022, 0.011, 0.075],
            [0.093, 0.154, 0.074, 0.021, 0.011, 0.066],
            [0.020, 0.142, 0.088, 0.033, 0.012, 0.132],
            [0.013, 0.141, 0.077, 0.019, 0.011, 0.094],
            [0.018, 0.142, 0.080, 0.029, 0.013, 0.117],
            [0.021, 0.162, 0.090, 0.034, 0.013, 0.103],
        ],
        "Q3.5-9B": [
            [0.014, 0.120, 0.066, 0.025, 0.010, 0.115],
            [0.014, 0.126, 0.068, 0.033, 0.012, 0.126],
            [0.015, 0.122, 0.076, 0.032, 0.017, 0.128],
            [0.020, 0.110, 0.059, 0.023, 0.009, 0.109],
            [0.017, 0.122, 0.073, 0.026, 0.012, 0.118],
            [0.018, 0.119, 0.066, 0.029, 0.015, 0.103],
        ],
    }
    # ── closed-source totals & per-dataset (3 datasets) ──
    closed_totals = {
        "GPT-5.4":           [nan, 0.098, 0.023, 0.033, 0.063, 0.003],
        "Gemini-2.5-Flash":  [nan, 0.063, 0.046, nan,   nan,   0.001],
        "Claude-Sonnet-4.6": [nan, 0.118, 0.043, nan,   nan,   0.004],
    }
    closed_datasets = {
        "GPT-5.4": [
            [nan, 0.086, 0.023, 0.034, 0.094, 0.003],
            [nan, 0.107, 0.028, 0.035, 0.061, 0.004],
            [nan, 0.102, 0.019, 0.031, 0.034, 0.003],
        ],
        "Gemini-2.5-Flash": [
            [nan, 0.069, 0.046, nan, nan, 0.000],
            [nan, 0.060, 0.045, nan, nan, 0.004],
            [nan, 0.060, 0.046, nan, nan, 0.000],
        ],
        "Claude-Sonnet-4.6": [
            [nan, 0.110, 0.057, nan, nan, 0.000],
            [nan, 0.131, 0.044, nan, nan, 0.013],
            [nan, 0.114, 0.028, nan, nan, 0.000],
        ],
    }

    # family assignment
    family_map = {}
    for m in open_totals:
        if m.startswith("L"):
            family_map[m] = "Llama"
        elif m.startswith("G"):
            family_map[m] = "Gemma"
        elif m.startswith("Q"):
            family_map[m] = "Qwen"
    for m in closed_totals:
        family_map[m] = "Closed"

    all_totals = {**open_totals, **closed_totals}
    all_datasets = {**open_datasets, **closed_datasets}
    return list(all_totals.keys()), family_map, METRICS, all_totals, all_datasets


def plot_sgs_sorted_bars(out_path: str, include_closed: bool = True) -> None:
    """
    Bar plot: x = metrics, y = Total SGS, one bar per model per metric,
    sorted smallest→largest within each metric.  Family-based colour shading.

    Parameters
    ----------
    out_path : str
        Output PNG path.
    include_closed : bool
        If False, drop the closed-source models.
    """
    _apply_style()

    all_names, family_map, metrics, totals, per_ds = _build_sgs_total_data()

    # filter
    if not include_closed:
        names = [n for n in all_names if family_map[n] != "Closed"]
    else:
        names = list(all_names)

    # compute std across datasets for each (model, metric)
    model_std = {}
    for n in names:
        ds_arr = np.array(per_ds[n])  # (n_datasets, 6)
        model_std[n] = np.nanstd(ds_arr, axis=0, ddof=1)  # (6,)

    # ── family colours (base hue) with shades per member ──
    family_bases = {
        "Llama":  (0.85, 0.33, 0.24),   # red-ish
        "Gemma":  (0.20, 0.63, 0.17),   # green
        "Qwen":   (0.22, 0.46, 0.87),   # blue
        "Closed": (0.55, 0.34, 0.70),   # purple
    }

    def _shade(base_rgb, idx, total):
        """Lighten/darken base colour: idx=0 lightest, idx=total-1 darkest."""
        if total <= 1:
            return base_rgb
        t = 0.35 + 0.65 * idx / (total - 1)          # 0.35 … 1.0
        return tuple(b * t + (1 - t) * 1.0 for b in base_rgb)   # lerp to white

    # count members per family among active names
    from collections import Counter
    fam_counts = Counter(family_map[n] for n in names)
    fam_seen = {f: 0 for f in fam_counts}

    model_color = {}
    for n in names:
        f = family_map[n]
        model_color[n] = _shade(family_bases[f], fam_seen[f], fam_counts[f])
        fam_seen[f] += 1

    n_metrics = len(metrics)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.grid(False)

    bar_width = 0.7 / max(len(names), 1)
    group_gap = 1.0      # spacing between metric groups

    xtick_positions = []
    legend_handles = {}

    for mi, metric in enumerate(metrics):
        # gather (model, value, std) triples that are not NaN
        triples = []
        for n in names:
            v = totals[n][mi]
            if not np.isnan(v):
                s = model_std[n][mi] if not np.isnan(model_std[n][mi]) else 0.0
                triples.append((n, v, s))
        # sort smallest → largest
        triples.sort(key=lambda p: p[1])

        group_left = mi * (len(names) + 2) * bar_width * group_gap
        for bi, (model_name, val, std) in enumerate(triples):
            xpos = group_left + bi * bar_width
            bar = ax.bar(
                xpos, val, width=bar_width * 0.9,
                yerr=std,
                color=model_color[model_name],
                edgecolor="gray", linewidth=0.6,
                capsize=2,
                error_kw=dict(elinewidth=0.8, ecolor="black"),
            )
            if model_name not in legend_handles:
                legend_handles[model_name] = bar[0]
        # tick at centre of group
        centre = group_left + (len(triples) - 1) * bar_width / 2 if triples else group_left
        xtick_positions.append((centre, metric))

    ax.set_xticks([p for p, _ in xtick_positions])
    ax.set_xticklabels([l for _, l in xtick_positions], fontsize=13, fontweight="bold")
    ax.set_ylabel("Total SGS", fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Metric axis family brackets below the x-axis
    # metrics order: 0=Δ-Cos, 1=Δ-BLEU, 2=Δ-BERT, 3=Δ-Prob, 4=Δ-Ent, 5=Δ-MR
    metric_axes = [
        ("Activation\nGeometry", [0]),        # Δ-Cos
        ("Generation Q.", [1, 2]),             # Δ-BLEU, Δ-BERT
        ("Confidence", [3, 4]),                # Δ-Prob, Δ-Ent
        ("Mirroring", [5]),                    # Δ-MR
    ]
    # get the x positions of metric group centres
    tick_xs = [p for p, _ in xtick_positions]
    y_base = ax.get_ylim()[0]
    # use axes transform for y offset below the axis
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)

    for axis_name, metric_indices in metric_axes:
        x_left = tick_xs[metric_indices[0]]
        x_right = tick_xs[metric_indices[-1]]
        # half-width of one metric group for bracket extent
        half_group = (len(names) - 1) * bar_width / 2 + bar_width
        x_start = x_left - half_group
        x_end = x_right + half_group
        x_mid = (x_start + x_end) / 2

        # bracket line
        y_brack = -0.12
        ax.plot([x_start, x_end], [y_brack, y_brack],
                color="black", lw=1.5, clip_on=False, transform=trans)
        ax.plot([x_start, x_start], [y_brack, y_brack + 0.02],
                color="black", lw=1.5, clip_on=False, transform=trans)
        ax.plot([x_end, x_end], [y_brack, y_brack + 0.02],
                color="black", lw=1.5, clip_on=False, transform=trans)
        # axis family label
        ax.text(x_mid, y_brack - 0.02, axis_name,
                ha="center", va="top",
                fontsize=11, fontweight="bold", fontstyle="italic",
                clip_on=False, transform=trans)

    # legend — vertical list grouped by family, placed to the right of the plot
    from matplotlib.patches import Patch
    family_order = ["Llama", "Gemma", "Qwen"] + (["Closed"] if include_closed else [])

    ordered_handles, ordered_labels = [], []
    for fi, fam in enumerate(family_order):
        if fi > 0:
            # spacer between families
            ordered_handles.append(Patch(facecolor="none", edgecolor="none"))
            ordered_labels.append("")
        # family header (bold, no patch)
        ordered_handles.append(Patch(facecolor="none", edgecolor="none"))
        ordered_labels.append(f"$\\bf{{{fam}}}$")
        # models in this family
        for n in names:
            if family_map[n] == fam and n in legend_handles:
                ordered_handles.append(legend_handles[n])
                ordered_labels.append(n)

    ax.legend(
        ordered_handles, ordered_labels,
        fontsize=9, ncol=1,
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        framealpha=0.95, borderaxespad=0,
        handletextpad=0.5, handlelength=1.2,
        labelspacing=0.3,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] {out_path}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    """Generate the SGS heatmap figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default=os.path.join(_ROOT, "results", "individual_plots"),
    )
    args = parser.parse_args()
    out = args.out_dir

    sgs_tex = os.path.join(_ROOT, "results", "sgs_table_minmax.tex")

    print("\n[1] SGS heatmap (Oranges)")
    if os.path.exists(sgs_tex):
        plot_sgs_heatmap(sgs_tex, os.path.join(out, "sgs_heatmap.png"))
    else:
        print(f"  [SKIP] {sgs_tex} not found — run utils/significance_test.py first")

    print("\n[2] SGS bar plots (mean ± std across datasets)")
    if os.path.exists(sgs_tex):
        plot_sgs_barplot(sgs_tex, os.path.join(out, "sgs_barplot.png"))
    else:
        print(f"  [SKIP] {sgs_tex} not found — run utils/significance_test.py first")

    print("\n[3] SGS sorted bar plot — all models (incl. closed)")
    plot_sgs_sorted_bars(os.path.join(out, "sgs_sorted_bars_all.png"), include_closed=True)

    print("\n[4] SGS sorted bar plot — open-source only")
    plot_sgs_sorted_bars(os.path.join(out, "sgs_sorted_bars_open.png"), include_closed=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
