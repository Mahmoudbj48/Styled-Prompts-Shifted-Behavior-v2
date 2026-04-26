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

    # Append mean column
    row_means = np.nanmean(data, axis=1, keepdims=True)
    data_plot = np.hstack([data, row_means])
    col_labels = METRIC_LABELS + ["Mean"]
    n_rows, n_cols = data_plot.shape

    vmin = float(np.nanmin(data_plot))
    vmax = float(np.nanmax(data_plot))
    if vmin == vmax:
        vmax = vmin + 1e-6

    fig_w = max(10, n_cols * 1.8 + 3.0)
    fig_h = max(5, n_rows * 0.65 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor("gray")

    im = ax.imshow(data_plot, cmap="Oranges", vmin=vmin, vmax=vmax, aspect="auto")

    # Metric column labels on top + x-axis name
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=16, fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Metrics", fontsize=16, fontweight="bold", labelpad=10)
    ax.tick_params(axis="x", which="both", length=0)

    # Model labels on y-axis + y-axis name
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(model_labels, fontsize=14, fontweight="bold")
    ax.set_ylabel("Models", fontsize=16, fontweight="bold", labelpad=10)
    ax.tick_params(axis="y", which="both", length=0)

    # Cell value annotations
    for i in range(n_rows):
        for j in range(n_cols):
            v = data_plot[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=12, color="#aaa")
            else:
                norm_v = (v - vmin) / (vmax - vmin)
                text_color = "white" if norm_v > 0.60 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=12, color=text_color)

    # Cell borders
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label("SGS", fontsize=14)

    plt.tight_layout()
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

    print("\nDone.")


if __name__ == "__main__":
    main()
