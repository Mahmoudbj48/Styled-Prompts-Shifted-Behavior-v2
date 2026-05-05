"""
plot_individual_figures.py
--------------------------
Publication-ready figures driven by the canonical CSVs produced by
``utils/compute_sgs_table``:

    results/sgs/sgs_total_per_model.csv
    results/sgs/sgs_per_dataset.csv
    results/sgs/per_prompt_s.csv

Outputs (under results/sgs/figures/):

  Main paper:
    heatmap_open_models.pdf         (Figure 2 -- open-only Total-SGS heatmap)
    pareto_bleu_vs_mirroring.pdf    (Figure 3 -- BLEU vs MR scatter)

  Appendix:
    per_axis_bar_chart.pdf          (Figure 10 -- open-only grouped bars)

All figures use Times Roman (ACL/EMNLP camera-ready compatible) with
TrueType-embedded fonts via ``apply_neurips_style`` from ``plots.plots``.

Run:
    python plots/plot_individual_figures.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from plots.plots import apply_neurips_style  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Display config (mirrors utils/compute_sgs_table.py)
# ─────────────────────────────────────────────────────────────────────────────

SGS_AXES = [
    "delta_activation_similarity",
    "delta_bleu",
    "delta_bertscore_response",
    "delta_log_prob",
    "delta_entropy",
    "delta_mirroring_rate",
]
AXIS_LABEL = {
    "delta_activation_similarity": r"$\Delta$-Cos",
    "delta_bleu":                  r"$\Delta$-BLEU",
    "delta_bertscore_response":    r"$\Delta$-BERT",
    "delta_log_prob":              r"$\Delta$-Prob",
    "delta_entropy":               r"$\Delta$-Ent",
    "delta_mirroring_rate":        r"$\Delta$-MR",
}

# (key, short_label, family). Open first, then closed.
MODEL_DISPLAY = [
    ("L3.2-3B",            "L-3B",              "Llama"),
    ("L3.1-8B",            "L-8B",              "Llama"),
    ("G-2B",               "G-2B",              "Gemma"),
    ("G-7B",               "G-7B",              "Gemma"),
    ("G4-E4B",             "G4-E4B",            "Gemma"),
    ("Q2.5-1.5B",          "Q-1.5B",            "Qwen"),
    ("Q2.5-7B",            "Q-7B",              "Qwen"),
    ("Q3.5-9B",            "Q3.5-9B",           "Qwen"),
    ("gpt-5.4",            "GPT-5.4",           "Closed"),
    ("gemini-2.5-flash",   "Gemini-2.5-Flash",  "Closed"),
    ("claude-sonnet-4-6",  "Claude-Sonnet-4.6", "Closed"),
]

# Family colours (color-blind-friendly base palette)
FAMILY_COLOR = {
    "Llama":  (0.85, 0.33, 0.24),    # red
    "Gemma":  (0.20, 0.63, 0.17),    # green
    "Qwen":   (0.22, 0.70, 0.85),    # cyan
    "Closed": (0.45, 0.45, 0.45),    # gray (fallback / open-aware plots)
}

# Per-closed-model distinct colours: brown / green / blue, all in darker
# shades so they read as a separate "tier" from the lighter open-family
# hues (Llama=red, Gemma=green, Qwen=cyan).
CLOSED_MODEL_COLOR = {
    "gpt-5.4":            (0.00, 0.35, 0.12),   # dark green
    "gemini-2.5-flash":   (0.06, 0.20, 0.55),   # dark blue (navy)
    "claude-sonnet-4-6":  (0.40, 0.20, 0.08),   # dark brown
}

METRIC_GROUPS = [
    ("Activation",                [0]),
    ("Generation Consistency",    [1, 2]),
    ("Confidence",                [3, 4]),
    ("Mirroring",                 [5]),
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_totals(only_open: bool = False) -> tuple[list[str], list[str], dict, np.ndarray]:
    """Return (keys, short_labels, fam_map, totals_2d)."""
    df = pd.read_csv(_ROOT / "results" / "sgs" / "sgs_total_per_model.csv")
    keys, labels, fams = [], [], {}
    for k, lbl, fam in MODEL_DISPLAY:
        if k not in df["model"].unique():
            continue
        if only_open and fam == "Closed":
            continue
        keys.append(k); labels.append(lbl); fams[lbl] = fam
    piv = (df[df["model"].isin(keys)]
           .pivot_table(index="model", columns="axis", values="sgs")
           .reindex(keys)[SGS_AXES])
    return keys, labels, fams, piv.to_numpy()


def load_per_prompt_std(only_open: bool = False) -> tuple[list[str], np.ndarray]:
    """Return (short_labels, std_2d) where std_2d[i, j] = std over prompts
    of s_X(i,j) for model i and axis j. Rows aligned with `load_totals`."""
    df = pd.read_csv(_ROOT / "results" / "sgs" / "per_prompt_s.csv")
    keys, labels, _, _ = load_totals(only_open=only_open)
    out = np.full((len(keys), len(SGS_AXES)), np.nan)
    for i, k in enumerate(keys):
        for j, ax in enumerate(SGS_AXES):
            sub = df[(df["model"] == k) & (df["axis"] == ax)]["s_X"].dropna()
            if len(sub) >= 2:
                out[i, j] = float(sub.std(ddof=1))
    return labels, out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Heatmap (open models)
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap_open(out_path: Path) -> None:
    """Figure 2: Total-SGS heatmap, 8 open models x 6 axes. Per-column
    normalisation (each column's colours span that column's own min..max)
    so within-axis differences are visible despite the very different
    cross-axis magnitudes (Δ-BLEU is ~10x larger than Δ-Ent etc.).
    Values shown to 3 decimal places; family brackets on the left;
    axis-group brackets on top."""
    apply_neurips_style()
    keys, labels, fams, totals = load_totals(only_open=True)
    n_rows, n_cols = totals.shape

    # per-column min/max for independent colour scales
    col_min = np.nanmin(totals, axis=0)
    col_max = np.nanmax(totals, axis=0)
    col_max = np.where(col_max == col_min, col_min + 1e-6, col_max)
    cmap = plt.get_cmap("Oranges")

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.8); sp.set_edgecolor("gray")

    from matplotlib.patches import Rectangle
    for i in range(n_rows):
        for j in range(n_cols):
            v = totals[i, j]
            if np.isnan(v):
                color = "#f0f0f0"
            else:
                t = (v - col_min[j]) / (col_max[j] - col_min[j])
                color = cmap(0.10 + 0.85 * t)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                    facecolor=color, edgecolor="gray", linewidth=0.4))

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    # column labels (top)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([AXIS_LABEL[a] for a in SGS_AXES],
                       fontsize=10, fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", which="both", length=0)

    # axis-group brackets above
    y_brack = -1.05
    for axis_name, indices in METRIC_GROUPS:
        cs, ce = indices[0], indices[-1]
        mid = (cs + ce) / 2.0
        ax.plot([cs - 0.4, ce + 0.4], [y_brack, y_brack],
                color="black", lw=0.9, clip_on=False)
        ax.plot([cs - 0.4, cs - 0.4], [y_brack, y_brack + 0.07],
                color="black", lw=0.9, clip_on=False)
        ax.plot([ce + 0.4, ce + 0.4], [y_brack, y_brack + 0.07],
                color="black", lw=0.9, clip_on=False)
        ax.text(mid, y_brack - 0.10, axis_name,
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", fontstyle="italic",
                clip_on=False)

    # model labels (left)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=9, fontweight="bold")
    ax.tick_params(axis="y", which="both", length=0)

    # family brackets (left)
    runs = []; cur = fams[labels[0]]; cs = 0
    for idx in range(1, n_rows):
        f = fams[labels[idx]]
        if f != cur:
            runs.append((cur, cs, idx - 1)); cur, cs = f, idx
    runs.append((cur, cs, n_rows - 1))
    x_brack = -0.95
    for fam_name, rs, re in runs:
        mid = (rs + re) / 2.0
        ax.plot([x_brack, x_brack], [rs - 0.4, re + 0.4],
                color="black", lw=0.9, clip_on=False)
        ax.plot([x_brack, x_brack + 0.06], [rs - 0.4, rs - 0.4],
                color="black", lw=0.9, clip_on=False)
        ax.plot([x_brack, x_brack + 0.06], [re + 0.4, re + 0.4],
                color="black", lw=0.9, clip_on=False)
        ax.text(x_brack - 0.10, mid, fam_name,
                ha="right", va="center",
                fontsize=9, fontweight="bold", fontstyle="italic",
                rotation=90, clip_on=False)

    # cell values (text colour switches based on per-column normalised value)
    for i in range(n_rows):
        for j in range(n_cols):
            v = totals[i, j]
            if np.isnan(v):
                ax.text(j, i, r"$\times$", ha="center", va="center",
                        fontsize=8, color="#999")
            else:
                t = (v - col_min[j]) / (col_max[j] - col_min[j])
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if t > 0.55 else "black")

    # colour bar -- normalised [0, 1] because each column has its own scale
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Best\n(column min)", "Worst\n(column max)"])
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] {out_path.relative_to(_ROOT)}  +  {out_path.with_suffix('.png').name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Pareto plot (BLEU vs Mirroring)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto_bleu_vs_mr(out_path: Path) -> None:
    """Figure 3: scatter of all 11 models in (SGS_BLEU, SGS_MR) plane.
    Open models are circles colored by family; closed models are gray
    squares. Pareto frontier (non-dominated set, both axes minimised) drawn
    as a dashed black line."""
    apply_neurips_style()
    keys, labels, fams, totals = load_totals(only_open=False)
    bleu_idx = SGS_AXES.index("delta_bleu")
    mr_idx   = SGS_AXES.index("delta_mirroring_rate")
    xs = totals[:, bleu_idx]
    ys = totals[:, mr_idx]

    # keep only points where both coords are defined
    valid = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[valid], ys[valid]
    labels = [labels[i] for i, ok in enumerate(valid) if ok]
    fam_for = [fams[lbl] for lbl in labels]

    # Pareto front (non-dominated under both-axes-minimised)
    order = np.argsort(xs)
    front_idx = []
    cur_min_y = np.inf
    for k in order:
        if ys[k] < cur_min_y:
            front_idx.append(k); cur_min_y = ys[k]

    # axis range starts at exactly (0, 0) so the Ideal star sits on the
    # origin / bottom-left axis intersection.
    x_lo, y_lo = 0.0, 0.0
    x_hi = np.nanmax(xs) * 1.05
    y_hi = np.nanmax(ys) * 1.10

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.grid(True, which="major", linestyle=":", alpha=0.45)
    ax.set_axisbelow(True)

    # ideal point at origin -- clip_on=False so it isn't cut by the axis
    ax.scatter([0], [0], marker="*", s=200, color="gold",
               edgecolors="black", linewidths=0.8, zorder=5,
               clip_on=False,
               label="Ideal")
    ax.annotate("Ideal", xy=(0, 0), xytext=(8, 6),
                textcoords="offset points",
                fontsize=8, fontstyle="italic", color="black",
                annotation_clip=False)

    # Pareto frontier as a staircase: vertical from top down to first
    # frontier point, then a step to the right at constant y for each
    # subsequent frontier point, finally extending right to the plot edge.
    fx = np.sort(xs[front_idx])
    fy = ys[front_idx][np.argsort(xs[front_idx])]
    stair_x = [fx[0]]
    stair_y = [y_hi]
    for i in range(len(fx)):
        stair_x.append(fx[i]); stair_y.append(fy[i])
        right_x = fx[i + 1] if i + 1 < len(fx) else x_hi
        stair_x.append(right_x); stair_y.append(fy[i])
    ax.plot(stair_x, stair_y, "k--", lw=1.1, alpha=0.6, zorder=2,
            label="Pareto frontier")

    # mapping from short label back to model key (for per-model colours)
    key_for_label = {lbl: keys[i] for i, lbl in enumerate(labels)
                     if i < len(keys)}
    # rebuild including the valid mask
    valid_keys = [keys[i] for i, ok in enumerate(valid) if ok]
    key_for_label = {labels[i]: valid_keys[i] for i in range(len(labels))}

    # data points (closed models get per-model colours; open models keep
    # the family colour)
    for i, lbl in enumerate(labels):
        fam = fam_for[i]
        is_closed = (fam == "Closed")
        marker = "s" if is_closed else "o"
        if is_closed:
            face = CLOSED_MODEL_COLOR.get(key_for_label[lbl], FAMILY_COLOR["Closed"])
        else:
            face = FAMILY_COLOR[fam]
        ax.scatter(xs[i], ys[i], marker=marker, s=72,
                   facecolors=face, edgecolors="black", linewidths=0.7,
                   zorder=3, clip_on=False)

    # Closed-model labels are abbreviated in the plot (full names live in
    # the legend) so that the labels fit cleanly between the closely
    # spaced bottom-row markers.
    DISPLAY_LABEL = {
        "Gemini-2.5-Flash":  "Gemini",
        "GPT-5.4":           "GPT",
        "Claude-Sonnet-4.6": "Claude",
    }

    # Per-model label offsets (in points). Each label sits on a distinct
    # side of its marker so labels never sit on the same horizontal line
    # as a neighbour.
    label_offsets = {
        # closed cluster (bottom row)
        "Gemini-2.5-Flash":  (-30,   6),    # upper-LEFT of square
        "GPT-5.4":           ( -9,  10),    # upper-middle (centered above)
        "Claude-Sonnet-4.6": (  8,   6),    # upper-right
        # top-right cluster (Q-7B / L-8B / G-7B all near MR ~ 0.55-0.59)
        "Q-7B":              (  6,   3),    # upper-right, tight
        "L-8B":              (-22,   3),    # upper-left,  tight
        "G-7B":              (  6,  -10),   # below-right, tight
        # middle cluster (MR ~ 0.32-0.44)
        "Q3.5-9B":           (-32,   3),    # upper-left,  tight
        "L-3B":              (-22,  -8),    # below-left,  tight
        "G-2B":              (  6,  -10),   # below-right, tight
        "Q-1.5B":            (-22,   4),    # upper-left,  tight
        "G4-E4B":            (  6,   3),    # upper-right, tight
    }
    for i, lbl in enumerate(labels):
        dx, dy = label_offsets.get(lbl, (6, 6))
        text = DISPLAY_LABEL.get(lbl, lbl)
        ax.annotate(text, xy=(xs[i], ys[i]), xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=7.5,
                    annotation_clip=False)

    # apply the axis range computed above
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel(r"$\mathrm{SGS}_{\mathrm{BLEU}}$  (generation consistency)",
                  fontsize=10, fontweight="bold")
    ax.set_ylabel(r"$\mathrm{SGS}_{\mathrm{MR}}$  (response mirroring)",
                  fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Legend organised into three labelled sections (Open / Closed /
    # Reference). Each section starts with a bold heading line (a blank
    # patch carrying just the heading text), followed by its members.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch as _Patch
    blank = _Patch(facecolor="none", edgecolor="none")

    handles = [
        # ── Open ─────
        blank, Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor=FAMILY_COLOR["Llama"], markeredgecolor="black",
                       markersize=8),
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=FAMILY_COLOR["Gemma"], markeredgecolor="black",
               markersize=8),
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=FAMILY_COLOR["Qwen"], markeredgecolor="black",
               markersize=8),
        # ── separator ─
        blank,
        # ── Closed ────
        blank, Line2D([0], [0], marker="s", linestyle="",
                       markerfacecolor=CLOSED_MODEL_COLOR["gpt-5.4"],
                       markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="s", linestyle="",
               markerfacecolor=CLOSED_MODEL_COLOR["gemini-2.5-flash"],
               markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="s", linestyle="",
               markerfacecolor=CLOSED_MODEL_COLOR["claude-sonnet-4-6"],
               markeredgecolor="black", markersize=8),
        # ── separator ─
        blank,
        # ── Reference ─
        blank, Line2D([0], [0], marker="*", linestyle="",
                       markerfacecolor="gold", markeredgecolor="black",
                       markersize=11),
        Line2D([0], [0], color="black", linestyle="--", lw=1.0),
    ]
    leg_labels = [
        r"$\bf{Open\ models}$", "Llama", "Gemma", "Qwen",
        " ",
        r"$\bf{Closed\ models}$", "GPT-5.4", "Gemini-2.5-Flash", "Claude-Sonnet-4.6",
        " ",
        r"$\bf{Reference}$", "Ideal", "Pareto frontier",
    ]
    ax.legend(handles, leg_labels, loc="upper left",
              fontsize=7.5,
              framealpha=0.92, handlelength=1.3, borderpad=0.6,
              labelspacing=0.55, handletextpad=0.6)

    plt.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] {out_path.relative_to(_ROOT)}  +  {out_path.with_suffix('.png').name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10 — Per-axis grouped bar chart (open models)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_axis_bars(out_path: Path) -> None:
    """Figure 10: 6 axis groups on x; one bar per open model in each group.
    Bars colored by family, error bars = std of s_X(i) across prompts."""
    apply_neurips_style()
    keys, labels, fams, totals = load_totals(only_open=True)
    _, stds = load_per_prompt_std(only_open=True)
    n_models = len(keys)

    # family-shaded colours for distinct bars within a family
    family_count = {}
    family_seen = {}
    for lbl in labels:
        family_count[fams[lbl]] = family_count.get(fams[lbl], 0) + 1
    for f in family_count:
        family_seen[f] = 0
    color_for: dict[str, tuple] = {}
    for lbl in labels:
        f = fams[lbl]
        idx = family_seen[f]; tot = family_count[f]
        base = FAMILY_COLOR[f]
        if tot <= 1:
            color_for[lbl] = base
        else:
            t = 0.45 + 0.55 * idx / (tot - 1)
            color_for[lbl] = tuple(b * t + (1 - t) * 1.0 for b in base)
        family_seen[f] += 1

    bar_width = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.grid(False)

    xtick_centres = []
    legend_handles = {}
    for mi, ax_col in enumerate(SGS_AXES):
        # collect (value, std, label) for non-NaN models, sort ascending so
        # the leftmost bar in each group is the most stable model
        triples = []
        for bi, lbl in enumerate(labels):
            v = totals[bi, mi]; s = stds[bi, mi]
            if np.isnan(v):
                continue
            err = 0.0 if np.isnan(s) else s
            triples.append((v, err, lbl))
        triples.sort(key=lambda t: t[0])

        group_left = mi * (n_models + 2) * bar_width
        for bi, (v, err, lbl) in enumerate(triples):
            xpos = group_left + bi * bar_width
            bar = ax.bar(xpos, v, width=bar_width * 0.92,
                         yerr=err,
                         color=color_for[lbl],
                         edgecolor="gray", linewidth=0.4,
                         capsize=2,
                         error_kw=dict(elinewidth=0.7, ecolor="black"))
            if lbl not in legend_handles:
                legend_handles[lbl] = bar[0]
        # centre is over the populated bars
        n_drawn = len(triples)
        centre = group_left + (n_drawn - 1) * bar_width / 2 if n_drawn else group_left
        xtick_centres.append(centre)

    ax.set_xticks(xtick_centres)
    ax.set_xticklabels([AXIS_LABEL[a] for a in SGS_AXES],
                       fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Total $\mathrm{SGS}_X$", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # axis-family brackets below the x-axis
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for axis_name, indices in METRIC_GROUPS:
        x_left  = xtick_centres[indices[0]]
        x_right = xtick_centres[indices[-1]]
        half = (n_models - 1) * bar_width / 2 + bar_width
        x_start = x_left  - half
        x_end   = x_right + half
        x_mid   = (x_start + x_end) / 2
        y_brack = -0.12
        ax.plot([x_start, x_end], [y_brack, y_brack],
                color="black", lw=1.0, clip_on=False, transform=trans)
        ax.plot([x_start, x_start], [y_brack, y_brack + 0.025],
                color="black", lw=1.0, clip_on=False, transform=trans)
        ax.plot([x_end, x_end], [y_brack, y_brack + 0.025],
                color="black", lw=1.0, clip_on=False, transform=trans)
        ax.text(x_mid, y_brack - 0.025, axis_name,
                ha="center", va="top",
                fontsize=9, fontweight="bold", fontstyle="italic",
                clip_on=False, transform=trans)

    # legend grouped by family
    from matplotlib.patches import Patch
    family_order = ["Llama", "Gemma", "Qwen"]
    handles, leg_labels = [], []
    for fi, fam in enumerate(family_order):
        if fi > 0:
            handles.append(Patch(facecolor="none", edgecolor="none"))
            leg_labels.append("")
        handles.append(Patch(facecolor="none", edgecolor="none"))
        leg_labels.append(rf"$\mathbf{{{fam}}}$")
        for lbl in labels:
            if fams[lbl] == fam and lbl in legend_handles:
                handles.append(legend_handles[lbl])
                leg_labels.append(lbl)
    ax.legend(handles, leg_labels,
              fontsize=8, ncol=1,
              loc="center left", bbox_to_anchor=(1.01, 0.5),
              framealpha=0.95, borderaxespad=0,
              handletextpad=0.5, handlelength=1.2,
              labelspacing=0.3)

    plt.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] {out_path.relative_to(_ROOT)}  +  {out_path.with_suffix('.png').name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",
                        default=str(_ROOT / "results" / "sgs" / "figures"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Figure 2]  Heatmap (open models)")
    plot_heatmap_open(out_dir / "heatmap_open_models.pdf")

    print("\n[Figure 3]  Pareto plot (BLEU vs Mirroring)")
    plot_pareto_bleu_vs_mr(out_dir / "pareto_bleu_vs_mirroring.pdf")

    print("\n[Figure 10] Per-axis bar chart (open models)")
    plot_per_axis_bars(out_dir / "per_axis_bar_chart.pdf")

    print("\n[done]")


if __name__ == "__main__":
    main()
