# utils/plot_utils.py
"""
Shared plotting helpers used across all utils plot modules.

Centralises the five helpers that were previously duplicated in
politeness_plots.py, surface_plots.py, structuredness_plots.py,
and cot_plots.py:

  - apply_neurips_style()
  - _coerce_numeric()
  - load_results_csvs()
  - _format_axis()
  - _save_figure()
"""
import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------
# NeurIPS-style matplotlib defaults
# --------------------------
def apply_neurips_style() -> None:
    """
    Apply conservative, LaTeX-friendly style that matches NeurIPS paper aesthetics:
    - readable fonts
    - clean lines
    - no heavy background
    """
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,

        "lines.linewidth": 1.5,
        "lines.markersize": 4,

        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.7,

        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        "figure.autolayout": False,
    })


# --------------------------
# I/O helpers
# --------------------------
def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
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


# --------------------------
# Axis / figure helpers
# --------------------------
def _format_axis(ax, xlabel: str, ylabel: str, title: str = None) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_axisbelow(True)
    try:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    except Exception:
        pass


def _save_figure(fig, out_png: str, *, save_pdf: bool, dpi: int = 300) -> None:
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        out_pdf = os.path.splitext(out_png)[0] + ".pdf"
        fig.savefig(out_pdf, bbox_inches="tight")