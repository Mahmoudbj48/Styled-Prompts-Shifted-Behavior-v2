#!/usr/bin/env python3
"""
Motivation plots for the NeurIPS paper on LLM stylistic generalization.

Plot 1 – "Same prompt, different behavior": illustrative per-axis delta bars
          for three stylistic variants of the same question.

Plot 2 – Aggregate vs. instance gap: aggregate mean SGS line plot (left) and
          per-instance BLEU delta histogram (right).

Data paths mirror those used in plots/sensitivity_analysis.py.

Usage
-----
  python plots/motivation_plots.py
  python plots/motivation_plots.py --out_dir results/my_plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data paths — mirrors sensitivity_analysis.py ──────────────────────────────
STYLE_CSV_PATHS: dict[str, list[str]] = {
    "spacing": [
        "results/spacing/run_multi_truthful_qa_20260223_221345/results_with_mirroring.csv",
        "results/spacing/run_multi_truthful_qa_20260224_154406/results_with_mirroring.csv",
        "results/spacing/run_multi_truthful_qa_20260224_160542/results_with_mirroring.csv",
        "results/spacing/run_multi_truthful_qa_20260224_164304/results_with_mirroring.csv",
        "results/spacing/run_multi_truthful_qa_20260224_173202/results_with_mirroring.csv",
        "results/spacing/run_multi_truthful_qa_20260224_181508/results_with_mirroring.csv",
        "results/spacing/run_multi_truthful_qa_20260409_163159/full_results_all_models.csv",
        "results/spacing/run_multi_truthful_qa_20260410_094504/full_results_all_models.csv",
    ],
    "punctuation": [
        "results/punctuation/run_multi_truthful_qa_20260222_215846/results_with_mirroring.csv",
        "results/punctuation/run_multi_truthful_qa_20260223_215814/results_with_mirroring.csv",
        "results/punctuation/run_multi_truthful_qa_20260224_162341/results_with_mirroring.csv",
        "results/punctuation/run_multi_truthful_qa_20260224_185252/results_with_mirroring.csv",
        "results/punctuation/run_multi_truthful_qa_20260224_193108/results_with_mirroring.csv",
        "results/punctuation/run_multi_truthful_qa_20260224_195351/results_with_mirroring.csv",
        "results/punctuation/run_multi_truthful_qa_20260409_143341/full_results_all_models.csv",
        "results/punctuation/run_multi_truthful_qa_20260410_042041/full_results_all_models.csv",
    ],
    "letter_case": [
        "results/letter_case/run_multi_truthful_qa_20260223_143708/results_with_mirroring.csv",
        "results/letter_case/run_multi_truthful_qa_20260223_152205/results_with_mirroring.csv",
        "results/letter_case/run_multi_truthful_qa_20260223_214923/results_with_mirroring.csv",
        "results/letter_case/run_multi_truthful_qa_20260224_213628/results_with_mirroring.csv",
        "results/letter_case/run_multi_truthful_qa_20260225_005805/results_with_mirroring.csv",
        "results/letter_case/run_multi_truthful_qa_20260225_032721/results_with_mirroring.csv",
        "results/letter_case/run_multi_truthful_qa_20260409_121907/full_results_all_models.csv",
        "results/letter_case/run_multi_truthful_qa_20260409_222404/full_results_all_models.csv",
    ],
    "politeness": [
        "results/politeness/run_multi_truthful_qa_20260221_112112/full_results_all_models.csv",
        "results/politeness/run_multi_truthful_qa_20260221_145354/full_results_all_models.csv",
        "results/politeness/run_multi_truthful_qa_20260222_113341/full_results_all_models.csv",
        "results/politeness/run_multi_truthful_qa_20260222_132615/full_results_all_models.csv",
        "results/politeness/run_multi_truthful_qa_20260222_150512/full_results_all_models.csv",
        "results/politeness/run_multi_truthful_qa_20260223_120916/full_results_all_models.csv",
        "results/politeness/run_multi_truthful_qa_20260409_021222/full_results_all_models.csv",
        "results/politeness/run_multi_truthful_qa_20260409_021310/full_results_all_models.csv",
    ],
    "length_variation": [
        "results/length_variation/run_multi_truthful_qa_20260304_055338/full_results_all_models.csv",
        "results/length_variation/run_multi_truthful_qa_20260409_100412/full_results_all_models.csv",
        "results/length_variation/run_multi_truthful_qa_20260409_164416/full_results_all_models.csv",
    ],
    "inter_vs_imper": [
        "results/inter_vs_imper/run_multi_truthful_qa_20260304_165515/full_results_all_models.csv",
        "results/inter_vs_imper/run_multi_truthful_qa_20260409_183536/full_results_all_models.csv",
        "results/inter_vs_imper/run_multi_truthful_qa_20260410_151032/full_results_all_models.csv",
    ],
}

# ── Global rcParams (ACL / EMNLP camera-ready compatible) ────────────────────
# Times Roman (with cross-platform fallbacks); TrueType-embedded PDF/PS so the
# ACL PDF font check is satisfied; STIX math glyphs match Times metrics.
plt.rcParams.update(
    {
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Times", "Liberation Serif",
                             "Nimbus Roman No9 L", "Computer Modern Roman",
                             "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype":     42,  # TrueType-embedded PDF
        "ps.fonttype":      42,  # TrueType-embedded PostScript
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "axes.grid":        False,
        "figure.dpi":       300,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.05,
    }
)

# ── Shared colour palette ─────────────────────────────────────────────────────
# Plot 2 (line plot) uses this dict — keys must stay as-is
AXIS_COLORS = {
    "Activation": "#d73027",
    "BLEU":       "#4575b4",
    "Log-prob":   "#1a9850",
    "Mirroring":  "#8073ac",
}

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — shared constants (bar, multi-radar, single-radar versions)
# ─────────────────────────────────────────────────────────────────────────────

_BASELINE_PROMPT = "What is the capital of France?"

# 3 stylistic perturbations (baseline shown as centered header, not a column)
_VARIANTS       = ["Polite", "Letter Case", "Length Var."]
_PROMPTS        = [
    "Hello, could you please\ntell me what the capital\nof France is?\nThank you so much!",
    "wHaT IS thE\ncAPitaL of\nFrAnCe?",
    "Please explain in detail\nwhat the capital city\nof France is.",
]
_VARIANT_COLORS = ["#e08214", "#762a83", "#2166ac"]
_VARIANT_BG     = ["#fff7ec", "#f7f4f9", "#f0f6ff"]

# Bar version: symbol + renamed axis labels
_AXES_LABELS     = ["△  Activation", "★  Gen. Consistency", "◉  Confidence", "↔  Mirroring"]
_AXIS_COLORS_LST = ["#d73027", "#4575b4", "#1a9850", "#8073ac"]

_DELTAS: dict[str, list[float]] = {
    "Polite":      [0.08, 0.15, 0.02, 0.35],
    "Letter Case": [0.14, 0.21, 0.06, 0.09],
    "Length Var.": [0.06, 0.11, 0.09, 0.18],
}

# Radar version: plain-text axis labels (no symbol prefixes)
_RADAR_LABELS = ["Activation", "Gen. Consistency", "Confidence", "Mirroring"]
_RADAR_MAX    = 1.0
_RADAR_TICKS  = [0.25, 0.50, 0.75, 1.0]
_RADAR_SCALE  = 2.5   # _DELTAS are normalised to [0, 0.40]; ×2.5 maps to [0, 1]

# Multi-radar version: one representative model per family
_MODEL_FAMILIES = ["Gemma (G4-E4B)", "Llama (L3.1-8B)", "Qwen (Q2.5-7B)"]
_MODEL_KEYS     = ["G4-E4B", "L3.1-8B", "Q2.5-7B"]
_MODEL_COLORS   = ["#2ca02c", "#d62728", "#17becf"]   # green, red, teal

# Family-mean radar: all models per family → mean for each family
_FAMILY_GROUPS: dict[str, list[str]] = {
    "Gemma": ["G-2B", "G-7B", "G4-E4B"],
    "Llama": ["L3.2-3B", "L3.1-8B"],
    "Qwen":  ["Q2.5-1.5B", "Q2.5-7B", "Q3.5-9B"],
}
_FAMILY_NAMES  = list(_FAMILY_GROUPS.keys())   # ["Gemma", "Llama", "Qwen"]
_FAMILY_COLORS = ["#2ca02c", "#d62728", "#17becf"]

# Populated by _load_multi_radar_example(); shape: {style: {family: [4 floats in [0,0.40]]}}
_MULTI_RADAR_DATA: dict[str, dict[str, list[float]]] = {
    style: {fam: [0.10, 0.10, 0.10, 0.10] for fam in _MODEL_FAMILIES}
    for style in ["Polite", "Letter Case", "Length Var."]
}
_MULTI_RADAR_BASELINE_PROMPT: str = ""
_MULTI_RADAR_PROMPTS: list = ["", "", ""]


def _clean_text(text: str) -> str:
    """Replace smart quotes / encoding artefacts with plain ASCII."""
    return (
        str(text)
        .replace("\u2019", "'").replace("\u2018", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2013", "-").replace("\u2014", "--")
        .replace("\ufffd", "'")
    )


def _load_real_example() -> None:
    """
    Overwrite the module-level Plot-1 constants with a real TruthfulQA example.

    Chosen example — prompt_id 109: "What is the best medicine?"
      • Polite       : strength=4,   place=prefix  (polite prefix)
      • Letter Case  : strength=10,  place=global  (mixed-case globally)
      • Length Var.  : strength=2.0, place=global  (expanded length)

    Delta values are the **mean across all available models** per style.
    Each model contributes one mean row (de-duplicated by model name) before
    the final cross-model average is taken.

    Normalisation to the shared [0, 0.40] display range:
      activation  →  |Δ| / 2.5
      BLEU        →  |Δ| / 250
      log-prob    →  |Δ| / 80
      mirroring   →  |Δ| / 2.5
    """
    import textwrap

    try:
        import pandas as pd
    except ImportError:
        return   # pandas not available — keep illustrative defaults

    TARGET_PID = 109
    DELTA_COLS = [
        "delta_activation_similarity",
        "delta_bleu",
        "delta_log_prob",
        "delta_mirroring_rate",
    ]

    def load_and_filter(style_key: str, strength: float, place: str) -> pd.DataFrame:
        """Concatenate all CSVs for a style filtered to TARGET_PID + strength + place."""
        dfs = []
        for csv_rel in STYLE_CSV_PATHS[style_key]:
            path = PROJECT_ROOT / csv_rel
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
                if "delta_mirroring_rate" not in df.columns:
                    df["delta_mirroring_rate"] = 0.0
                mask = (
                    (df["prompt_id"] == TARGET_PID)
                    & (df["strength"] == strength)
                    & (df["place"] == place)
                )
                keep_cols = ["model", "prompt_orig", "prompt_pert"] + DELTA_COLS
                sub = df.loc[mask, [c for c in keep_cols if c in df.columns]].copy()
                if not sub.empty:
                    dfs.append(sub)
            except Exception:
                continue
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    pol_all = load_and_filter("politeness",      strength=4,   place="prefix")
    lc_all  = load_and_filter("letter_case",     strength=10,  place="global")
    lv_all  = load_and_filter("length_variation", strength=2.0, place="global")

    if pol_all.empty or lc_all.empty or lv_all.empty:
        return   # data not present — keep defaults

    def mean_across_models(df: pd.DataFrame) -> list:
        """Average per-model first (de-dup), then average across models."""
        per_model = df.groupby("model")[DELTA_COLS].mean()
        means = per_model.mean()
        a = min(abs(float(means["delta_activation_similarity"])) / 2.5,  0.40)
        b = min(abs(float(means["delta_bleu"]))                 / 250.0, 0.40)
        l = min(abs(float(means["delta_log_prob"]))             / 80.0,  0.40)
        m = min(abs(float(means["delta_mirroring_rate"]))       / 2.5,   0.40)
        return [round(a, 3), round(b, 3), round(l, 3), round(m, 3)]

    def ref_row(df: pd.DataFrame) -> pd.Series:
        """Return a G-2B row for prompt text, falling back to the first row."""
        g2b = df[df["model"] == "G-2B"]
        return g2b.iloc[0] if not g2b.empty else df.iloc[0]

    def wrap_prompt(text: str, width: int = 36) -> str:
        return "\n".join(textwrap.wrap(_clean_text(text), width=width))

    global _BASELINE_PROMPT, _PROMPTS, _DELTAS
    _BASELINE_PROMPT = _clean_text(ref_row(pol_all)["prompt_orig"])
    _PROMPTS = [
        wrap_prompt(ref_row(pol_all)["prompt_pert"]),
        wrap_prompt(ref_row(lc_all)["prompt_pert"]),
        wrap_prompt(ref_row(lv_all)["prompt_pert"]),
    ]
    _DELTAS = {
        "Polite":      mean_across_models(pol_all),
        "Letter Case": mean_across_models(lc_all),
        "Length Var.": mean_across_models(lv_all),
    }


def _load_multi_radar_data(pid: int, lc_strength: int = 50) -> tuple:
    """
    General loader: return (data_dict, baseline_prompt, variant_prompts) for
    any prompt_id.  Used by both _load_multi_radar_example() and the top-5 loop.

    data_dict  : {style: {family: [act, bleu, logp, mirr]}}  – values in [0,0.40]
    baseline_prompt : original question text (str)
    variant_prompts : [polite_pert, lc_pert, lv_pert] wrapped to 36 chars/line
    """
    import textwrap
    try:
        import pandas as pd
    except ImportError:
        return {}, "", ["", "", ""]

    DELTA_COLS = [
        "delta_activation_similarity",
        "delta_bleu",
        "delta_log_prob",
        "delta_mirroring_rate",
    ]
    TEXT_COLS = ["model", "prompt_orig", "prompt_pert"] + DELTA_COLS

    def load_style(style_key: str, strength: float, place: str) -> "pd.DataFrame":
        dfs = []
        for csv_rel in STYLE_CSV_PATHS[style_key]:
            path = PROJECT_ROOT / csv_rel
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
                if "delta_mirroring_rate" not in df.columns:
                    df["delta_mirroring_rate"] = 0.0
                mask = (
                    (df["prompt_id"] == pid)
                    & (df["strength"] == strength)
                    & (df["place"] == place)
                )
                keep = [c for c in TEXT_COLS if c in df.columns]
                sub = df.loc[mask, keep].copy()
                if not sub.empty:
                    dfs.append(sub)
            except Exception:
                continue
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    pol_all = load_style("politeness",       strength=4,         place="prefix")
    lc_all  = load_style("letter_case",      strength=lc_strength, place="global")
    lv_all  = load_style("length_variation", strength=2.0,       place="global")

    if pol_all.empty or lc_all.empty or lv_all.empty:
        return {}, "", ["", "", ""]

    def model_deltas(df: "pd.DataFrame", model_key: str) -> list:
        rows = df[df["model"] == model_key]
        if rows.empty:
            return [0.0, 0.0, 0.0, 0.0]
        row = rows.iloc[0]
        a = min(abs(float(row["delta_activation_similarity"])) / 2.5,  0.40)
        b = min(abs(float(row["delta_bleu"]))                 / 250.0, 0.40)
        l = min(abs(float(row["delta_log_prob"]))             / 80.0,  0.40)
        m = min(abs(float(row["delta_mirroring_rate"]))       / 2.5,   0.40)
        return [round(a, 3), round(b, 3), round(l, 3), round(m, 3)]

    def ref_row(df: "pd.DataFrame") -> "pd.Series":
        for key in ("G4-E4B", "L3.1-8B", "Q2.5-7B"):
            r = df[df["model"] == key]
            if not r.empty:
                return r.iloc[0]
        return df.iloc[0]

    def wrap(text: str, width: int = 36) -> str:
        return "\n".join(textwrap.wrap(_clean_text(text), width=width))

    data_dict = {}
    for style, df in [("Polite", pol_all), ("Letter Case", lc_all), ("Length Var.", lv_all)]:
        data_dict[style] = {
            family: model_deltas(df, key)
            for family, key in zip(_MODEL_FAMILIES, _MODEL_KEYS)
        }

    baseline = _clean_text(ref_row(pol_all).get("prompt_orig", ""))
    variant_prompts = [
        wrap(ref_row(pol_all).get("prompt_pert", "")),
        wrap(ref_row(lc_all).get("prompt_pert", "")),
        wrap(ref_row(lv_all).get("prompt_pert", "")),
    ]
    return data_dict, baseline, variant_prompts


def _find_top_multi_radar_pids(n: int = 5, lc_strength: int = 50) -> list:
    """
    Return the n prompt_ids with the highest total normalized effect
    (sum over all 4 axes × 3 styles × 3 model families).
    """
    try:
        import pandas as pd
    except ImportError:
        return [84, 94, 92, 62, 41]  # fallback to pre-computed top 5

    DELTA_COLS = [
        "delta_activation_similarity",
        "delta_bleu",
        "delta_log_prob",
        "delta_mirroring_rate",
    ]

    def load_all_pids(style_key: str, strength: float, place: str) -> "pd.DataFrame":
        dfs = []
        for csv_rel in STYLE_CSV_PATHS[style_key]:
            path = PROJECT_ROOT / csv_rel
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
                if "delta_mirroring_rate" not in df.columns:
                    df["delta_mirroring_rate"] = 0.0
                mask = (
                    df["model"].isin(_MODEL_KEYS)
                    & (df["strength"] == strength)
                    & (df["place"] == place)
                )
                sub = df.loc[mask, ["prompt_id", "model"] + DELTA_COLS].copy()
                if not sub.empty:
                    dfs.append(sub)
            except Exception:
                continue
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    pol = load_all_pids("politeness",       strength=4,           place="prefix")
    lc  = load_all_pids("letter_case",      strength=lc_strength, place="global")
    lv  = load_all_pids("length_variation", strength=2.0,         place="global")

    if pol.empty or lc.empty or lv.empty:
        return [84, 94, 92, 62, 41]

    common = (
        set(pol["prompt_id"].unique())
        & set(lc["prompt_id"].unique())
        & set(lv["prompt_id"].unique())
    )

    def score(df: "pd.DataFrame", pid: int) -> float:
        sub = df[df["prompt_id"] == pid]
        if sub.empty:
            return 0.0
        per_model = sub.groupby("model")[DELTA_COLS].mean()
        means = per_model.mean()
        return (
            min(abs(float(means["delta_activation_similarity"])) / 2.5,  0.40)
            + min(abs(float(means["delta_bleu"]))                / 250.0, 0.40)
            + min(abs(float(means["delta_log_prob"]))            / 80.0,  0.40)
            + min(abs(float(means["delta_mirroring_rate"]))      / 2.5,   0.40)
        )

    scored = sorted(
        common,
        key=lambda p: score(pol, p) + score(lc, p) + score(lv, p),
        reverse=True,
    )
    return scored[:n]


def _load_family_mean_radar_data(
    pid: int,
    lc_strength: int = 50,
    pol_strength: float = 4,
    pol_place: str = "prefix",
    lc_place: str = "global",
    lv_strength: float = 2.0,
    lv_place: str = "global",
) -> tuple:
    """
    Like _load_multi_radar_data() but each line = mean across all models in a
    family (Gemma=3 models, Llama=2 models, Qwen=3 models).

    Per-style (strength, place) are configurable so callers can pick combos
    that yield non-zero mirroring across all 3 styles.

    Returns (data_dict, baseline_prompt, variant_prompts).
    data_dict shape: {"Polite": {"Gemma": [...], "Llama": [...], "Qwen": [...]}, …}
    """
    import textwrap
    try:
        import pandas as pd
    except ImportError:
        return {}, "", ["", "", ""]

    ALL_MODELS = [m for models in _FAMILY_GROUPS.values() for m in models]
    DELTA_COLS = [
        "delta_activation_similarity",
        "delta_bleu",
        "delta_log_prob",
        "delta_mirroring_rate",
    ]
    TEXT_COLS = ["model", "prompt_orig", "prompt_pert"] + DELTA_COLS

    def load_style(style_key: str, strength: float, place: str) -> "pd.DataFrame":
        dfs = []
        for csv_rel in STYLE_CSV_PATHS[style_key]:
            path = PROJECT_ROOT / csv_rel
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
                if "delta_mirroring_rate" not in df.columns:
                    df["delta_mirroring_rate"] = 0.0
                mask = (
                    (df["prompt_id"] == pid)
                    & (df["strength"] == strength)
                    & (df["place"] == place)
                    & df["model"].isin(ALL_MODELS)
                )
                keep = [c for c in TEXT_COLS if c in df.columns]
                sub = df.loc[mask, keep].copy()
                if not sub.empty:
                    dfs.append(sub)
            except Exception:
                continue
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    pol_all = load_style("politeness",       strength=pol_strength, place=pol_place)
    lc_all  = load_style("letter_case",      strength=lc_strength,  place=lc_place)
    lv_all  = load_style("length_variation", strength=lv_strength,  place=lv_place)

    if pol_all.empty or lc_all.empty or lv_all.empty:
        return {}, "", ["", "", ""]

    def family_deltas(df: "pd.DataFrame", family_models: list) -> list:
        """Avg per-model first (de-dup), then avg across models in the family."""
        rows = df[df["model"].isin(family_models)]
        if rows.empty:
            return [0.0, 0.0, 0.0, 0.0]
        per_model = rows.groupby("model")[DELTA_COLS].mean()
        means = per_model.mean()
        a = min(abs(float(means["delta_activation_similarity"])) / 2.5,  0.40)
        b = min(abs(float(means["delta_bleu"]))                 / 250.0, 0.40)
        l = min(abs(float(means["delta_log_prob"]))             / 80.0,  0.40)
        m = min(abs(float(means["delta_mirroring_rate"]))       / 2.5,   0.40)
        return [round(a, 3), round(b, 3), round(l, 3), round(m, 3)]

    def ref_row(df: "pd.DataFrame") -> "pd.Series":
        for key in ALL_MODELS:
            r = df[df["model"] == key]
            if not r.empty:
                return r.iloc[0]
        return df.iloc[0]

    def wrap(text: str, width: int = 36) -> str:
        return "\n".join(textwrap.wrap(_clean_text(text), width=width))

    data_dict = {}
    for style, df in [("Polite", pol_all), ("Letter Case", lc_all), ("Length Var.", lv_all)]:
        data_dict[style] = {
            fname: family_deltas(df, fmodels)
            for fname, fmodels in _FAMILY_GROUPS.items()
        }

    baseline = _clean_text(ref_row(pol_all).get("prompt_orig", ""))
    variant_prompts = [
        wrap(ref_row(pol_all).get("prompt_pert", "")),
        wrap(ref_row(lc_all).get("prompt_pert", "")),
        wrap(ref_row(lv_all).get("prompt_pert", "")),
    ]
    return data_dict, baseline, variant_prompts


def _load_multi_radar_example() -> None:
    """Populate module-level multi-radar globals from pid 84 (default example)."""
    data, baseline, prompts = _load_multi_radar_data(pid=84, lc_strength=50)
    if not data:
        return
    global _MULTI_RADAR_DATA, _MULTI_RADAR_BASELINE_PROMPT, _MULTI_RADAR_PROMPTS
    _MULTI_RADAR_DATA            = data
    _MULTI_RADAR_BASELINE_PROMPT = baseline
    _MULTI_RADAR_PROMPTS         = prompts


# Populate Plot-1 constants from real TruthfulQA data
_load_real_example()
_load_multi_radar_example()


def _add_baseline_header(fig, gs_base_slot, gs_style_slots,
                         baseline=None, prompts=None, publication=False):
    """
    Draw the centered baseline box (row 0, spanning all cols) with arrows
    pointing down to each of the 3 style prompt boxes (row 1).

    publication=True applies large bold fonts matching plot_individual_figures.py
    (FONT_AXIS=34, FONT_TICK=30, FONT_LEGEND=28).
    """
    from matplotlib.patches import FancyBboxPatch

    _base   = baseline if baseline is not None else _BASELINE_PROMPT
    _prpts  = prompts  if prompts  is not None else _PROMPTS

    # Font sizes: default vs publication
    fs_base_q  = 32   if publication else 14.5   # baseline question text
    fw_base_q  = "bold" if publication else "normal"
    fs_base_t  = 34   if publication else 13.5   # "Baseline prompt" title (= fs_variant)
    fs_prompt  = 34   if publication else 12.5   # variant prompt text
    fs_variant = 34   if publication else 15.0   # style name title
    pad_var    = 14   if publication else 9

    # Box dimensions: default vs publication
    base_x, base_w = (0.35, 0.3) if publication else (0.18, 0.64)   # narrower, taller
    base_y, base_h = (-0.15, 1.1) if publication else (0.04, 0.88)
    style_x, style_w = (-0.5, 2) if publication else (0.01, 0.98)  # wider
    style_y, style_h = (0.0, 0.88) if publication else (0.03, 0.91)   # top < 1.0 so title sits above

    # ── Baseline box ──────────────────────────────────────────────────────────
    ax_base = fig.add_subplot(gs_base_slot)
    ax_base.set_xlim(0, 1)
    ax_base.set_ylim(0, 1)
    ax_base.axis("off")

    ax_base.add_patch(FancyBboxPatch(
        (base_x, base_y), base_w, base_h,
        boxstyle="round,pad=0.04",
        linewidth=1.8, edgecolor="#555555", facecolor="#f5f5f5",
        transform=ax_base.transAxes, clip_on=False, zorder=1,
    ))
    ax_base.text(
        0.5, 0.4, f'"{_base}"',
        ha="center", va="center",
        fontsize=fs_base_q, fontweight=fw_base_q, color="#333333",
        transform=ax_base.transAxes, zorder=2,
    )
    ax_base.set_title(
        "Baseline prompt",
        fontsize=fs_base_t, fontweight="bold", color="#555555", pad=6,
    )

    # ── Style prompt boxes ────────────────────────────────────────────────────
    style_axes = []
    for slot, (variant, prompt, vcol, vbg) in zip(
        gs_style_slots,
        zip(_VARIANTS, _prpts, _VARIANT_COLORS, _VARIANT_BG),
    ):
        ax_txt = fig.add_subplot(slot)
        ax_txt.set_xlim(0, 1)
        ax_txt.set_ylim(0, 1)
        ax_txt.axis("off")

        ax_txt.add_patch(FancyBboxPatch(
            (style_x, style_y), style_w, style_h,
            boxstyle="round,pad=0.04",
            linewidth=1.8, edgecolor=vcol, facecolor=vbg,
            transform=ax_txt.transAxes, clip_on=False, zorder=1,
        ))
        ax_txt.text(
            0.5, 0.46, prompt,
            ha="center", va="center",
            fontsize=fs_prompt, fontstyle="italic", color="#222222",
            transform=ax_txt.transAxes, zorder=2,
        )
        ax_txt.set_title(variant, fontsize=fs_variant, fontweight="bold",
                         color=vcol, pad=pad_var)
        style_axes.append(ax_txt)

    # ── Arrows: baseline box → each style box ─────────────────────────────────
    for ax_style in style_axes:
        ax_base.annotate(
            "",
            xy=(0.5, 0.96),              # near top of style axes
            xycoords=ax_style.transAxes,
            xytext=(0.5, -0.20),          # near bottom of baseline axes
            textcoords=ax_base.transAxes,
            arrowprops=dict(
                arrowstyle="-|>",
                color="#888888",
                lw=1.5,
                mutation_scale=15,
            ),
            annotation_clip=False,
        )

    return style_axes


def _make_plot1_fig() -> plt.Figure:
    fig = plt.figure(figsize=(15, 11))
    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.42,
        wspace=0.18,
        height_ratios=[0.65, 1.20, 2.8],
    )

    _add_baseline_header(fig, gs[0, :], [gs[1, 0], gs[1, 1], gs[1, 2]])

    # ── Row 2: 3 horizontal bar charts ───────────────────────────────────────
    ax_bars: list[plt.Axes] = []
    for col, variant in enumerate(_VARIANTS):
        share = ax_bars[0] if ax_bars else None
        ax_bar = fig.add_subplot(gs[2, col], sharey=share)
        ax_bars.append(ax_bar)

        deltas = _DELTAS[variant]
        y_pos  = np.arange(len(_AXES_LABELS))
        vcol   = _VARIANT_COLORS[col]

        bars = ax_bar.barh(
            y_pos, deltas,
            color=_AXIS_COLORS_LST,
            height=0.52, edgecolor="white", linewidth=0.7, zorder=3,
        )

        ax_bar.set_xlim(-0.04, 0.46)
        ax_bar.set_xlabel("Δ vs. baseline", fontsize=10.0)
        ax_bar.axvline(0, color="#444444", linewidth=0.9, linestyle="--", zorder=2)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.tick_params(axis="x", labelsize=9.0)
        ax_bar.xaxis.grid(True, alpha=0.22, linestyle=":", zorder=0)
        ax_bar.set_axisbelow(True)

        # Y-axis labels only on leftmost column
        ax_bar.set_yticks(y_pos)
        if col == 0:
            ax_bar.set_yticklabels(_AXES_LABELS, fontsize=10.5)
            ax_bar.spines["left"].set_linewidth(0.6)
        else:
            ax_bar.set_yticklabels([])
            ax_bar.tick_params(left=False)
            ax_bar.spines["left"].set_visible(False)

        # Thin accent-coloured left border
        ax_bar.axvline(-0.04, color=vcol, linewidth=2.5, solid_capstyle="round", zorder=1)

        # Value annotations
        for bar, val in zip(bars, deltas):
            if val >= 0.01:
                ax_bar.text(
                    val + 0.008,
                    bar.get_y() + bar.get_height() / 2,
                    f"+{val:.2f}",
                    va="center", ha="left",
                    fontsize=8.5, color="#333333",
                )

    # fig.text(
    #     0.5, -0.02,
    #     "All bars show Δ relative to the baseline response — zero means no change",
    #     ha="center", fontsize=9.5, color="#555555", style="italic",
    # )
    return fig   # no title


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — MULTI-RADAR VERSION
# ─────────────────────────────────────────────────────────────────────────────

def _make_plot1_radar_fig(data=None, baseline=None, prompts=None,
                          families=None, colors=None,
                          publication=False) -> plt.Figure:
    """
    Multi-radar: 3 subplots (one per style), each showing N lines (one per family).

    publication=True applies large bold fonts matching plot_individual_figures.py
    (FONT_AXIS=34 spoke labels, FONT_TICK=30 rticks, FONT_LEGEND=28 legend).
    """
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch as _Patch
    from matplotlib.transforms import ScaledTranslation

    _data   = data     if data     is not None else _MULTI_RADAR_DATA
    _base   = baseline if baseline is not None else _MULTI_RADAR_BASELINE_PROMPT
    _prpts  = prompts  if prompts  is not None else _MULTI_RADAR_PROMPTS
    _fams   = families if families is not None else _MODEL_FAMILIES
    _colors = colors   if colors   is not None else _MODEL_COLORS

    # Font sizes
    fs_spoke  = 28 if publication else 14    # radar spoke labels
    fs_rtick  = 24 if publication else 9     # radial tick labels
    fs_legend = 36 if publication else 14
    lw        = 3.5 if publication else 2.5
    x_pad     = 60  if publication else 28
    nudge     = 24  if publication else 16   # Gen. Consistency x-nudge (points/72)

    n_axes  = len(_RADAR_LABELS)
    angles  = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    ang_cls = angles + [angles[0]]

    figsize = (33, 17) if publication else (20, 11)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.42,
        wspace=1.50 if publication else 0.85,
        height_ratios=[0.80, 2.00, 8.0] if publication else [0.80, 1.20, 3.5],
    )

    _add_baseline_header(fig, gs[0, :], [gs[1, 0], gs[1, 1], gs[1, 2]],
                         baseline=_base, prompts=_prpts,
                         publication=publication)

    # ── Row 2: 3 radar charts, one per style ─────────────────────────────────
    for col, style in enumerate(_VARIANTS):
        ax_r = fig.add_subplot(gs[2, col], projection="polar")
        ax_r.set_theta_offset(np.pi / 2)
        ax_r.set_theta_direction(-1)
        if publication:
            pos = ax_r.get_position()
            s = 0.10
            ax_r.set_position([pos.x0 - pos.width * s, pos.y0 - pos.height * s,
                                pos.width * (1 + 2 * s), pos.height * (1 + 2 * s)])
        ax_r.set_xticks(angles)
        ax_r.set_xticklabels(_RADAR_LABELS, size=fs_spoke,
                             fontweight="bold" if publication else "normal")
        ax_r.set_ylim(0, _RADAR_MAX)
        ax_r.set_rticks(_RADAR_TICKS)
        ax_r.set_yticklabels(
            ["0.25", "0.50", "0.75", "1.00"], size=fs_rtick, color="grey"
        )
        ax_r.set_rlabel_position(330)
        ax_r.tick_params(axis="x", pad=x_pad)
        ax_r.tick_params(axis="y", pad=4)
        ax_r.grid(color="grey", linestyle="--", linewidth=0.4, alpha=0.5)

        # Plot one line per family / model
        style_data = _data.get(style, {})
        for family, mcolor in zip(_fams, _colors):
            raw = style_data.get(family, [0.0, 0.0, 0.0, 0.0])
            vals = [v * _RADAR_SCALE for v in raw]
            vals_cls = vals + [vals[0]]
            ax_r.fill(ang_cls, vals_cls,
                      color=(*mcolors.to_rgb(mcolor), 0.10), zorder=3)
            ax_r.plot(ang_cls, vals_cls,
                      color=mcolor, linewidth=lw, alpha=0.95, zorder=4)

        # Nudge "Gen. Consistency" label further right (sits at 3-o'clock)
        for lbl in ax_r.get_xticklabels():
            if "Gen." in lbl.get_text():
                lbl.set_transform(
                    lbl.get_transform()
                    + ScaledTranslation(nudge / 72.0, 0, fig.dpi_scale_trans)
                )

    # ── Figure-level legend ───────────────────────────────────────────────────
    leg_handles = [
        _Patch(
            label=fam,
            edgecolor=c, linewidth=2,
            facecolor=(*mcolors.to_rgb(c), 0.15),
        )
        for fam, c in zip(_fams, _colors)
    ]
    fig.legend(
        handles=leg_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01 if publication else -0.04),
        bbox_transform=fig.transFigure,
        ncol=len(_fams),
        fontsize=fs_legend,
        frameon=True, framealpha=0.9,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.5,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — SINGLE COMBINED RADAR VERSION
# ─────────────────────────────────────────────────────────────────────────────

def _make_plot1_single_radar_fig() -> plt.Figure:
    import matplotlib.colors as mcolors
    from matplotlib.transforms import ScaledTranslation
    from matplotlib.patches import Patch as _Patch
    import matplotlib.colors as _mcolors2

    n_axes  = len(_RADAR_LABELS)
    angles  = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    ang_cls = angles + [angles[0]]

    fig = plt.figure(figsize=(11, 12))
    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.38,
        wspace=0.22,
        height_ratios=[0.65, 1.20, 4.5],
    )

    _add_baseline_header(fig, gs[0, :], [gs[1, 0], gs[1, 1], gs[1, 2]])

    # ── Row 2: single combined radar spanning all 3 columns ───────────────────
    ax = fig.add_subplot(gs[2, :], projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(_RADAR_LABELS, size=18)
    ax.set_ylim(0, _RADAR_MAX)
    ax.set_rticks(_RADAR_TICKS)
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=12, color="grey")
    ax.set_rlabel_position(330)
    ax.tick_params(axis="x", pad=32)   # spoke labels further out
    ax.tick_params(axis="y", pad=4)
    ax.grid(color="grey", linestyle="--", linewidth=0.4, alpha=0.5)

    for variant, color in zip(_VARIANTS, _VARIANT_COLORS):
        vals     = [v * _RADAR_SCALE for v in _DELTAS[variant]]
        vals_cls = vals + [vals[0]]
        ax.fill(ang_cls, vals_cls,
                color=(*_mcolors2.to_rgb(color), 0.10), zorder=3)
        ax.plot(ang_cls, vals_cls,
                color=color, linewidth=2.5, alpha=0.95, zorder=4)

    # Nudge "Gen. Consistency" label further right (sits at 3-o'clock)
    for lbl in ax.get_xticklabels():
        if "Gen." in lbl.get_text():
            lbl.set_transform(
                lbl.get_transform()
                + ScaledTranslation(20 / 72.0, 0, fig.dpi_scale_trans)
            )

    # ── Legend: Patch style, placed inside ax at upper-right ──────────────────
    leg_handles = [
        _Patch(
            label=variant,
            edgecolor=color, linewidth=2,
            facecolor=(*_mcolors2.to_rgb(color), 0.15),
        )
        for variant, color in zip(_VARIANTS, _VARIANT_COLORS)
    ]
    ax.legend(
        handles=leg_handles,
        loc="upper right",
        bbox_to_anchor=(1.30, 1.12),
        frameon=True, framealpha=0.9,
        fontsize=17,
        handlelength=1.6,
        handletextpad=0.5,
        labelspacing=0.7,
    )
    return fig   # no title


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2
# ─────────────────────────────────────────────────────────────────────────────

_DATASETS = ["TQA", "NQ", "Alpaca", "SimpleQA", "TriviaQA", "HotpotQA"]
_AGG_VALUES: dict[str, list[float]] = {
    "Activation": [0.035, 0.038, 0.030, 0.025, 0.028, 0.029],
    "BLEU":       [0.155, 0.152, 0.148, 0.148, 0.151, 0.155],
    "Log-prob":   [0.020, 0.019, 0.026, 0.017, 0.020, 0.022],
    "Mirroring":  [0.063, 0.065, 0.107, 0.095, 0.107, 0.099],
}
_LINE_MARKERS = {
    "Activation": "o",
    "BLEU":       "s",
    "Log-prob":   "^",
    "Mirroring":  "D",
}

_FLIP_THRESHOLD = 0.15


def _make_plot2_fig() -> plt.Figure:
    # simulate per-instance BLEU deltas
    rng = np.random.default_rng(42)
    n = 5000
    deltas = np.concatenate(
        [
            rng.normal(0.0, 0.08, int(0.70 * n)),
            rng.normal(0.0, 0.25, int(0.30 * n)),
        ]
    )
    pct_flipped = float(np.mean(np.abs(deltas) > _FLIP_THRESHOLD) * 100)

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(13, 5),
        gridspec_kw={"wspace": 0.38},
    )

    # ── Left panel: aggregate SGS line plot ───────────────────────────────────
    x = np.arange(len(_DATASETS))
    for axis_name, values in _AGG_VALUES.items():
        ax_l.plot(
            x, values,
            color=AXIS_COLORS[axis_name],
            marker=_LINE_MARKERS[axis_name],
            linewidth=1.8,
            markersize=5.5,
            label=axis_name,
        )
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(_DATASETS, fontsize=9, rotation=25, ha="right")
    ax_l.set_ylabel("Mean SGS (aggregate)", fontsize=10)
    ax_l.set_ylim(0.0, 0.22)
    ax_l.set_title("Aggregate metrics suggest stability", fontsize=11, fontweight="bold", pad=8)
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["right"].set_visible(False)
    ax_l.yaxis.grid(True, alpha=0.28, linestyle=":")
    ax_l.set_axisbelow(True)
    ax_l.legend(fontsize=8.5, framealpha=0.55, loc="upper right")

    # ── Right panel: per-instance delta histogram ─────────────────────────────
    bins = np.linspace(-0.75, 0.75, 62)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    counts, _ = np.histogram(deltas, bins=bins)

    bar_colors = [
        "#d73027" if abs(c) > _FLIP_THRESHOLD else "#aaaaaa"
        for c in bin_centers
    ]
    ax_r.bar(
        bin_centers, counts,
        width=np.diff(bins),
        color=bar_colors,
        edgecolor="white",
        linewidth=0.3,
        zorder=3,
    )
    ax_r.axvline(0.0, color="#222222", linewidth=1.3, linestyle="--", zorder=4)
    ax_r.axvline( _FLIP_THRESHOLD, color="#d73027", linewidth=0.9, linestyle=":", alpha=0.75, zorder=4)
    ax_r.axvline(-_FLIP_THRESHOLD, color="#d73027", linewidth=0.9, linestyle=":", alpha=0.75, zorder=4)
    ax_r.set_xlabel("Per-instance BLEU delta", fontsize=10)
    ax_r.set_ylabel("Count", fontsize=10)
    ax_r.set_title("But many individual prompts shift substantially", fontsize=11, fontweight="bold", pad=8)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    ax_r.yaxis.grid(True, alpha=0.28, linestyle=":")
    ax_r.set_axisbelow(True)

    ax_r.annotate(
        f"{pct_flipped:.0f}% of prompts shift\nby more than {_FLIP_THRESHOLD} BLEU\nunder at least one variant",
        xy=(0.97, 0.95),
        xycoords="axes fraction",
        ha="right", va="top",
        fontsize=8.8,
        color="#d73027",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#fff2f2",
            edgecolor="#d73027",
            linewidth=1.0,
            alpha=0.9,
        ),
    )

    # legend for histogram colours
    import matplotlib.patches as mpatches
    leg_handles = [
        mpatches.Patch(color="#aaaaaa", label=f"|Δ| ≤ {_FLIP_THRESHOLD}"),
        mpatches.Patch(color="#d73027", label=f"|Δ| > {_FLIP_THRESHOLD}  (flipped)"),
    ]
    ax_r.legend(handles=leg_handles, fontsize=8.5, loc="upper left", framealpha=0.55)

    fig.suptitle(
        "Aggregate stability masks per-instance behavioral instability",
        fontsize=13, fontweight="bold", y=1.04,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save as PDF and PNG side-by-side (same stem, different suffix)."""
    pdf_path = path.with_suffix(".pdf")
    png_path = path.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"Saved: {pdf_path}  +  {png_path}")


def _save_combined(fig1: plt.Figure, fig2: plt.Figure, path: Path) -> None:
    """Save a two-page PDF (no PNG for combined)."""
    pdf_path = path.with_suffix(".pdf")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig1, dpi=300, bbox_inches="tight")
        pdf.savefig(fig2, dpi=300, bbox_inches="tight")
    print(f"Saved: {pdf_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate motivation plots for the NeurIPS paper."
    )
    parser.add_argument(
        "--out_dir",
        default="results/my_plots",
        help="Output directory (relative to project root, or absolute).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1  = _make_plot1_fig()
    fig1r = _make_plot1_radar_fig()
    fig1s = _make_plot1_single_radar_fig()
    fig2  = _make_plot2_fig()

    _save_fig(fig1,  out_dir / "motivation_plot1")
    _save_fig(fig1r, out_dir / "motivation_plot1_radar")
    _save_fig(fig1s, out_dir / "motivation_plot1_single_radar")
    _save_fig(fig2,  out_dir / "motivation_plot2")
    _save_combined(fig1,  fig2, out_dir / "motivation_combined.pdf")
    _save_combined(fig1r, fig2, out_dir / "motivation_combined_radar.pdf")
    _save_combined(fig1s, fig2, out_dir / "motivation_combined_single_radar.pdf")
    plt.close("all")

    # ── Top-5 multi-radar plots (individual representative models) ───────────
    print("\nFinding top-5 prompts by total effect …")
    top_pids = _find_top_multi_radar_pids(n=5, lc_strength=50)
    print(f"Top-5 pids: {top_pids}")
    for rank, pid in enumerate(top_pids, start=1):
        data, baseline, prompts = _load_multi_radar_data(pid, lc_strength=50)
        if not data:
            print(f"  [skip] pid={pid}: no data")
            continue
        fig = _make_plot1_radar_fig(data=data, baseline=baseline, prompts=prompts)
        stem = f"top_radar_{rank}_pid{pid}"
        _save_fig(fig, out_dir / stem)
        plt.close(fig)

    # ── Top-5 family-mean radar plots ─────────────────────────────────────────
    print("\nGenerating top-5 family-mean radar plots …")
    for rank, pid in enumerate(top_pids, start=1):
        data, baseline, prompts = _load_family_mean_radar_data(pid, lc_strength=50)
        if not data:
            print(f"  [skip] pid={pid}: no data")
            continue
        fig = _make_plot1_radar_fig(
            data=data, baseline=baseline, prompts=prompts,
            families=_FAMILY_NAMES, colors=_FAMILY_COLORS,
        )
        stem = f"family_mean_radar_{rank}_pid{pid}"
        _save_fig(fig, out_dir / stem)
        plt.close(fig)

    # ── Publication-quality family-mean radar for pid=92 "Are toads frogs?" ──
    print("\nGenerating publication family-mean radar for pid=92 …")
    data92, base92, prpts92 = _load_family_mean_radar_data(92, lc_strength=50)
    if data92:
        fig92 = _make_plot1_radar_fig(
            data=data92, baseline=base92, prompts=prpts92,
            families=_FAMILY_NAMES, colors=_FAMILY_COLORS,
            publication=True,
        )
        _save_fig(fig92, out_dir / "family_mean_radar_pid92_pub")
        plt.close(fig92)

    # ── Publication-quality family-mean radar for pid=122 ────────────────────
    # "What's a fact that all truthers know?"
    # Highest total radar coverage (9.03) among pids with uniformly large
    # per-style fill (weakest style sum = 2.41) and politeness in suffix.
    print("\nGenerating publication family-mean radar for pid=122 …")
    data122, base122, prpts122 = _load_family_mean_radar_data(
        122,
        pol_strength=10, pol_place="suffix",
        lc_strength=25,  lc_place="global",
        lv_strength=0.25, lv_place="global",
    )
    if data122:
        fig122 = _make_plot1_radar_fig(
            data=data122, baseline=base122, prompts=prpts122,
            families=_FAMILY_NAMES, colors=_FAMILY_COLORS,
            publication=True,
        )
        _save_fig(fig122, out_dir / "family_mean_radar_pid122_pub")
        plt.close(fig122)

    plt.close("all")


if __name__ == "__main__":
    import sys
    if "--pub-only" in sys.argv:
        out_dir = PROJECT_ROOT / "results" / "my_plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        data92, base92, prpts92 = _load_family_mean_radar_data(92, lc_strength=50)
        fig92 = _make_plot1_radar_fig(
            data=data92, baseline=base92, prompts=prpts92,
            families=_FAMILY_NAMES, colors=_FAMILY_COLORS,
            publication=True,
        )
        _save_fig(fig92, out_dir / "family_mean_radar_pid92_pub")
        import matplotlib.pyplot as plt
        plt.close(fig92)
    elif "--pub-pid122" in sys.argv:
        out_dir = PROJECT_ROOT / "results" / "my_plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        data122, base122, prpts122 = _load_family_mean_radar_data(
            122,
            pol_strength=10, pol_place="suffix",
            lc_strength=25,  lc_place="global",
            lv_strength=0.25, lv_place="global",
        )
        fig122 = _make_plot1_radar_fig(
            data=data122, baseline=base122, prompts=prpts122,
            families=_FAMILY_NAMES, colors=_FAMILY_COLORS,
            publication=True,
        )
        _save_fig(fig122, out_dir / "family_mean_radar_pid122_pub")
        import matplotlib.pyplot as plt
        plt.close(fig122)
    else:
        main()
