# utils/significance_test.py
"""
Pairwise statistical significance testing for styled-prompt experiments.

For each metric:
  - Computes mean ± std of metric values across strength levels per (model, placement).
  - Performs pairwise paired t-tests between:
      A. All model pairs (placement fixed)   → robustness comparison
      B. All placement pairs (model fixed)   → sensitivity comparison

Metric directions
-----------------
Metric                  Direction   Notes
----------------------  ----------  ----------------------------------------
activation_similarity   max         higher = more stable activations
silhouette              max         higher = better cluster separation
bleu                    max         higher = more similar to original output
bertscore_*             max         higher = better semantic preservation
delta_log_prob          abs_min     closer to 0 = less perturbed likelihood
entropy_shift           min         smaller shift = more stable distribution
jsd_drift               min         smaller divergence = more stable
mirroring_rate*         max         higher = model mirrors user style more
asr                     min         lower = safer (fewer attacks succeed)
refusal_stability       max         higher = more consistent refusals
reasoning_similarity    max         higher = more stable reasoning

Inputs
------
One or more CSV files with columns: model, place, strength, <metric ...>
Typically: combined_means_by_model_place_strength.csv or summary.csv files
produced by experiment scripts.

Outputs (saved to --out_dir)
----------------------------
model_comparisons.csv        All pairwise model comparisons per (metric, place)
place_comparisons.csv        All pairwise place comparisons per (metric, model)
model_ranking.csv            Best model per (metric, place) with mean ± std ranking
place_sensitivity.csv        Most sensitive placement per (metric, model)

Run
---
  python utils/significance_test.py \\
      --runs results/politeness/run_multi_truthful_qa_20260221_112112/plots_metrics/combined_means_by_model_place_strength.csv   results/politeness/run_multi_truthful_qa_20260221_145354/plots_metrics/combined_means_by_model_place_strength.csv   results/politeness/run_multi_truthful_qa_20260222_113341/plots_metrics/combined_means_by_model_place_strength.csv   results/politeness/run_multi_truthful_qa_20260222_132615/plots_metrics/combined_means_by_model_place_strength.csv   results/politeness/run_multi_truthful_qa_20260222_150512/plots_metrics/combined_means_by_model_place_strength.csv   results/politeness/run_multi_truthful_qa_20260223_120916/plots_metrics/combined_means_by_model_place_strength.csv
 \\
      --out_dir results/significance/politeness \\
      --alpha 0.05

  # Multiple run directories (merged):
  python utils/significance_test.py \\
      --runs results/combined_plots/polite/combined_means_by_model_place_strength.csv \\
      --out_dir results/significance/polite

      python utils/significance_test.py --prompt_level results/politeness/run_multi_truthful_qa_20260221_112112/full_results_all_models.csv   results/politeness/run_multi_truthful_qa_20260221_145354/full_results_all_models.csv   results/politeness/run_multi_truthful_qa_20260222_113341/full_results_all_models.csv   results/politeness/run_multi_truthful_qa_20260222_132615/full_results_all_models.csv   results/politeness/run_multi_truthful_qa_20260222_150512/plots_metrics/full_results_all_models.csv   results/politeness/run_multi_truthful_qa_20260223_120916/full_results_all_models.csv --out_dir results/significance/prompt_level  --alpha 0.05 --binary_agg mean
"""
import argparse
import os
import sys
import warnings
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Metric direction registry
# ============================================================

# "max"     → higher mean is better
# "min"     → lower mean is better
# "abs_min" → value closer to 0 is better (uses |value| for comparisons)
METRIC_DIRECTION: Dict[str, str] = {
    "activation_similarity":  "max",
    "silhouette":             "max",
    "bleu":                   "max",
    "bertscore_response":     "max",
    "bertscore_prompt":       "max",
    "delta_log_prob":         "abs_min",
    "entropy_shift":          "min",
    "jsd_drift":              "min",
    "mirroring_rate":         "max",
    "mirroring_rate_batch":   "max",
    "asr":                    "max",  # higher ASR = more attacks succeed = less safe
    "asr_judged":             "max",
    "refusal_stability":      "max",
    "reasoning_similarity":   "max",
}

# Columns that are never metrics
_KEY_COLS = {"model", "place", "strength", "run_dir",
             "prompt_id", "category", "prompt_orig", "prompt_pert",
             "n_alpaca", "n_harmbench", "asr_judged_n", "asr_complied_n",
             # mirroring raw columns are converted to `mirroring` (0/1) in preprocessing
             "mirroring_rate_batch", "mirroring_judge_raw", "mirroring_verdict",
             "mirroring_debug",
             # response text columns (main runs)
             "response_orig", "response_pert",
             # CoT runs (results_cleaned.csv)
             "style", "question_original", "prompt_original_cot", "prompt_styled_cot",
             "response_original", "response_styled", "cot_judge_raw", "cot_judge_attempts"}


# ============================================================
# Data loading
# ============================================================

def load_runs(paths: List[str]) -> pd.DataFrame:
    """
    Accept CSV files or run directories.
    Returns a merged DataFrame with at least: model, place, strength, <metrics>.
    """
    dfs = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isfile(p) and p.endswith(".csv"):
            dfs.append(pd.read_csv(p))
        elif os.path.isdir(p):
            # Try common filenames
            for fname in (
                "combined_means_by_model_place_strength.csv",
                "summary.csv",
                "full_results_all_models.csv",
            ):
                fp = os.path.join(p, fname)
                if os.path.isfile(fp):
                    dfs.append(pd.read_csv(fp))
                    break
            else:
                # Last resort: any CSV in the directory
                for fname in sorted(os.listdir(p)):
                    if fname.endswith(".csv"):
                        dfs.append(pd.read_csv(os.path.join(p, fname)))
                        break
        else:
            warnings.warn(f"Skipping unresolvable input: {p}")
    if not dfs:
        raise ValueError("No CSV data could be loaded from the provided inputs.")
    return pd.concat(dfs, ignore_index=True)


def infer_metric_cols(df: pd.DataFrame) -> List[str]:
    """Return columns that look like numeric metrics (not key columns)."""
    candidates = [c for c in df.columns if c not in _KEY_COLS]
    out = []
    for c in candidates:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            out.append(c)
    return out


def direction_of(metric: str) -> str:
    """Return direction for a metric. Falls back to 'max' for unknowns."""
    for key, d in METRIC_DIRECTION.items():
        if key in metric.lower():
            return d
    return "max"


# ============================================================
# Core: aggregate across strengths → mean ± std per (model, place)
# ============================================================

def _apply_direction_transform(series: pd.Series, direction: str) -> pd.Series:
    """For abs_min metrics, return |values|; otherwise return as-is."""
    if direction == "abs_min":
        return series.abs()
    return series


def _better_group(mean1: float, mean2: float, direction: str) -> int:
    """Return 1 if group1 is better, 2 if group2, 0 if equal."""
    if direction in ("max",):
        if mean1 > mean2:
            return 1
        elif mean2 > mean1:
            return 2
    elif direction in ("min", "abs_min"):
        if mean1 < mean2:
            return 1
        elif mean2 < mean1:
            return 2
    return 0


def aggregate_per_model_place(
        df: pd.DataFrame,
        metric: str,
        direction: str,
) -> pd.DataFrame:
    """
    For each (model, place): collect metric values across all strengths,
    then compute mean ± std.

    Returns DataFrame with columns:
        model, place, mean, std, n_strengths, values (list)
    """
    d = df[["model", "place", "strength", metric]].copy()
    d[metric] = _apply_direction_transform(
        pd.to_numeric(d[metric], errors="coerce"), direction
    )
    d = d.dropna(subset=[metric])

    rows = []
    for (model, place), g in d.groupby(["model", "place"], sort=True):
        vals = g.sort_values("strength")[metric].to_numpy(dtype=float)
        rows.append({
            "model":      model,
            "place":      place,
            "mean":       float(np.nanmean(vals)),
            "std":        float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_strengths": int(len(vals)),
            "_values":    vals,        # kept for t-test; dropped before saving
        })
    return pd.DataFrame(rows)


# ============================================================
# T-test helpers
# ============================================================

def _paired_ttest(
        v1: np.ndarray,
        v2: np.ndarray,
) -> Tuple[float, float]:
    """
    Paired t-test on two equal-length arrays (one value per strength level).
    Falls back to independent t-test if lengths differ.
    Returns (t_stat, p_value).
    """
    v1 = v1[np.isfinite(v1)]
    v2 = v2[np.isfinite(v2)]
    if len(v1) < 2 or len(v2) < 2:
        return float("nan"), float("nan")
    if len(v1) == len(v2):
        result = stats.ttest_rel(v1, v2)
    else:
        result = stats.ttest_ind(v1, v2, equal_var=False)
    return float(result.statistic), float(result.pvalue)


# ============================================================
# A. Model comparison (placement fixed)
# ============================================================

def compare_models(
        df: pd.DataFrame,
        metric: str,
        alpha: float = 0.05,
) -> pd.DataFrame:
    """
    For each placement, compare every pair of models using a paired t-test
    across strength levels.

    Returns DataFrame with columns:
        metric, place, model_1, model_2,
        mean_1, std_1, mean_2, std_2,
        t_stat, p_value, significant, better_model, direction
    """
    direction = direction_of(metric)
    agg = aggregate_per_model_place(df, metric, direction)
    if agg.empty:
        return pd.DataFrame()

    rows = []
    for place, grp in agg.groupby("place", sort=True):
        models = grp["model"].tolist()
        for m1, m2 in combinations(models, 2):
            r1 = grp[grp["model"] == m1].iloc[0]
            r2 = grp[grp["model"] == m2].iloc[0]
            t, p = _paired_ttest(r1["_values"], r2["_values"])
            sig = (not np.isnan(p)) and (p < alpha)
            better_idx = _better_group(r1["mean"], r2["mean"], direction) if sig else 0
            better = [None, m1, m2][better_idx]
            rows.append({
                "metric":       metric,
                "place":        place,
                "model_1":      m1,
                "model_2":      m2,
                "mean_1":       round(r1["mean"], 6),
                "std_1":        round(r1["std"],  6),
                "mean_2":       round(r2["mean"], 6),
                "std_2":        round(r2["std"],  6),
                "n_strengths":  r1["n_strengths"],
                "t_stat":       round(t, 4) if np.isfinite(t) else None,
                "p_value":      round(p, 6) if np.isfinite(p) else None,
                "significant":  sig,
                "better_model": better,
                "direction":    direction,
            })
    return pd.DataFrame(rows)


# ============================================================
# B. Placement comparison (model fixed)
# ============================================================

def compare_places(
        df: pd.DataFrame,
        metric: str,
        alpha: float = 0.05,
) -> pd.DataFrame:
    """
    For each model, compare every pair of placements using a paired t-test
    across strength levels.

    Also computes std_across_strengths per placement to rank sensitivity
    (the placement with the highest std is the most sensitive to strength variation).

    Returns DataFrame with columns:
        metric, model, place_1, place_2,
        mean_1, std_1, mean_2, std_2,
        t_stat, p_value, significant, more_sensitive_place, direction
    """
    direction = direction_of(metric)
    agg = aggregate_per_model_place(df, metric, direction)
    if agg.empty:
        return pd.DataFrame()

    rows = []
    for model, grp in agg.groupby("model", sort=True):
        places = grp["place"].tolist()
        for p1, p2 in combinations(places, 2):
            r1 = grp[grp["place"] == p1].iloc[0]
            r2 = grp[grp["place"] == p2].iloc[0]
            t, p = _paired_ttest(r1["_values"], r2["_values"])
            sig = (not np.isnan(p)) and (p < alpha)
            # More sensitive = higher std across strengths
            more_sensitive = p1 if r1["std"] >= r2["std"] else p2
            rows.append({
                "metric":             metric,
                "model":              model,
                "place_1":            p1,
                "place_2":            p2,
                "mean_1":             round(r1["mean"], 6),
                "std_1":              round(r1["std"],  6),
                "mean_2":             round(r2["mean"], 6),
                "std_2":              round(r2["std"],  6),
                "n_strengths":        r1["n_strengths"],
                "t_stat":             round(t, 4) if np.isfinite(t) else None,
                "p_value":            round(p, 6) if np.isfinite(p) else None,
                "significant":        sig,
                "more_sensitive_place": more_sensitive,
                "direction":          direction,
            })
    return pd.DataFrame(rows)


# ============================================================
# Summary tables
# ============================================================

def model_ranking(
        df: pd.DataFrame,
        metric: str,
) -> pd.DataFrame:
    """
    For each placement, rank all models by their mean metric value
    (adjusted for direction).  Tied ranks broken by std (lower std = more robust).

    Returns DataFrame with columns:
        metric, place, rank, model, mean, std, n_strengths, direction
    """
    direction = direction_of(metric)
    agg = aggregate_per_model_place(df, metric, direction)
    if agg.empty:
        return pd.DataFrame()

    rows = []
    for place, grp in agg.groupby("place", sort=True):
        ascending = direction in ("min", "abs_min")
        ranked = (
            grp.sort_values("mean", ascending=ascending)
               .reset_index(drop=True)
        )
        for rank, (_, r) in enumerate(ranked.iterrows(), start=1):
            rows.append({
                "metric":      metric,
                "place":       place,
                "rank":        rank,
                "model":       r["model"],
                "mean":        round(r["mean"], 6),
                "std":         round(r["std"],  6),
                "n_strengths": r["n_strengths"],
                "direction":   direction,
            })
    return pd.DataFrame(rows)


def place_sensitivity_ranking(
        df: pd.DataFrame,
        metric: str,
) -> pd.DataFrame:
    """
    For each model, rank placements by their std across strength levels.
    Higher std = more sensitive to stylistic variation.

    Returns DataFrame with columns:
        metric, model, sensitivity_rank, place, mean, std, n_strengths, direction
    """
    direction = direction_of(metric)
    agg = aggregate_per_model_place(df, metric, direction)
    if agg.empty:
        return pd.DataFrame()

    rows = []
    for model, grp in agg.groupby("model", sort=True):
        ranked = grp.sort_values("std", ascending=False).reset_index(drop=True)
        for rank, (_, r) in enumerate(ranked.iterrows(), start=1):
            rows.append({
                "metric":           metric,
                "model":            model,
                "sensitivity_rank": rank,
                "place":            r["place"],
                "mean":             round(r["mean"], 6),
                "std":              round(r["std"],  6),
                "n_strengths":      r["n_strengths"],
                "direction":        direction,
            })
    return pd.DataFrame(rows)


# ============================================================
# Main runner
# ============================================================

def run(
        *,
        runs: List[str],
        out_dir: str,
        alpha: float,
        metrics: Optional[List[str]],
        models: Optional[List[str]],
        places: Optional[List[str]],
):
    os.makedirs(out_dir, exist_ok=True)

    print("[LOAD] Reading input files …")
    df = load_runs(runs)
    print(f"       {len(df):,} rows loaded, columns: {list(df.columns)}")

    # Optional filters
    if models and "model" in df.columns:
        df = df[df["model"].isin(models)]
    if places and "place" in df.columns:
        df = df[df["place"].isin(places)]

    if df.empty:
        raise SystemExit("[ERROR] No data after filtering.")

    all_metrics = infer_metric_cols(df)
    if metrics:
        all_metrics = [m for m in all_metrics if m in set(metrics)]
    if not all_metrics:
        raise SystemExit("[ERROR] No metric columns found.")
    print(f"[METRICS] {all_metrics}\n")

    model_cmp_parts   = []
    place_cmp_parts   = []
    model_rank_parts  = []
    place_sens_parts  = []

    for metric in all_metrics:
        print(f"[TEST] {metric}")
        model_cmp_parts.append(compare_models(df, metric, alpha=alpha))
        place_cmp_parts.append(compare_places(df, metric, alpha=alpha))
        model_rank_parts.append(model_ranking(df, metric))
        place_sens_parts.append(place_sensitivity_ranking(df, metric))

    def _save(parts, fname):
        combined = pd.concat([p for p in parts if not p.empty], ignore_index=True)
        path = os.path.join(out_dir, fname)
        combined.to_csv(path, index=False)
        print(f"[SAVE] {path}  ({len(combined):,} rows)")
        return combined

    _save(model_cmp_parts,  "model_comparisons.csv")
    _save(place_cmp_parts,  "place_comparisons.csv")
    _save(model_rank_parts, "model_ranking.csv")
    _save(place_sens_parts, "place_sensitivity.csv")

    # ── Quick console summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY — significant model pairs (p < {:.2f})".format(alpha))
    print("=" * 60)
    mc = pd.concat([p for p in model_cmp_parts if not p.empty], ignore_index=True)
    sig_mc = mc[mc["significant"] == True]
    if sig_mc.empty:
        print("  No significant model differences found.")
    else:
        for (metric, place), g in sig_mc.groupby(["metric", "place"]):
            pairs = ", ".join(
                f"{r.model_1} > {r.model_2}" if r.better_model == r.model_1
                else f"{r.model_2} > {r.model_1}"
                for _, r in g.iterrows()
            )
            print(f"  {metric} | {place}: {pairs}")

    print("\n" + "=" * 60)
    print("SUMMARY — most sensitive placements (highest std, p < {:.2f})".format(alpha))
    print("=" * 60)
    pc = pd.concat([p for p in place_cmp_parts if not p.empty], ignore_index=True)
    sig_pc = pc[pc["significant"] == True]
    if sig_pc.empty:
        print("  No significant placement differences found.")
    else:
        ps = pd.concat([p for p in place_sens_parts if not p.empty], ignore_index=True)
        top = ps[ps["sensitivity_rank"] == 1][["metric", "model", "place", "std"]]
        for _, r in top.iterrows():
            print(f"  {r['metric']} | {r['model']}: most sensitive = {r['place']} (std={r['std']:.4f})")

    print(f"\n[DONE] Results saved to: {out_dir}")


# ============================================================
# Prompt-level baseline-vs-perturbed analysis
# ============================================================

# Test strategy per metric type:
#
#   "constant_baseline" → one-sample t-test: H0: mean(perturbed) = BASELINE_CONSTANT[metric]
#                         perturbed[i] = mean(metric, strength≠0) per prompt
#
#   "paired"            → two-sided Wilcoxon signed-rank on (baseline[i], perturbed[i]) pairs
#
#   "binary"            → McNemar's test (exact=True).
#                         baseline[i]  = value at strength==0
#                         perturbed[i] = 1 if ANY strength≠0 fires, else 0
#
#   "descriptive_only"  → report mean±std only, no significance test
#
# Raw column aliases handled in preprocessing (before this registry is used):
#   unsafe_score       → kept as-is  (ASR, binary)
#   mirroring_judge_raw → converted to `mirroring` (MR, binary: "yes"→1, "no"→0)

BASELINE_CONSTANT: Dict[str, float] = {
    "activation_similarity": 1.0,
    "bleu":                  100.0,
    "bertscore_response":    1.0,
    "bertscore_prompt":      1.0,
    "delta_log_prob":        0.0,
    "entropy_shift":         0.0,
}

METRIC_PROMPT_TYPE: Dict[str, str] = {
    "activation_similarity":  "constant_baseline",
    "silhouette":             "descriptive_only",   # report mean±std, no test
    "bleu":                   "constant_baseline",
    "bertscore_response":     "constant_baseline",
    "bertscore_prompt":       "constant_baseline",
    "delta_log_prob":         "constant_baseline",
    "entropy_shift":          "constant_baseline",
    "jsd_drift":              "paired",
    "unsafe_score":           "binary",             # ASR: McNemar, perturbed=any
    "mirroring":              "binary",             # MR after conversion from mirroring_judge_raw
    "mirroring_rate_batch":   "binary",
    "mirroring_rate":         "binary",
    "asr":                    "binary",
    "refusal_stability":      "paired",
    "reasoning_similarity":   "paired",
    "truthfulqa_correctness": "binary",
    "length_ratio":           "paired",
    "cot_similarity":         "paired",
    "cot_steps":              "paired_ttest",   # CoT length: paired t-test
    "cot_correct":            "binary",         # CoT correctness: McNemar, abs_max selection
}


def _prompt_metric_type(metric_name: str) -> str:
    """Return 'constant_baseline', 'paired', 'paired_ttest', 'binary', or 'descriptive_only'."""
    for key, mtype in METRIC_PROMPT_TYPE.items():
        if key in metric_name:
            return mtype
    return "paired"


def _baseline_constant(metric_name: str) -> Optional[float]:
    """Return the known constant baseline value for metric_name, or None."""
    for key, val in BASELINE_CONSTANT.items():
        if key in metric_name:
            return val
    return None


# ── Peak-effect strength selection ───────────────────────────────────────────
#
# Rule:
#   "min"     — s* = argmin_s mean(metric, s≠0)   (lower = stronger perturbation)
#   "max"     — s* = argmax_s mean(metric, s≠0)   (higher = stronger perturbation)
#   "abs_max" — s* = argmax_s |mean(metric, s≠0)| (deviation from 0 = stronger)

METRIC_SELECTION_RULE: Dict[str, str] = {
    "activation_similarity": "min",
    "silhouette":            "min",
    "bleu":                  "min",
    "bertscore":             "min",
    "refusal_stability":     "min",
    "reasoning_similarity":  "min",
    "unsafe_score":          "max",
    "asr":                   "max",
    "mirroring":             "max",
    "mirroring_rate":        "max",
    "cot_steps":             "max",
    "delta_log_prob":        "abs_max",
    "entropy_shift":         "abs_max",
    "cot_steps":             "abs_max",   # CoT length: largest absolute deviation
    "cot_correct":           "abs_max",    # CoT correctness: largest absolute deviation
}


def _metric_selection_rule(metric_name: str) -> str:
    """Return 'min', 'max', or 'abs_max' for metric_name. Defaults to 'max'."""
    for key, rule in METRIC_SELECTION_RULE.items():
        if key in metric_name:
            return rule
    return "max"


def _select_best_strength(
        group: pd.DataFrame,
        metric_col: str,
        rule: str,
        baseline_strength=0,
) -> Optional[float]:
    """
    Choose s* from non-baseline strengths using the aggregate mean curve.
    Returns the selected strength value, or None if no data.
    """
    non_base = group[group["strength"] != baseline_strength]
    if non_base.empty:
        return None
    means = non_base.groupby("strength")[metric_col].mean().dropna()
    if means.empty:
        return None
    if rule == "min":
        return means.idxmin()
    elif rule == "max":
        return means.idxmax()
    else:  # abs_max
        return means.abs().idxmax()


def _get_paired_at_strength(
        group: pd.DataFrame,
        metric_col: str,
        s_star,
        baseline_strength=0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return aligned (B_i, P_i) arrays where:
      B_i = metric(prompt_i, strength == baseline_strength)
      P_i = metric(prompt_i, strength == s_star)
    Only prompts present in both sides are returned.
    """
    base = group[group["strength"] == baseline_strength].groupby("prompt_id")[metric_col].mean()
    pert = group[group["strength"] == s_star].groupby("prompt_id")[metric_col].mean()
    common = base.index.intersection(pert.index)
    b = base.loc[common].values.astype(float)
    p = pert.loc[common].values.astype(float)
    mask = np.isfinite(b) & np.isfinite(p)
    return b[mask], p[mask]


def _get_perturbed_at_strength(
        group: pd.DataFrame,
        metric_col: str,
        s_star,
) -> np.ndarray:
    """Return P_i = metric(prompt_i, s=s_star), NaNs dropped."""
    vals = group[group["strength"] == s_star][metric_col].values.astype(float)
    return vals[np.isfinite(vals)]


_PVAL_FLOOR = np.finfo(float).tiny  # ~2.2e-308: floor to avoid exact 0.0 from underflow


def _floor_pvalue(p: float) -> float:
    """Replace exact 0.0 (floating-point underflow) with the smallest positive float."""
    if np.isfinite(p) and p == 0.0:
        return _PVAL_FLOOR
    return p


def _run_one_sample_ttest(v: np.ndarray, mu: float) -> Tuple[float, str]:
    """One-sample two-sided t-test: H0: mean(v) = mu. Returns (p_value, 'ttest_1samp')."""
    if len(v) < 5:
        return float("nan"), "ttest_1samp"
    try:
        result = stats.ttest_1samp(v, popmean=mu, alternative="two-sided")
        return _floor_pvalue(float(result.pvalue)), "ttest_1samp"
    except Exception:
        return float("nan"), "ttest_1samp"


def _run_wilcoxon(b: np.ndarray, p: np.ndarray) -> Tuple[float, str]:
    """Two-sided Wilcoxon signed-rank test on paired arrays. Returns (p_value, 'wilcoxon')."""
    if len(b) < 5:
        return float("nan"), "wilcoxon"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = stats.wilcoxon(b, p, alternative="two-sided", zero_method="wilcox")
            return _floor_pvalue(float(result.pvalue)), "wilcoxon"
        except ValueError:
            return float("nan"), "wilcoxon"


def _run_mcnemar(b: np.ndarray, p: np.ndarray) -> Tuple[float, str]:
    """
    McNemar's exact test on paired binary (0/1) arrays.

    Contingency table:
        [[a, b],   a = (0→0), b = (0→1)
         [c, d]]   c = (1→0), d = (1→1)

    Always uses exact=True as specified.
    Returns (p_value, 'mcnemar').
    """
    if len(b) < 5:
        return float("nan"), "mcnemar"
    b_bin = (b >= 0.5).astype(int)
    p_bin = (p >= 0.5).astype(int)
    a = int(np.sum((b_bin == 0) & (p_bin == 0)))
    bv = int(np.sum((b_bin == 0) & (p_bin == 1)))   # 0→1
    c = int(np.sum((b_bin == 1) & (p_bin == 0)))    # 1→0
    d = int(np.sum((b_bin == 1) & (p_bin == 1)))
    if bv + c == 0:
        return 1.0, "mcnemar"
    table = np.array([[a, bv], [c, d]])
    try:
        result = mcnemar_test(table, exact=True)
        return _floor_pvalue(float(result.pvalue)), "mcnemar"
    except Exception:
        return float("nan"), "mcnemar"


def _preprocess_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw safety/mirroring columns to standardised binary columns:
      - mirroring_judge_raw ("yes"/"no") → mirroring (1/0)  [politeness data]
      - mirroring_verdict   ("YES"/"NO")  → mirroring (1/0)  [spacing/punctuation/letter_case]
      - unsafe_score kept as-is (already 0/1)
    mirroring_judge_raw takes priority if both columns are present.
    The original raw columns are dropped after conversion.
    """
    df = df.copy()
    if "mirroring_judge_raw" in df.columns:
        df["mirroring"] = (
            df["mirroring_judge_raw"].astype(str).str.strip().str.lower() == "yes"
        ).astype(int)
        df = df.drop(columns=["mirroring_judge_raw"])
    elif "mirroring_verdict" in df.columns:
        df["mirroring"] = (
            df["mirroring_verdict"].astype(str).str.strip().str.upper() == "YES"
        ).astype(int)
        df = df.drop(columns=["mirroring_verdict"])
    return df


def run_baseline_vs_perturbed(
        df: pd.DataFrame,
        *,
        alpha: float = 0.05,
        baseline_strength: float = 0,
) -> pd.DataFrame:
    """
    Peak-effect significance analysis.

    For each (metric, model, place):
      1. Select s* — the single non-baseline strength with the strongest aggregate effect
         (argmin/argmax/argmax|mean| depending on METRIC_SELECTION_RULE).
      2. Extract P_i = metric(prompt_i, s=s*) and B_i = metric(prompt_i, s=baseline_strength).
      3. Run the appropriate test:
           constant_baseline → one-sample t-test vs. known constant (B_i constant by design)
           binary            → McNemar exact test on (B_i, P_i) pairs at s*
           paired_ttest      → paired t-test on (B_i, P_i)  [CoT length]
           paired            → Wilcoxon signed-rank on (B_i, P_i)
      4. Apply BH FDR correction across all (model, place) pairs within each metric.

    Output columns:
        metric, model, place, selected_strength,
        baseline_value, perturbed_mean, perturbed_std, delta,
        test_used, p_value_raw, p_value_corrected, significant, n_prompts, note
    """
    df = _preprocess_binary_columns(df)

    metric_cols = infer_metric_cols(df)
    required = {"prompt_id", "model", "place", "strength"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    def _row(metric, model, place, s_star, note, **kw):
        base = {
            "metric": metric, "model": model, "place": place,
            "selected_strength": s_star,
            "baseline_value": float("nan"),
            "perturbed_mean": float("nan"), "perturbed_std": float("nan"),
            "delta": float("nan"), "test_used": "none",
            "p_value_raw": float("nan"), "p_value_corrected": float("nan"),
            "significant": False, "n_prompts": 0, "note": note,
        }
        base.update(kw)
        return base

    rows = []
    for metric in metric_cols:
        sub = df[["prompt_id", "model", "place", "strength", metric]].copy()
        metric_type  = _prompt_metric_type(metric)
        sel_rule     = _metric_selection_rule(metric)
        const        = _baseline_constant(metric)

        for (model, place), grp in sub.groupby(["model", "place"], sort=True):

            s_star = _select_best_strength(grp, metric, sel_rule,
                                           baseline_strength=baseline_strength)
            if s_star is None:
                rows.append(_row(metric, model, place, None, "no_nonzero_strengths"))
                continue

            if metric_type == "constant_baseline":
                # P_i = single value at s*; compare vs known constant
                p_vals = _get_perturbed_at_strength(grp, metric, s_star)
                n = len(p_vals)
                if n < 5:
                    rows.append(_row(metric, model, place, s_star, "insufficient_data",
                                     baseline_value=const, n_prompts=n))
                    continue
                pval, test_used = _run_one_sample_ttest(p_vals, mu=const)
                rows.append(_row(metric, model, place, s_star, "",
                    baseline_value  = const,
                    perturbed_mean  = float(np.mean(p_vals)),
                    perturbed_std   = float(np.std(p_vals, ddof=1)),
                    delta           = float(np.mean(p_vals) - const),
                    test_used       = test_used,
                    p_value_raw     = pval,
                    n_prompts       = n,
                ))

            elif metric_type == "binary":
                b, p = _get_paired_at_strength(grp, metric, s_star,
                                               baseline_strength=baseline_strength)
                n = len(b)
                if n < 5:
                    rows.append(_row(metric, model, place, s_star, "insufficient_data",
                                     n_prompts=n))
                    continue
                pval, test_used = _run_mcnemar(b, p)
                rows.append(_row(metric, model, place, s_star, "",
                    baseline_value  = float(np.mean(b)),
                    perturbed_mean  = float(np.mean(p)),
                    perturbed_std   = float(np.std(p, ddof=1)) if n > 1 else float("nan"),
                    delta           = float(np.mean(p) - np.mean(b)),
                    test_used       = test_used,
                    p_value_raw     = pval,
                    n_prompts       = n,
                ))

            elif metric_type == "paired_ttest":
                b, p = _get_paired_at_strength(grp, metric, s_star,
                                               baseline_strength=baseline_strength)
                n = len(b)
                if n < 5:
                    rows.append(_row(metric, model, place, s_star, "insufficient_data",
                                     n_prompts=n))
                    continue
                pval, test_used = _run_paired_ttest(b, p)
                rows.append(_row(metric, model, place, s_star, "",
                    baseline_value  = float(np.mean(b)),
                    perturbed_mean  = float(np.mean(p)),
                    perturbed_std   = float(np.std(p, ddof=1)),
                    delta           = float(np.mean(p) - np.mean(b)),
                    test_used       = test_used,
                    p_value_raw     = pval,
                    n_prompts       = n,
                ))

            else:  # "paired" — Wilcoxon
                b, p = _get_paired_at_strength(grp, metric, s_star,
                                               baseline_strength=baseline_strength)
                n = len(b)
                if n < 5:
                    rows.append(_row(metric, model, place, s_star, "insufficient_data",
                                     n_prompts=n))
                    continue
                pval, test_used = _run_wilcoxon(b, p)
                rows.append(_row(metric, model, place, s_star, "",
                    baseline_value  = float(np.mean(b)),
                    perturbed_mean  = float(np.mean(p)),
                    perturbed_std   = float(np.std(p, ddof=1)),
                    delta           = float(np.mean(p) - np.mean(b)),
                    test_used       = test_used,
                    p_value_raw     = pval,
                    n_prompts       = n,
                ))

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)

    # BH correction per metric
    corrected_parts = []
    for metric, grp in result_df.groupby("metric"):
        idx = grp.index
        raw = grp["p_value_raw"].values
        valid = np.isfinite(raw)
        p_corr = np.full(len(raw), float("nan"))
        if valid.sum() > 0:
            _, p_adj, _, _ = multipletests(raw[valid], alpha=alpha, method="fdr_bh")
            p_corr[valid] = p_adj
        sig = p_corr < alpha
        grp = grp.copy()
        grp["p_value_corrected"] = p_corr
        grp["significant"] = sig & valid
        corrected_parts.append(grp)

    result_df = pd.concat(corrected_parts, ignore_index=True)
    # Round for readability
    for col in ("baseline_value", "perturbed_mean", "perturbed_std",
                "delta", "p_value_raw", "p_value_corrected"):
        result_df[col] = result_df[col].round(6)

    return result_df


# ============================================================
# inter_vs_imper: per-prompt conditional pairing
# ============================================================

def _get_inter_vs_imper_pairs(
        group: pd.DataFrame,
        metric_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each prompt in `group`, determine which strength is the baseline:
      1. If strength='original' exists for the prompt → use it as baseline,
         then pick the other form (interrogative or imperative) as perturbed.
      2. Otherwise, fall back to '?' detection on the original prompt text:
           - prompt contains '?' → baseline=interrogative, perturbed=imperative
           - prompt has no '?'   → baseline=imperative,    perturbed=interrogative
      3. If no original text column found, default to interrogative as baseline.

    Returns aligned (B_arr, P_arr) arrays (NaNs dropped).
    """
    orig_col = next(
        (c for c in ("prompt_orig", "question_original", "prompt_original")
         if c in group.columns),
        None,
    )

    b_list, p_list = [], []
    for pid, pid_grp in group.groupby("prompt_id"):
        inter = pid_grp[pid_grp["strength"] == "interrogative"][metric_col].dropna()
        imper = pid_grp[pid_grp["strength"] == "imperative"][metric_col].dropna()
        orig  = pid_grp[pid_grp["strength"] == "original"][metric_col].dropna()

        # ── Priority 1: explicit 'original' strength ──────────────────────────
        if not orig.empty:
            base_val = float(orig.mean())
            # perturbed = whichever non-original form is available (prefer both)
            if not inter.empty and not imper.empty:
                # use the form that differs most from original
                inter_val = float(inter.mean())
                imper_val = float(imper.mean())
                pert_val = inter_val if abs(inter_val - base_val) >= abs(imper_val - base_val) else imper_val
            elif not inter.empty:
                pert_val = float(inter.mean())
            elif not imper.empty:
                pert_val = float(imper.mean())
            else:
                continue
            b_list.append(base_val)
            p_list.append(pert_val)
            continue

        # ── Priority 2: '?' detection ─────────────────────────────────────────
        if inter.empty or imper.empty:
            continue

        if orig_col:
            texts = pid_grp[orig_col].dropna().astype(str)
            has_question = texts.str.contains(r"\?").any()
        else:
            has_question = True  # default: treat interrogative as baseline

        if has_question:
            b_list.append(float(inter.mean()))
            p_list.append(float(imper.mean()))
        else:
            b_list.append(float(imper.mean()))
            p_list.append(float(inter.mean()))

    b = np.array(b_list, dtype=float)
    p = np.array(p_list, dtype=float)
    mask = np.isfinite(b) & np.isfinite(p)
    return b[mask], p[mask]


def run_inter_vs_imper_analysis(
        df: pd.DataFrame,
        *,
        alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Significance analysis for inter_vs_imper style.

    Baseline is determined per-prompt (priority order):
      1. strength='original' exists → use as baseline, perturbed = other form
      2. '?' present  → baseline=interrogative, perturbed=imperative
      - '?' absent   → baseline=imperative,    perturbed=interrogative

    Output format is identical to run_baseline_vs_perturbed (selected_strength='conditional').
    """
    df = _preprocess_binary_columns(df)
    metric_cols = infer_metric_cols(df)

    required = {"prompt_id", "model", "place", "strength"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    def _row(metric, model, place, note, **kw):
        base = {
            "metric": metric, "model": model, "place": place,
            "selected_strength": "conditional",
            "baseline_value": float("nan"),
            "perturbed_mean": float("nan"), "perturbed_std": float("nan"),
            "delta": float("nan"), "test_used": "none",
            "p_value_raw": float("nan"), "p_value_corrected": float("nan"),
            "significant": False, "n_prompts": 0, "note": note,
        }
        base.update(kw)
        return base

    # Carry orig-text cols into sub-df so _get_inter_vs_imper_pairs can use them
    orig_cols = [c for c in ("prompt_orig", "question_original", "prompt_original")
                 if c in df.columns]

    rows = []
    for metric in metric_cols:
        keep = ["prompt_id", "model", "place", "strength", metric] + orig_cols
        sub = df[[c for c in keep if c in df.columns]].copy()
        metric_type = _prompt_metric_type(metric)

        for (model, place), grp in sub.groupby(["model", "place"], sort=True):
            b, p = _get_inter_vs_imper_pairs(grp, metric)
            n = len(b)
            if n < 5:
                rows.append(_row(metric, model, place, "insufficient_data", n_prompts=n))
                continue

            if metric_type == "binary":
                pval, test_used = _run_mcnemar(b, p)
            elif metric_type in ("paired_ttest", "constant_baseline"):
                pval, test_used = _run_paired_ttest(b, p)
            else:  # "paired" — Wilcoxon
                pval, test_used = _run_wilcoxon(b, p)

            rows.append(_row(metric, model, place, "",
                baseline_value  = float(np.mean(b)),
                perturbed_mean  = float(np.mean(p)),
                perturbed_std   = float(np.std(p, ddof=1)) if n > 1 else float("nan"),
                delta           = float(np.mean(p) - np.mean(b)),
                test_used       = test_used,
                p_value_raw     = pval,
                n_prompts       = n,
            ))

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)

    # BH correction per metric
    corrected_parts = []
    for metric, grp in result_df.groupby("metric"):
        raw   = grp["p_value_raw"].values
        valid = np.isfinite(raw)
        p_corr = np.full(len(raw), float("nan"))
        if valid.sum() > 0:
            _, p_adj, _, _ = multipletests(raw[valid], alpha=alpha, method="fdr_bh")
            p_corr[valid] = p_adj
        grp = grp.copy()
        grp["p_value_corrected"] = p_corr
        grp["significant"] = (p_corr < alpha) & valid
        corrected_parts.append(grp)

    result_df = pd.concat(corrected_parts, ignore_index=True)
    for col in ("baseline_value", "perturbed_mean", "perturbed_std",
                "delta", "p_value_raw", "p_value_corrected"):
        result_df[col] = result_df[col].round(6)
    return result_df


# ============================================================
# Specialised data loaders
# ============================================================

def load_asr_data(asr_dirs: List[str]) -> pd.DataFrame:
    """
    Load ASR (safety) data from harmbench_judged_<place>_s<strength>.csv files.

    asr_dirs: list of model-level directories, e.g.
        results/politeness_safety/run_.../asr_outputs/L3.1-8B

    Returns DataFrame with columns: model, place, strength, prompt_id, unsafe_score
    """
    parts = []
    for d in asr_dirs:
        if not os.path.isdir(d):
            print(f"  [WARN] ASR dir not found: {d}")
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".csv"):
                continue
            # Accept both naming conventions:
            #   newer: harmbench_judged_<place>_s<strength>.csv
            #   older: asr_<place>_s<strength>.csv
            if not (fname.startswith("harmbench_judged_") or fname.startswith("asr_")):
                continue
            fp = os.path.join(d, fname)
            df = pd.read_csv(fp)
            keep = [c for c in ["model", "place", "strength", "prompt_id", "unsafe_score"]
                    if c in df.columns]
            parts.append(df[keep])
    if not parts:
        return pd.DataFrame()
    result = pd.concat(parts, ignore_index=True)
    print(f"  [ASR]  {len(result):,} rows loaded from {len(asr_dirs)} dir(s)")
    return result


def load_silhouette_data(paths: List[str]) -> pd.DataFrame:
    """
    Load silhouette data from summary.csv files.
    These are aggregate (no prompt_id) — one row per (model, place, strength).

    Returns DataFrame with columns: model, place, strength, silhouette
    """
    parts = []
    for p in paths:
        if not os.path.isfile(p):
            print(f"  [WARN] Silhouette file not found: {p}")
            continue
        df = pd.read_csv(p)
        keep = [c for c in ["model", "place", "strength", "silhouette"] if c in df.columns]
        parts.append(df[keep])
    if not parts:
        return pd.DataFrame()
    result = pd.concat(parts, ignore_index=True)
    print(f"  [SIL]  {len(result):,} rows loaded from {len(paths)} file(s)")
    return result


def load_cot_data(paths: List[str]) -> pd.DataFrame:
    """
    Load CoT reasoning data from results_cleaned.csv or results_with_cot_analysis.csv files.
    Renames problem_id → prompt_id.
    Keeps cot_steps (paired t-test, abs_max) and cot_correct (binary McNemar, max).

    Accepts file paths directly, or directories — in which case results_cleaned.csv
    inside the directory is tried first, then results_with_cot_analysis.csv.

    Returns DataFrame with columns: model, place, strength, prompt_id, cot_steps, cot_correct
    """
    parts = []
    for p in paths:
        fp = p
        if os.path.isdir(p):
            for fname in ("results_cleaned.csv", "results_with_cot_analysis.csv"):
                candidate = os.path.join(p, fname)
                if os.path.isfile(candidate):
                    fp = candidate
                    break
            else:
                print(f"  [WARN] No CoT CSV found in directory: {p}")
                continue
        if not os.path.isfile(fp):
            print(f"  [WARN] CoT file not found: {fp}")
            continue
        df = pd.read_csv(fp)
        if "problem_id" in df.columns:
            df = df.rename(columns={"problem_id": "prompt_id"})
        keep = [c for c in ["model", "place", "strength", "prompt_id",
                             "cot_steps", "cot_correct"]
                if c in df.columns]
        parts.append(df[keep])
    if not parts:
        return pd.DataFrame()
    result = pd.concat(parts, ignore_index=True)
    print(f"  [COT]  {len(result):,} rows loaded from {len(paths)} path(s)")
    return result


def run_silhouette_analysis(df: pd.DataFrame, *, baseline_strength: float = 0) -> pd.DataFrame:
    """
    Silhouette is an aggregate metric (no prompt_id).
    Uses peak-effect selection: s* = argmin mean(silhouette, s≠baseline) per (model, place).
    Reports the silhouette value at s* only. No statistical test.
    """
    rows = []

    for (model, place), grp in df.groupby(["model", "place"], sort=True):
        base_vals = grp[grp["strength"] == baseline_strength]["silhouette"].dropna().values.astype(float)
        baseline  = float(np.mean(base_vals)) if len(base_vals) else float("nan")

        # Select s* = strength with lowest mean silhouette (most disrupted clustering)
        s_star = _select_best_strength(grp, "silhouette", rule="min",
                                       baseline_strength=baseline_strength)
        if s_star is None:
            rows.append({
                "metric": "silhouette", "model": model, "place": place,
                "selected_strength": None,
                "baseline_value": baseline,
                "perturbed_mean": float("nan"), "perturbed_std": float("nan"),
                "delta": float("nan"), "test_used": "none",
                "p_value_raw": float("nan"), "p_value_corrected": float("nan"),
                "significant": False, "n_prompts": 0, "note": "no_nonzero_strengths",
            })
            continue

        pert_vals = grp[grp["strength"] == s_star]["silhouette"].dropna().values.astype(float)
        n = len(pert_vals)
        p_mean = float(np.mean(pert_vals)) if n else float("nan")
        p_std  = float(np.std(pert_vals, ddof=1)) if n > 1 else float("nan")
        delta  = float(p_mean - baseline) if np.isfinite(p_mean) and np.isfinite(baseline) else float("nan")

        rows.append({
            "metric":            "silhouette",
            "model":             model,
            "place":             place,
            "selected_strength": s_star,
            "baseline_value":    baseline,
            "perturbed_mean":    p_mean,
            "perturbed_std":     p_std,
            "delta":             delta,
            "test_used":         "none",
            "p_value_raw":       float("nan"),
            "p_value_corrected": float("nan"),
            "significant":       False,
            "n_prompts":         n,
            "note":              "descriptive_only",
        })
    return pd.DataFrame(rows)


def _run_paired_ttest(b: np.ndarray, p: np.ndarray) -> Tuple[float, str]:
    """
    Paired t-test: H0: mean(P - B) = 0.
    D_i = P_i - B_i; t = mean(D) / (std(D) / sqrt(n)), df = n-1.
    Returns (p_value, 'ttest_paired').
    """
    if len(b) < 5:
        return float("nan"), "ttest_paired"
    try:
        result = stats.ttest_rel(b, p, alternative="two-sided")
        return _floor_pvalue(float(result.pvalue)), "ttest_paired"
    except Exception:
        return float("nan"), "ttest_paired"


def pivot_significance(
        results: pd.DataFrame,
        style: str,
        metric_name: str,
        value_col: str = "significant",
) -> pd.DataFrame:
    """
    Return a pivot table (rows=models, cols=places) of value_col
    for the given style and metric.
    """
    sub = results[(results.get("style", style) == style) & (results["metric"] == metric_name)]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="model", columns="place", values=value_col, aggfunc="first")


def run_prompt_level(
        *,
        prompt_csvs: List[str],
        out_dir: str,
        alpha: float = 0.05,
        baseline_strength: float = 0,
        metrics: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        places: Optional[List[str]] = None,
        asr_dirs: Optional[List[str]] = None,
        silhouette_csvs: Optional[List[str]] = None,
        cot_csvs: Optional[List[str]] = None,
) -> None:
    """
    File-based orchestrator for prompt-level baseline-vs-perturbed analysis.

    Primary data (prompt_csvs): full_results_all_models.csv with prompt_id, model,
        place, strength, and metric columns.

    Additional specialised sources:
      asr_dirs        — model-level dirs containing harmbench_judged_*.csv
                        (e.g. results/politeness_safety/run_.../asr_outputs/L3.1-8B)
      silhouette_csvs — summary.csv files with model, place, strength, silhouette
                        (aggregate, no prompt_id — descriptive only)
      cot_csvs        — results_with_cot_analysis.csv files with problem_id, cot_steps
                        (paired t-test on CoT length)
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Main prompt-level data ────────────────────────────────────────────────
    parts = []
    for path in prompt_csvs:
        print(f"[LOAD] {path}")
        parts.append(pd.read_csv(path))
    df = pd.concat(parts, ignore_index=True)
    if "problem_id" in df.columns and "prompt_id" not in df.columns:
        df = df.rename(columns={"problem_id": "prompt_id"})
    print(f"       {len(df):,} rows total, columns: {list(df.columns)}")

    # ── ASR data ──────────────────────────────────────────────────────────────
    if asr_dirs:
        asr_df = load_asr_data(asr_dirs)
        if not asr_df.empty:
            df = pd.concat([df, asr_df], ignore_index=True)

    # ── CoT data ──────────────────────────────────────────────────────────────
    if cot_csvs:
        cot_df = load_cot_data(cot_csvs)
        if not cot_df.empty:
            df = pd.concat([df, cot_df], ignore_index=True)

    # ── Filters ───────────────────────────────────────────────────────────────
    if models:
        df = df[df["model"].isin(models)]
    if places and "place" in df.columns:
        df = df[df["place"].isin(places)]
    if metrics:
        keep = [c for c in df.columns
                if c in {"prompt_id", "model", "place", "strength"} or c in metrics]
        df = df[keep]

    if df.empty:
        raise SystemExit("[ERROR] No data after filtering.")

    # ── Detect inter_vs_imper and route accordingly ───────────────────────────
    str_strengths = df["strength"].dropna().astype(str).unique()
    is_inter_vs_imper = any(s in ("interrogative", "imperative") for s in str_strengths)

    if is_inter_vs_imper:
        print("[TEST] Detected inter_vs_imper style — using per-prompt conditional pairing …")
        results = run_inter_vs_imper_analysis(df, alpha=alpha)
    else:
        print(f"[TEST] Running prompt-level baseline-vs-perturbed analysis (baseline_strength={baseline_strength}) …")
        results = run_baseline_vs_perturbed(df, alpha=alpha, baseline_strength=baseline_strength)

    # ── Silhouette (aggregate, separate path) ─────────────────────────────────
    if silhouette_csvs:
        sil_df = load_silhouette_data(silhouette_csvs)
        if not sil_df.empty:
            if models:
                sil_df = sil_df[sil_df["model"].isin(models)]
            if places and "place" in sil_df.columns:
                sil_df = sil_df[sil_df["place"].isin(places)]
            sil_results = run_silhouette_analysis(sil_df, baseline_strength=baseline_strength)
            if not sil_results.empty:
                results = pd.concat([results, sil_results], ignore_index=True)

    if results.empty:
        print("[WARN] No results produced.")
        return

    out_path = os.path.join(out_dir, "baseline_vs_perturbed.csv")
    results.to_csv(out_path, index=False)
    print(f"[SAVE] {out_path}  ({len(results):,} rows)")

    # Save pivot tables + heatmaps per metric
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HEATMAP_CFG = {
        "significant":        dict(cmap="RdYlGn", vmin=0, vmax=1, fmt=".0f",
                                   title_suffix="Significant (BH-corrected)"),
        "delta":              dict(cmap="RdBu_r", vmin=None, vmax=None, fmt=".3f",
                                   title_suffix="Delta (perturbed − baseline)"),
        "p_value_corrected":  dict(cmap="RdYlGn_r", vmin=0, vmax=1, fmt=".3f",
                                   title_suffix="p-value (BH-corrected)"),
    }

    for metric in results["metric"].unique():
        for val_col, cfg in HEATMAP_CFG.items():
            pivot = pivot_significance(results, style="", metric_name=metric, value_col=val_col)
            if pivot.empty:
                continue

            # Save CSV
            pivot_path = os.path.join(out_dir, f"pivot_{metric}_{val_col}.csv")
            pivot.to_csv(pivot_path)

            # Save heatmap PNG
            fig, ax = plt.subplots(figsize=(max(3, len(pivot.columns) * 1.2),
                                            max(2, len(pivot.index) * 0.7)))
            data = pivot.values.astype(float)
            vmin = cfg["vmin"] if cfg["vmin"] is not None else np.nanmin(data)
            vmax = cfg["vmax"] if cfg["vmax"] is not None else np.nanmax(data)
            im = ax.imshow(data, cmap=cfg["cmap"], vmin=vmin, vmax=vmax, aspect="auto")
            plt.colorbar(im, ax=ax)

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=8)

            fmt = cfg["fmt"]
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    v = data[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, format(v, fmt), ha="center", va="center",
                                fontsize=7, color="black")

            ax.set_title(f"{metric} — {cfg['title_suffix']}", fontsize=9)
            ax.set_xlabel("Place", fontsize=8)
            ax.set_ylabel("Model", fontsize=8)
            fig.tight_layout()

            png_path = os.path.join(out_dir, f"heatmap_{metric}_{val_col}.png")
            fig.savefig(png_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"[SAVE] {png_path}")

    sig_count = results["significant"].sum()
    total = results["significant"].notna().sum()
    print(f"\n[DONE] {sig_count}/{total} (metric, model, place) combos are significant.")
    print(f"[DONE] Results saved to: {out_dir}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pairwise significance tests for styled-prompt experiment results."
    )
    parser.add_argument(
        "--runs", nargs="+", required=False, default=None,
        help=(
            "Input CSV files or run directories. "
            "Accepts combined_means_by_model_place_strength.csv, summary.csv, "
            "or full_results_all_models.csv files."
        ),
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Directory to save all output CSV files.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level for t-tests (default: 0.05).",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=None,
        help="Restrict to these metric columns (default: all detected).",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Restrict to these models.",
    )
    parser.add_argument(
        "--places", nargs="+", default=None,
        help="Restrict to these placements.",
    )

    # ── Prompt-level mode ────────────────────────────────────────────────────
    parser.add_argument(
        "--prompt_level",
        nargs="+",
        metavar="CSV",
        default=None,
        help=(
            "One or more prompt-level CSV files (full_results_all_models.csv). "
            "When given, runs prompt-level baseline-vs-perturbed analysis instead "
            "of the default aggregate significance tests. --runs is not needed."
        ),
    )
    parser.add_argument(
        "--baseline_strength",
        type=float,
        default=0,
        help=(
            "Strength value that represents the unperturbed baseline (default: 0). "
            "Use 1.0 for length_variation where s=1.0 is the unperturbed condition."
        ),
    )
    parser.add_argument(
        "--asr_dirs",
        nargs="+",
        metavar="DIR",
        default=None,
        help=(
            "Model-level ASR output directories containing harmbench_judged_*.csv. "
            "E.g. results/politeness_safety/run_.../asr_outputs/L3.1-8B. "
            "Pass multiple dirs for multiple models/runs. "
            "Only used with --prompt_level."
        ),
    )
    parser.add_argument(
        "--silhouette_csvs",
        nargs="+",
        metavar="CSV",
        default=None,
        help=(
            "summary.csv files with columns model, place, strength, silhouette. "
            "Silhouette is aggregate (no prompt_id) — reported as mean±std only, "
            "no significance test. Only used with --prompt_level."
        ),
    )
    parser.add_argument(
        "--cot_csvs",
        nargs="+",
        metavar="CSV",
        default=None,
        help=(
            "results_with_cot_analysis.csv files. Uses cot_steps as CoT length "
            "metric and runs a paired t-test (baseline vs perturbed per prompt). "
            "Only used with --prompt_level."
        ),
    )
    args = parser.parse_args()

    if args.prompt_level:
        run_prompt_level(
            prompt_csvs=args.prompt_level,
            out_dir=args.out_dir,
            alpha=args.alpha,
            metrics=args.metrics,
            baseline_strength=args.baseline_strength,
            models=args.models,
            places=args.places,
            asr_dirs=args.asr_dirs,
            silhouette_csvs=args.silhouette_csvs,
            cot_csvs=args.cot_csvs,
        )
    else:
        if not args.runs:
            parser.error("--runs is required unless --prompt_level is given.")
        run(
            runs=args.runs,
            out_dir=args.out_dir,
            alpha=args.alpha,
            metrics=args.metrics,
            models=args.models,
            places=args.places,
        )


if __name__ == "__main__":
    main()
