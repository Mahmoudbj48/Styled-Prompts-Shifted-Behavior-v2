"""
sgs_core.py
-----------
Paper-aligned SGS pipeline.

This module is the SINGLE source of truth for SGS math. It is imported by
``utils/compute_sgs_table.py`` (the driver), and indirectly by the legacy
shims ``utils/significance_test.py`` and ``utils/compute_sgs_closed_table.py``
which now delegate to the driver.

Pipeline (numbered to match the paper appendix):

  1. Load aggregated open + closed CSVs into a single dataframe.
  2. Sanity-check that baseline rows have delta_X ~= 0 for every axis.
  3. Subsample TruthfulQA prompts to 16 per model (seed=42).
  4. Normalise each delta by a fixed constant R_X (axis-specific).
  5. Compute per-prompt RMS instability:
        s_X(i) = sqrt( mean_{v in V} ( delta_tilde_X(i,v) )^2 )
  6. Compute per-dataset and total SGS_X.
  7. Use FIXED epsilon thresholds (0.01, 0.05, 0.10) shared across axes.
  8. One-sided one-sample t-test of s_X(i) vs popmean = epsilon.

The variant set V used at step 5 is "all rows whose strength differs from
the baseline", with one explicit exception: the ``original`` rows in the
``Form`` variation are filtered out at load time. In practice the
``original`` text is most often identical (or near-identical) to the
``interrogative`` rewrite, so its delta ~= 0 anyway; dropping it is both
semantically correct (it's not part of the interrogative<->imperative
contrast) and prevents a ~50% inflation of V for Form prompts that would
otherwise be silent zero contributions.

Note on column naming: the aggregated CSVs store the variation family in a
column named ``style``. This module renames it to ``variation`` at load
time so that every downstream symbol uses ``variation`` consistently.

Note on entropy: the ``entropy_shift`` row-level metric is re-anchored at
the baseline row by ``compute_delta_metrics.add_delta_columns`` and stored
as ``delta_entropy``. SGS uses ``delta_entropy``, never the raw
``entropy_shift``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# CONSTANTS  (paper-frozen -- DO NOT change without updating the paper)
# ---------------------------------------------------------------------------

# 6 SGS axes, in display order
SGS_AXES: list[str] = [
    "delta_activation_similarity",   # Activation
    "delta_bleu",                    # Generation Consistency - BLEU
    "delta_bertscore_response",      # Generation Consistency - BERT
    "delta_log_prob",                # Confidence - Prob
    "delta_entropy",                 # Confidence - Entropy
    "delta_mirroring_rate",          # Mirroring
]

# Fixed normalisation constants R_X (verified against observed data ranges)
R_X: dict[str, float] = {
    "delta_activation_similarity": 2.0,    # observed |Delta|max ~= 1.79
    "delta_bleu":                  100.0,  # BLEU stored on [0,100] scale
    "delta_bertscore_response":    1.0,    # BERTScore in [-1, 1]
    "delta_log_prob":              250.0,  # observed |Delta|max ~= 213
    "delta_entropy":               10.0,   # observed |Delta|max ~= 8
    "delta_mirroring_rate":        1.0,    # values in {-1, 0, +1}
}

# Baseline strength per variation
BASELINE_STRENGTH: dict[str, object] = {
    "Polite.": 0,
    "Punct.":  0,
    "Spacing": 0,
    "Casing":  0,
    "Length":  1.0,
    "Form":    "interrogative",
}

# FIXED epsilon thresholds (shared across axes; primary = 0.05).
# These replace the earlier per-axis percentile-pooling approach. Rationale:
# the pooled epsilon was either too lax (BLEU eps_95 = 0.91, almost nothing
# significant) or saturated (mirroring eps = 1.0 ceiling). A single
# normalised-magnitude threshold has clear paper-friendly semantics:
# "instability exceeds X% of the maximum possible variation".
FIXED_EPSILONS: list[float] = [0.01, 0.05, 0.10]
PRIMARY_EPSILON: float = 0.05

# Subsample knobs
TRUTHFUL_QA_N: int = 16
SUBSAMPLE_SEED: int = 42

# Tolerance for the baseline-delta sanity assertion
BASELINE_TOL: float = 1e-6


# ---------------------------------------------------------------------------
# Step 1 -- load
# ---------------------------------------------------------------------------

OPEN_AGG_DEFAULT   = Path("results/aggregated_full_results.csv")
CLOSED_AGG_DEFAULT = Path("results/closed_models/aggregated_closed_models.csv")


def load_aggregated(
    open_csv: Path = OPEN_AGG_DEFAULT,
    closed_csv: Path = CLOSED_AGG_DEFAULT,
) -> pd.DataFrame:
    """Load both aggregated CSVs, tag rows with a ``provider`` column, rename
    the on-disk ``style`` column to ``variation`` (if present), and drop
    ``Form/original`` rows that are not part of the SGS contrast."""
    frames = []
    if open_csv.exists():
        d = pd.read_csv(open_csv, low_memory=False)
        d["provider"] = "open"
        frames.append(d)
    if closed_csv.exists():
        d = pd.read_csv(closed_csv, low_memory=False)
        d["provider"] = "closed"
        frames.append(d)
    if not frames:
        raise FileNotFoundError(
            f"No aggregated CSV found. Looked at {open_csv} and {closed_csv}."
        )
    df = pd.concat(frames, ignore_index=True)

    # Standardise column name: 'style' on disk -> 'variation' in code.
    if "style" in df.columns and "variation" not in df.columns:
        df = df.rename(columns={"style": "variation"})

    # Ensure all 6 axis columns exist (some closed-model files may be missing
    # an axis entirely -- keep as NaN so per-row NaN handling kicks in).
    for col in SGS_AXES:
        if col not in df.columns:
            df[col] = np.nan

    # Drop Form/original rows (un-rewritten dataset prompts, not part of V;
    # in practice their text usually equals the interrogative rewrite anyway).
    df = df[~((df["variation"] == "Form")
              & (df["strength"].astype(str) == "original"))].copy()
    return df


# ---------------------------------------------------------------------------
# Step 2 -- baseline sanity check
# ---------------------------------------------------------------------------

def _baseline_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting baseline rows for each variation."""
    s_str = df["strength"].astype(str)
    mask = pd.Series(False, index=df.index)
    for variation, base in BASELINE_STRENGTH.items():
        if variation == "Form":
            mask |= (df["variation"] == variation) & (s_str == str(base))
        else:
            sub = df["variation"] == variation
            num = pd.to_numeric(df["strength"], errors="coerce")
            mask |= sub & (num == float(base))
    return mask


def assert_baseline_deltas_zero(
    df: pd.DataFrame,
    tol: float = BASELINE_TOL,
) -> pd.DataFrame:
    """Verify that every baseline row has |delta_X| <= tol for every SGS axis.

    Returns a dataframe of offending rows (empty if all good). Logs a one-line
    summary to stdout. Does NOT raise.
    """
    base = df[_baseline_mask(df)]
    violations = []
    for axis in SGS_AXES:
        if axis not in base.columns:
            continue
        bad = base[base[axis].abs() > tol]
        if not bad.empty:
            violations.append(
                bad.assign(axis=axis, observed_abs=bad[axis].abs())
                   [["provider", "model", "dataset", "variation", "strength",
                     "place", "prompt_id", "axis", "observed_abs"]]
            )
    out = pd.concat(violations, ignore_index=True) if violations else pd.DataFrame(
        columns=["provider", "model", "dataset", "variation", "strength",
                 "place", "prompt_id", "axis", "observed_abs"]
    )
    n_baseline = len(base)
    n_bad      = len(out)
    pct        = 100.0 * n_bad / max(n_baseline, 1)
    if n_bad == 0:
        print(f"[Q0 chk]  baseline-delta sanity: all {n_baseline:,} baseline rows |delta| <= {tol}  OK")
    else:
        print(f"[Q0 chk]  WARN -- {n_bad}/{n_baseline} baseline rows ({pct:.4f}%) "
              f"exceed |delta| > {tol}. See returned dataframe.")
    return out


# ---------------------------------------------------------------------------
# Step 3 -- TruthfulQA prompt subsample
# ---------------------------------------------------------------------------

def subsample_truthful_qa(
    df: pd.DataFrame,
    n: int = TRUTHFUL_QA_N,
    seed: int = SUBSAMPLE_SEED,
) -> tuple[pd.DataFrame, dict]:
    """Subsample TruthfulQA prompts.

    Strategy:
      * intersection := prompt_ids present for every model in TruthfulQA
      * if |intersection| >= n: pick `n` from the intersection (seeded RNG)
      * else: take all of intersection, then per-model fill the remaining
        slots from each model's own TruthfulQA prompt_ids (seeded RNG).

    Other datasets are left untouched.

    Returns
    -------
    df_filtered : pd.DataFrame
        Same as `df` but with TruthfulQA rows restricted to the chosen ids.
    record : dict
        Diagnostic metadata about the selection (saved to disk by the driver).
    """
    rng = np.random.default_rng(seed)
    tqa = df[df["dataset"] == "TruthfulQA"]
    if tqa.empty:
        return df, {"seed": seed, "n": n, "common_ids": [], "per_model_ids": {},
                    "note": "TruthfulQA not present"}

    ids_per_model = {m: set(g["prompt_id"].unique())
                     for m, g in tqa.groupby("model")}
    common = sorted(set.intersection(*ids_per_model.values()))

    if len(common) >= n:
        chosen_common = sorted(rng.choice(common, size=n, replace=False).tolist())
        per_model_extra: dict[str, list] = {}
        keep_mask = (
            (df["dataset"] != "TruthfulQA")
            | (df["prompt_id"].isin(chosen_common))
        )
    else:
        chosen_common = list(common)
        per_model_extra = {}
        for m, ids in ids_per_model.items():
            need = n - len(common)
            extras_pool = sorted(ids - set(common))
            take = min(need, len(extras_pool))
            extras = sorted(rng.choice(extras_pool, size=take, replace=False).tolist()) \
                     if take > 0 else []
            per_model_extra[m] = extras
        keep = np.zeros(len(df), dtype=bool)
        keep |= (df["dataset"] != "TruthfulQA").values
        common_mask = (df["dataset"] == "TruthfulQA") & (df["prompt_id"].isin(chosen_common))
        keep |= common_mask.values
        for m, extras in per_model_extra.items():
            extra_mask = (
                (df["dataset"] == "TruthfulQA")
                & (df["model"] == m)
                & (df["prompt_id"].isin(extras))
            )
            keep |= extra_mask.values
        keep_mask = pd.Series(keep, index=df.index)

    record = {
        "seed": seed,
        "n_target": n,
        "common_ids": list(map(int, chosen_common)),
        "per_model_ids": {str(k): list(map(int, v)) for k, v in per_model_extra.items()},
        "n_models_in_tqa": len(ids_per_model),
        "n_common_total": len(common),
    }
    return df.loc[keep_mask].copy(), record


# ---------------------------------------------------------------------------
# Step 4 -- normalisation
# ---------------------------------------------------------------------------

def normalize_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``<axis>_norm`` columns: ``delta_X / R_X``. Returns a new DataFrame.

    No shift, no centering: zero stays zero. After normalisation each axis
    column lives in [-1, 1] (verified against observed data ranges).
    """
    out = df.copy()
    for axis, r in R_X.items():
        out[axis + "_norm"] = out[axis] / r
    return out


# ---------------------------------------------------------------------------
# Step 5 -- per-prompt RMS instability
# ---------------------------------------------------------------------------

def _is_in_full_V(df: pd.DataFrame) -> pd.Series:
    """Mask of rows that belong to V (non-baseline)."""
    s_str = df["strength"].astype(str)
    is_baseline = pd.Series(False, index=df.index)
    for variation, base in BASELINE_STRENGTH.items():
        sub = df["variation"] == variation
        if variation == "Form":
            is_baseline |= sub & (s_str == str(base))
        else:
            num = pd.to_numeric(df["strength"], errors="coerce")
            is_baseline |= sub & (num == float(base))
    # Form/original is already filtered at load time; everything not-baseline
    # is therefore in V.
    return ~is_baseline


def compute_per_prompt_s(df_norm: pd.DataFrame) -> pd.DataFrame:
    """Compute ``s_X(i) = sqrt( mean_v ( delta_tilde_X(i,v) )^2 )`` per
    (provider, model, dataset, prompt_id, axis).

    The variant set V is the union of all non-baseline rows for the prompt.
    Lowest-strength filtering is no longer used (dropped together with the
    pooled-epsilon approach -- see FIXED_EPSILONS).

    Returns
    -------
    long DataFrame with columns:
        provider, model, dataset, prompt_id, axis, s_X, n_variants_used
    """
    sub  = df_norm.loc[_is_in_full_V(df_norm)].copy()
    norm_cols = [a + "_norm" for a in SGS_AXES]

    sq = sub[norm_cols].pow(2)
    sq.columns = SGS_AXES
    grp = pd.concat(
        [sub[["provider", "model", "dataset", "prompt_id"]], sq],
        axis=1,
    )

    counts = (
        grp.groupby(["provider", "model", "dataset", "prompt_id"])[SGS_AXES]
           .apply(lambda g: g.notna().sum())
    )
    means = (
        grp.groupby(["provider", "model", "dataset", "prompt_id"])[SGS_AXES]
           .mean()
    )
    s_values = np.sqrt(means)

    long = (
        s_values.stack()
                .rename("s_X")
                .reset_index()
                .rename(columns={"level_4": "axis"})
    )
    cnt_long = (
        counts.stack()
              .rename("n_variants_used")
              .reset_index()
              .rename(columns={"level_4": "axis"})
    )
    out = long.merge(
        cnt_long,
        on=["provider", "model", "dataset", "prompt_id", "axis"],
    )
    return out


# ---------------------------------------------------------------------------
# Step 6 -- SGS aggregation
# ---------------------------------------------------------------------------

def compute_sgs_per_dataset(per_prompt: pd.DataFrame) -> pd.DataFrame:
    """SGS_X^(j) = mean over prompts in dataset j (NaN-skipped)."""
    return (
        per_prompt
        .groupby(["provider", "model", "dataset", "axis"], dropna=False)["s_X"]
        .mean()
        .reset_index()
        .rename(columns={"s_X": "sgs"})
    )


def compute_sgs_total(per_prompt: pd.DataFrame) -> pd.DataFrame:
    """Total SGS_X = (1/N) * Sum_i s_X(i) -- flat mean over all prompts pooled
    across datasets. Equal to the dataset-size-weighted mean of per-dataset
    SGS by construction."""
    return (
        per_prompt
        .groupby(["provider", "model", "axis"], dropna=False)["s_X"]
        .mean()
        .reset_index()
        .rename(columns={"s_X": "sgs"})
    )


# ---------------------------------------------------------------------------
# Step 7 -- fixed epsilon table
# ---------------------------------------------------------------------------

def fixed_epsilon_table() -> pd.DataFrame:
    """Return the fixed epsilon thresholds used at every axis. Same
    epsilon for every axis -- no pooling, no per-axis percentile."""
    rows = []
    for axis in SGS_AXES:
        row = {"axis": axis}
        for eps in FIXED_EPSILONS:
            # column name uses 2-digit pct for filename-friendliness
            row[f"epsilon_{int(round(eps * 100)):02d}"] = eps
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 8 -- t-tests
# ---------------------------------------------------------------------------

def _ttest_against(sample: np.ndarray, popmean: float) -> tuple[float, float]:
    """One-sided one-sample t-test, alternative='greater'. Returns (t, p)."""
    sample = np.asarray(sample, dtype=float)
    sample = sample[~np.isnan(sample)]
    if len(sample) < 2 or np.isnan(popmean):
        return np.nan, np.nan
    res = stats.ttest_1samp(sample, popmean=popmean, alternative="greater")
    return float(res.statistic), float(res.pvalue)


def significance_marker(p: float) -> str:
    """LaTeX-friendly significance marker for the supplied p-value."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return r"^{\dagger}"
    if p < 0.01:
        return r"^{**}"
    if p < 0.05:
        return r"^{*}"
    return ""


def run_ttests(
    per_prompt: pd.DataFrame,
    level: str,
    epsilons: Iterable[float] = FIXED_EPSILONS,
) -> pd.DataFrame:
    """Run the t-test at the requested aggregation level against fixed
    epsilon thresholds.

    Parameters
    ----------
    per_prompt : long df from compute_per_prompt_s.
    level      : 'model'  -> per (provider, model, axis), pooled across datasets
                 'dataset' -> per (provider, model, dataset, axis)
    epsilons   : fixed thresholds (default FIXED_EPSILONS).
    """
    if level == "model":
        keys = ["provider", "model", "axis"]
    elif level == "dataset":
        keys = ["provider", "model", "dataset", "axis"]
    else:
        raise ValueError(f"Unknown level={level!r}")

    rows = []
    for grp_keys, sub in per_prompt.groupby(keys, dropna=False):
        if not isinstance(grp_keys, tuple):
            grp_keys = (grp_keys,)
        row = dict(zip(keys, grp_keys))
        sample = sub["s_X"].to_numpy()
        for eps in epsilons:
            tag = f"{int(round(eps * 100)):02d}"
            t, p = _ttest_against(sample, eps)
            row[f"t_{tag}"]   = t
            row[f"p_{tag}"]   = p
            row[f"sig_{tag}"] = significance_marker(p)
        row["n"] = int(np.sum(~np.isnan(sample)))
        rows.append(row)
    return pd.DataFrame(rows)


def epsilon_significance_summary(
    sig_per_model: pd.DataFrame,
    epsilons: Iterable[float] = FIXED_EPSILONS,
) -> pd.DataFrame:
    """Return a per-axis count of how many (provider, model) cells reach
    each significance level at each epsilon. Used as the appendix epsilon
    table content (see paper)."""
    rows = []
    for axis in SGS_AXES:
        row = {"axis": axis}
        sub = sig_per_model[sig_per_model["axis"] == axis]
        for eps in epsilons:
            tag = f"{int(round(eps * 100)):02d}"
            p_col = f"p_{tag}"
            if p_col not in sub.columns:
                row[f"epsilon_{tag}"]   = eps
                row[f"n_sig05_{tag}"]   = 0
                row[f"n_sig01_{tag}"]   = 0
                row[f"n_sig001_{tag}"]  = 0
                continue
            row[f"epsilon_{tag}"]   = eps
            row[f"n_sig05_{tag}"]   = int((sub[p_col] < 0.05).sum())
            row[f"n_sig01_{tag}"]   = int((sub[p_col] < 0.01).sum())
            row[f"n_sig001_{tag}"]  = int((sub[p_col] < 0.001).sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience wrappers used by the driver
# ---------------------------------------------------------------------------

def run_full_pipeline(
    open_csv: Path = OPEN_AGG_DEFAULT,
    closed_csv: Path = CLOSED_AGG_DEFAULT,
) -> dict:
    """End-to-end orchestration. Returns a dict of dataframes the driver
    can dump to disk and turn into LaTeX tables."""
    print(f"[load]    {open_csv}")
    print(f"          {closed_csv}")
    df_raw = load_aggregated(open_csv, closed_csv)
    print(f"          -> {len(df_raw):,} rows ({df_raw['provider'].value_counts().to_dict()})")

    print("[Q0 chk]  baseline-delta sanity ...")
    baseline_violations = assert_baseline_deltas_zero(df_raw)

    print("[Q1 sub]  TruthfulQA subsample ...")
    df_sub, tqa_record = subsample_truthful_qa(df_raw)
    print(f"          common ids kept: {len(tqa_record['common_ids'])}; "
          f"per-model extras: {sum(len(v) for v in tqa_record['per_model_ids'].values())}")

    print("[norm]    R_X = " + ", ".join(f"{a}={r}" for a, r in R_X.items()))
    df_norm = normalize_deltas(df_sub)

    print("[s_X]     full V ...")
    per_prompt = compute_per_prompt_s(df_norm)

    print("[SGS]     per-dataset and total ...")
    sgs_per_dataset = compute_sgs_per_dataset(per_prompt)
    sgs_total       = compute_sgs_total(per_prompt)

    print(f"[eps]     fixed epsilons = {FIXED_EPSILONS} (primary={PRIMARY_EPSILON})")
    epsilons_table = fixed_epsilon_table()

    print("[ttest]   level-1 (model, axis, dataset) ...")
    sig_dataset = run_ttests(per_prompt, level="dataset")
    print("[ttest]   level-2 (model, axis) ...")
    sig_model   = run_ttests(per_prompt, level="model")

    print("[summary] epsilon -> count of significant (model, axis) cells ...")
    eps_summary = epsilon_significance_summary(sig_model)

    return {
        "df_norm":             df_norm,
        "tqa_record":          tqa_record,
        "baseline_violations": baseline_violations,
        "per_prompt":          per_prompt,
        "sgs_per_dataset":     sgs_per_dataset,
        "sgs_total":           sgs_total,
        "epsilons":            epsilons_table,
        "sig_dataset":         sig_dataset,
        "sig_model":           sig_model,
        "eps_summary":         eps_summary,
    }
