"""
CLI runner for generating all publication-ready plots from experiment results.

Aggregates CSV outputs from one or more experiment runs and produces:
  - Line plots (metric vs. strength, grouped by model or placement)
  - Ridge plots (distribution of metric values across strengths)
  - Radar plots (multi-metric, multi-model or multi-placement summaries)
  - Multi-variation radar plots comparing all variation families across models
  - BERTScore prompt-preservation plots across datasets and variations

Modes (select via flags):
  Default          – line/ridge/radar plots from --runs CSV files
  --prompt_check   – BERTScore prompt-preservation check across all variations/datasets
  --multi_style_radar – radar plots aggregating multiple variation families

Usage:
  python utils/run_plots.py \
      --runs results/politeness/run_*/summary.csv \
      --out_dir results/combined_plots/polite \
      --dataset_name "TruthfulQA" \
      --variation_name "politeness"

  python utils/run_plots.py \
      --runs results/combined_plots/safety_polite/combined_means_by_model_place_strength.csv \
      --out_dir results/combined_plots/safety_polite

  # length_variation with extra structuredness plot:
  python utils/run_plots.py \
      --runs results/length_variation/run_a/summary.csv results/length_variation/run_b/summary.csv \
      --row_runs results/length_variation/run_a/full_results_all_models.csv results/length_variation/run_b/full_results_all_models.csv \
      --out_dir results/combined_plots/length_variation \
      --dataset_name "TruthfulQA" \
      --variation_name "length_variation"

  # BERTScore prompt-preservation check (no --runs needed):
  python utils/run_plots.py \
      --prompt_check \
      --out_dir results/prompt_bertscore_check

  # Multi-variation radar plots (Type A + Type B, 3 subplots each):
  python utils/run_plots.py \
      --multi_style_radar \
      --out_dir results/combined_plots/radar_multi_style \
      --variation_data \
          "politeness:results/politeness/run_a/summary.csv,results/politeness/run_b/summary.csv" \
          "spacing:results/spacing/run_x/summary.csv" \
          "letter_case:results/letter_case/run_y/summary.csv" \
          "punctuation:results/punctuation/run_z/summary.csv" \
          "length_variation:results/length_variation/run_lv/summary.csv" \
          "inter_vs_imper:results/inter_vs_imper/run_iv/summary.csv"

Plots produced in --prompt_check mode (strengths & datasets from config.yaml):
  - bertscore_prompt_politeness.png        : line plot, all datasets × places
  - bertscore_prompt_spacing_case_punct.png: 3 subplots (spacing|letter_case|punctuation)
  - bertscore_prompt_llm_styles.png        : 2 subplots (length_variation|inter_vs_imper)
                                             uses data/llm_variation_cache/ for LLM variations
  - *.csv: raw BERTScore means per subplot

Extra behavior:
  - When --variation_name length_variation is used, this script also tries to
    generate the structuredness-specific plots from utils.plots under
    <out_dir>/plots_structuredness.
  - Those extra plots use --row_runs if provided; otherwise they fall back to --runs.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _maybe_generate_structuredness_plots(args) -> None:
    """
    For length_variation runs, also generate plots_structuredness/
    length_ratio_boxplot.png when row-level CSVs are available.
    """
    if args.variation_name != "length_variation":
        return

    from plots.plots import load_results_csvs, make_structuredness_plots

    row_inputs = args.row_runs if args.row_runs else args.runs

    try:
        df_rows = load_results_csvs(row_inputs)
    except Exception as e:
        print(f"[WARN] Could not load row-level inputs for structuredness plots: {e}")
        return

    if args.models is not None and "model" in df_rows.columns:
        df_rows = df_rows[df_rows["model"].isin(args.models)]
    if args.places is not None and "place" in df_rows.columns:
        df_rows = df_rows[df_rows["place"].isin(args.places)]

    required = {"strength", "prompt_orig", "prompt_pert"}
    missing = [c for c in sorted(required) if c not in df_rows.columns]
    if missing:
        print(
            "[WARN] Skipping plots_structuredness: row-level inputs are missing "
            f"required columns {missing}. "
            "Pass --row_runs with full_results_all_models.csv files."
        )
        return

    struct_dir = os.path.join(args.out_dir, "plots_structuredness")
    os.makedirs(struct_dir, exist_ok=True)

    try:
        make_structuredness_plots(
            df_rows,
            struct_dir,
            dataset_name=args.dataset_name or "",
            models_filter=args.models,
            include_title=True,
            save_pdf=args.save_pdf,
        )
        print(f"[DONE] Structuredness-specific plots saved to: {struct_dir}")
    except Exception as e:
        print(f"[WARN] Structuredness-specific plots failed: {e}")


def _run_prompt_check(args) -> None:
    """
    Generates 3 BERTScore(prompt) preservation plots, all in the same theme as
    the original polite_prompt_check.py.  Strengths and datasets come from
    config.yaml in the repo root.

    Plot 1 – bertscore_prompt_politeness.png
        Single line plot, all datasets × places.  Politeness variation only.

    Plot 2 – bertscore_prompt_spacing_case_punct.png
        One figure, 3 side-by-side subplots: spacing | letter_case | punctuation.
        Shared legend.  Style applied deterministically on-the-fly.

    Plot 3 – bertscore_prompt_llm_styles.png
        One figure, 2 side-by-side subplots: length_variation | inter_vs_imper.
        Prompts and their varied counterparts loaded from data/llm_variation_cache/.
    """
    import json
    import numpy as np
    import torch
    import yaml

    from utils.data import load_dataset_by_name
    from utils.variations import apply_politeness, apply_spacing, apply_letter_case, apply_punctuation
    from plots.plots import (
        plot_bertscore_prompt_lines,
        plot_bertscore_prompt_subplots,
        save_bertscore_legend_image,
        build_dataset_color_map,
    )

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ── Load config ───────────────────────────────────────────────────────────
    config_path = os.path.join(repo_root, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    variation_levels = cfg.get("variation_levels", {})
    variation_positions = cfg.get("variation_positions", {})
    dataset_cfgs = cfg.get("datasets", {})

    # Datasets available in config (with their settings).
    # config_name/split of None → loader is called without those kwargs (alpaca, harmbench).
    DATASETS_CFG = {
        "truthful_qa": {
            "config_name": dataset_cfgs.get("truthful_qa", {}).get("config_name", "generation"),
            "split":       dataset_cfgs.get("truthful_qa", {}).get("split",       "validation"),
            "sample_size": dataset_cfgs.get("truthful_qa", {}).get("sample_size", 128),
        },
        "gsm8k": {
            "config_name": dataset_cfgs.get("gsm8k", {}).get("config_name", "main"),
            "split":       dataset_cfgs.get("gsm8k", {}).get("split",       "test"),
            "sample_size": dataset_cfgs.get("gsm8k", {}).get("sample_size", 128),
        },
        "natural_questions": {
            "config_name": dataset_cfgs.get("natural_questions", {}).get("config_name", "default"),
            "split":       dataset_cfgs.get("natural_questions", {}).get("split",       "validation"),
            "sample_size": dataset_cfgs.get("natural_questions", {}).get("sample_size", 128),
        },
        "harmbench": {
            "config_name": "standard",
            "split":       None,
            "sample_size": 128,
        },
        "alpaca": {
            "config_name": None,
            "split":       None,
            "sample_size": 128,
        },
    }
    seed = args.seed
    # args.sample_size overrides config per-dataset size when provided
    sample_size_override = args.sample_size  # None → use config value

    # ── BERTScore — load model once ───────────────────────────────────────────
    try:
        from bert_score import BERTScorer
    except ImportError as e:
        raise ImportError("Install bert-score:  pip install bert-score") from e

    bert_batch = args.bert_batch_size
    bert_model = args.bert_model_type
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[BERT] Loading BERTScorer ({bert_model}) on {device} …")
    _scorer = BERTScorer(
        model_type=bert_model,
        lang="en",
        device=device,
        rescale_with_baseline=False,
    )
    print("[BERT] Ready.\n")

    def _bertscore_f1(refs, cands):
        """Compute mean BERTScore F1 between reference and candidate lists."""
        with torch.inference_mode():
            _, _, F1 = _scorer.score(cands, refs, batch_size=bert_batch, verbose=False)
        return float(np.nanmean(F1.cpu().numpy()))

    # ── Dataset loader ────────────────────────────────────────────────────────
    def _get_prompt(item):
        """Extract a prompt string from a dataset item dict or raw string."""
        if isinstance(item, dict):
            v = item.get("question") or item.get("prompt") or item.get("instruction") or item.get("text")
            if v is not None:
                if isinstance(v, dict):
                    v = v.get("text", "")
                return str(v).strip()
        return str(item).strip()

    def _load_prompts(ds_name):
        """Load and format prompts for the named dataset using the configured sample size."""
        dc = DATASETS_CFG[ds_name]
        n = sample_size_override if sample_size_override is not None else dc["sample_size"]
        kwargs = {"sample_size": n, "seed": seed}
        if dc["config_name"] is not None:
            kwargs["config_name"] = dc["config_name"]
        if dc["split"] is not None:
            kwargs["split"] = dc["split"]
        items = load_dataset_by_name(ds_name, **kwargs)
        return [_get_prompt(x) for x in items]

    print("[DATA] Loading datasets …")
    dataset_prompts = {ds: _load_prompts(ds) for ds in DATASETS_CFG}
    print("[DATA] Done.\n")

    os.makedirs(args.out_dir, exist_ok=True)
    threshold = args.threshold
    save_pdf = args.save_pdf

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 1: Politeness — single line plot, all datasets × places
    # ─────────────────────────────────────────────────────────────────────────
    print("[PROMPT CHECK] Plot 1: politeness …")
    pol_strengths = variation_levels.get("politeness", list(range(-10, 11, 2)))
    pol_places    = variation_positions.get("politeness", ["prefix", "suffix", "global"])

    rows_pol = []
    for ds, prompts in dataset_prompts.items():
        for place in pol_places:
            for s in pol_strengths:
                varied = [apply_politeness(p, int(s), place=place) for p in prompts]
                rows_pol.append({
                    "dataset": ds, "place": place, "strength": s,
                    "bertscore_prompt": _bertscore_f1(prompts, varied),
                })
                print(f"  politeness | {ds} | {place} | s={s}")

    import pandas as pd
    df_pol = pd.DataFrame(rows_pol)
    csv1 = os.path.join(args.out_dir, "bertscore_prompt_politeness.csv")
    df_pol.to_csv(csv1, index=False)
    print(f"[SAVE] {csv1}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 2: Spacing | Letter Case | Punctuation — 3-subplot figure
    # ─────────────────────────────────────────────────────────────────────────
    print("[PROMPT CHECK] Plot 2: spacing | letter_case | punctuation …")
    det_styles = {
        "spacing":     (apply_spacing,      variation_levels.get("spacing",     [0, 1, 5, 20, 50, 100]),
                        variation_positions.get("spacing",     ["prefix", "suffix", "global"])),
        "letter_case": (apply_letter_case,  variation_levels.get("letter_case", [0, 10, 25, 50, 75, 100]),
                        variation_positions.get("letter_case", ["prefix", "suffix", "global"])),
        "punctuation": (apply_punctuation,  variation_levels.get("punctuation", [0, 1, 3, 5, 10, 20]),
                        variation_positions.get("punctuation", ["prefix", "suffix", "global"])),
    }

    dfs_det = []
    titles_det = []
    for variation_key, (apply_fn, strengths, places) in det_styles.items():
        rows = []
        for ds, prompts in dataset_prompts.items():
            for place in places:
                for s in strengths:
                    varied = [apply_fn(p, s, place=place) for p in prompts]
                    rows.append({
                        "dataset": ds, "place": place, "strength": s,
                        "bertscore_prompt": _bertscore_f1(prompts, varied),
                    })
                    print(f"  {variation_key} | {ds} | {place} | s={s}")
        dfs_det.append(pd.DataFrame(rows))
        titles_det.append(variation_key.replace("_", " ").title())

    csv2 = os.path.join(args.out_dir, "bertscore_prompt_spacing_case_punct.csv")
    pd.concat(
        [df.assign(variation=t) for df, t in zip(dfs_det, titles_det)],
        ignore_index=True,
    ).to_csv(csv2, index=False)

    png2 = os.path.join(args.out_dir, "bertscore_prompt_spacing_case_punct.png")
    plot_bertscore_prompt_subplots(
        dfs_det, titles_det, png2,
        threshold=threshold, save_pdf=save_pdf, force_xmin_zero=True,
    )
    print(f"[SAVE] {png2}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 3: Length Variation | Inter vs Imper — 2-subplot figure (from cache)
    # ─────────────────────────────────────────────────────────────────────────
    print("[PROMPT CHECK] Plot 3: length_variation | inter_vs_imper (from cache) …")
    cache_root = os.path.join(repo_root, "data", "llm_variation_cache")
    lv_strengths  = variation_levels.get("length_variation", [0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    imper_params  = variation_levels.get("inter_vs_imper",   ["interrogative", "imperative"])

    def _load_llm_cache(ds_name, variation_prefix, param):
        """Load (orig, varied) pairs from the jsonl cache file for a given variation param."""
        fname = os.path.join(cache_root, ds_name, f"{variation_prefix}__{param}.jsonl")
        if not os.path.isfile(fname):
            return [], []
        origs, varied_list = [], []
        with open(fname) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                origs.append(rec.get("prompt_orig", ""))
                varied_list.append(rec.get("prompt_varied", ""))
        return origs, varied_list

    dfs_llm = []
    titles_llm = []

    # Length variation
    rows_lv = []
    for ds in DATASETS_CFG:
        for s in lv_strengths:
            origs, varied_list = _load_llm_cache(ds, "length_variation", s)
            if not origs:
                print(f"  [WARN] No cache for length_variation | {ds} | s={s} — skipping")
                continue
            rows_lv.append({
                "dataset": ds, "place": "global", "strength": s,
                "bertscore_prompt": _bertscore_f1(origs, varied_list),
            })
            print(f"  length_variation | {ds} | s={s}")
    dfs_llm.append(pd.DataFrame(rows_lv))
    titles_llm.append("Length Variation")

    # Inter vs Imper
    rows_iv = []
    for ds in DATASETS_CFG:
        for param in imper_params:
            origs, varied_list = _load_llm_cache(ds, "inter_vs_imper", param)
            if not origs:
                print(f"  [WARN] No cache for inter_vs_imper | {ds} | param={param} — skipping")
                continue
            rows_iv.append({
                "dataset": ds, "place": "global", "strength": param,
                "bertscore_prompt": _bertscore_f1(origs, varied_list),
            })
            print(f"  inter_vs_imper | {ds} | param={param}")
    dfs_llm.append(pd.DataFrame(rows_iv))
    titles_llm.append("Interrogative vs Imperative")

    # Combine politeness + LLM variations into one figure
    dfs_llm_pol    = [df_pol] + dfs_llm
    titles_llm_pol = ["Politeness"] + titles_llm

    csv3 = os.path.join(args.out_dir, "bertscore_prompt_llm_styles.csv")
    pd.concat(
        [df.assign(variation=t) for df, t in zip(dfs_llm_pol, titles_llm_pol)],
        ignore_index=True,
    ).to_csv(csv3, index=False)

    png3 = os.path.join(args.out_dir, "bertscore_prompt_llm_styles.png")
    plot_bertscore_prompt_subplots(
        dfs_llm_pol, titles_llm_pol, png3,
        threshold=threshold, save_pdf=save_pdf,
    )
    print(f"[SAVE] {png3}\n")

    # Save standalone legend image (3 rows, one per place)
    all_datasets_leg = sorted({ds for df in dfs_llm_pol for ds in df["dataset"].dropna().unique()})
    all_places_leg   = sorted({pl for df in dfs_llm_pol for pl in df["place"].dropna().unique()})
    color_map_leg    = build_dataset_color_map(all_datasets_leg)
    png_leg = os.path.join(args.out_dir, "bertscore_prompt_legend.png")
    save_bertscore_legend_image(
        all_datasets_leg, all_places_leg, color_map_leg, png_leg,
        threshold=threshold, save_pdf=save_pdf,
    )
    print(f"[SAVE] {png_leg}\n")

    print(f"\n[DONE] All prompt-check plots saved to: {args.out_dir}")


def _run_multi_style_radar(args) -> None:
    """
    Generate multi-variation radar plots (Type A and Type B) using radar_plots.py.

    Expects args.variation_data entries like:
        politeness:results/politeness/run_a/summary.csv,results/politeness/run_b/summary.csv
        spacing:results/spacing/run_x/summary.csv
    """
    from plots.radar_plots import make_multi_style_radar_plots

    if not args.variation_data:
        raise SystemExit("[ERROR] --variation_data is required in --multi_style_radar mode.")

    def _parse_style_entries(entries):
        """Parse 'STYLE:path1,path2' CLI entries into a dict mapping variation name to CSV path list."""
        result = {}
        if not entries:
            return result
        for entry in entries:
            if ":" not in entry:
                raise SystemExit(
                    f"[ERROR] entries must be 'STYLE:path1,path2,...'. Got: {entry!r}"
                )
            variation_key, paths_str = entry.split(":", 1)
            paths = [p.strip() for p in paths_str.split(",") if p.strip()]
            if paths:
                result[variation_key.strip()] = paths
        return result

    variation_csvs            = _parse_style_entries(args.variation_data)
    cot_variation_dirs        = _parse_style_entries(getattr(args, "cot_data", None))
    asr_variation_csvs        = _parse_style_entries(getattr(args, "asr_data", None))
    silhouette_variation_csvs = _parse_style_entries(getattr(args, "silhouette_data", None))

    os.makedirs(args.out_dir, exist_ok=True)

    make_multi_style_radar_plots(
        variation_csvs=variation_csvs,
        out_dir=args.out_dir,
        cot_variation_dirs=cot_variation_dirs or None,
        asr_variation_csvs=asr_variation_csvs or None,
        silhouette_variation_csvs=silhouette_variation_csvs or None,
        models=args.models,
        places=args.places,
        save_pdf=args.save_pdf,
    )

    print(f"\n[DONE] Multi-variation radar plots saved to: {args.out_dir}")


def main():
    """Parse CLI arguments and dispatch to the requested plot mode."""
    parser = argparse.ArgumentParser()

    # ── Standard aggregate-plot mode ─────────────────────────────────────────
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help=(
            "Inputs for aggregate plots. Typically summary.csv files, "
            "combined_means_by_model_place_strength.csv files, or run directories "
            "that load_all_runs(...) can resolve.  Not required in --prompt_check mode."
        ),
    )
    parser.add_argument(
        "--row_runs",
        nargs="+",
        default=None,
        help=(
            "Optional row-level inputs used only for structuredness-specific plots "
            "when --variation_name length_variation. Typically full_results_all_models.csv "
            "files or run directories containing them."
        ),
    )

    parser.add_argument("--out_dir", required=True,
                        help="Directory to save all plots.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Restrict to these models.")
    parser.add_argument("--places", nargs="+", default=None,
                        help="Restrict to these placement positions.")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Restrict to these metrics (optional; defaults to all allowed).")
    parser.add_argument("--radar_norm", default="minmax",
                        choices=["none", "minmax", "zscore"],
                        help="Normalisation for radar plots.")
    parser.add_argument("--dataset_name", default=None,
                        choices=["TruthfulQA", "Natural Questions"],
                        help="Dataset label shown in plot titles.")
    parser.add_argument("--variation_name", default=None,
                        help="Style type (e.g. politeness, spacing, letter_case, "
                             "punctuation, length_variation, inter_vs_imper). "
                             "Added to all plot titles; also controls which strength "
                             "value is excluded from radar averages. "
                             "In --prompt_check mode use 'all' to run every non-LLM variation.")
    parser.add_argument("--save_pdf", action="store_true",
                        help="Also save each plot as PDF.")

    # ── Multi-variation radar plot mode ───────────────────────────────────────────
    parser.add_argument(
        "--multi_style_radar",
        action="store_true",
        help=(
            "Generate multi-variation radar plots (Type A: metrics on spokes; "
            "Type B: variations on spokes). Requires --variation_data."
        ),
    )
    parser.add_argument(
        "--variation_data",
        nargs="+",
        default=None,
        help=(
            "[multi_style_radar] One entry per variation in the format "
            "'STYLE:path1,path2,...'. E.g.: "
            "politeness:results/pol/run_a/summary.csv,results/pol/run_b/summary.csv "
            "spacing:results/spacing/run_x/summary.csv"
        ),
    )
    parser.add_argument(
        "--cot_data",
        nargs="+",
        default=None,
        help=(
            "[multi_style_radar] Per-variation CoT run directories or CSV paths "
            "with cot_correct/cot_steps columns. Format: 'STYLE:dir1,dir2,...'. "
            "Directories are searched for results_cleaned.csv automatically."
        ),
    )
    parser.add_argument(
        "--asr_data",
        nargs="+",
        default=None,
        help=(
            "[multi_style_radar] Per-variation ASR CSV paths or directories "
            "(combined_means_by_model_place_strength.csv / summary.csv) with "
            "unsafe_score/asr columns. Format: 'STYLE:csv1,csv2,...'."
        ),
    )
    parser.add_argument(
        "--silhouette_data",
        nargs="+",
        default=None,
        help=(
            "[multi_style_radar] Per-variation silhouette CSV paths or directories "
            "(summary.csv files) with silhouette column. "
            "Format: 'STYLE:csv1,csv2,...'."
        ),
    )

    # ── BERTScore prompt-preservation check mode ─────────────────────────────
    parser.add_argument(
        "--prompt_check",
        action="store_true",
        help=(
            "Generate BERTScore(prompt) preservation plots for all variations. "
            "Strengths and dataset configs are read from config.yaml.  "
            "--runs is not required in this mode."
        ),
    )
    parser.add_argument(
        "--sample_size", type=int, default=None,
        help=(
            "[prompt_check] Override prompts-per-dataset (default: per-dataset "
            "sample_size from config.yaml)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="[prompt_check] Random seed for dataset sampling.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="[prompt_check] Semantic-preservation threshold line on plots.",
    )
    parser.add_argument(
        "--bert_batch_size", type=int, default=32,
        help="[prompt_check] BERTScore batch size.",
    )
    parser.add_argument(
        "--bert_model_type", type=str, default="roberta-large",
        help="[prompt_check] BERTScore backbone model.",
    )

    args = parser.parse_args()

    # ── Multi-variation radar plot mode ───────────────────────────────────────────
    if args.multi_style_radar:
        _run_multi_style_radar(args)
        return

    # ── BERTScore prompt-preservation check mode ─────────────────────────────
    if args.prompt_check:
        _run_prompt_check(args)
        return

    # ── Standard aggregate-plot mode ─────────────────────────────────────────
    if args.runs is None:
        raise SystemExit(
            "[ERROR] --runs is required in standard mode. "
            "Use --prompt_check for the BERTScore prompt-preservation check."
        )

    from plots.plots import (
        load_all_runs,
        select_metrics,
        aggregate_plot_metric_lines,
        plot_metric_lines_per_model,
        plot_metric_lines_per_place,
        plot_radar_places_axes,
        plot_radar_models_axes,
        plot_radar_metrics_axes,
        plot_metric_ridge,
        ALLOWED_METRICS,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_all_runs(args.runs)

    if args.models is not None:
        df = df[df["model"].isin(args.models)]
    if args.places is not None:
        df = df[df["place"].isin(args.places)]

    if df.empty:
        raise SystemExit("[ERROR] No data left after filtering.")

    metrics = select_metrics(df)
    if args.metrics is not None:
        metrics = [m for m in metrics if m in set(args.metrics)]

    # ── Line plots ────────────────────────────────────────────────────────────
    for metric in metrics:
        print(f"[PLOT] {metric}")

        aggregate_plot_metric_lines(
            df, metric,
            out_path=os.path.join(args.out_dir, f"{metric}_line.png"),
            models=args.models,
            places=args.places,
            save_pdf=args.save_pdf,
            dataset_name=args.dataset_name,
            variation_name=args.variation_name,
        )

        plot_metric_lines_per_model(
            df, metric,
            out_dir=os.path.join(args.out_dir, "line_per_model"),
            models=args.models,
            places=args.places,
            save_pdf=args.save_pdf,
            dataset_name=args.dataset_name,
            variation_name=args.variation_name,
        )

        plot_metric_lines_per_place(
            df, metric,
            out_dir=os.path.join(args.out_dir, "line_per_place"),
            models=args.models,
            places=args.places,
            save_pdf=args.save_pdf,
            dataset_name=args.dataset_name,
            variation_name=args.variation_name,
        )

    # ── Radar plots ───────────────────────────────────────────────────────────
    radar_dir = os.path.join(args.out_dir, "radar")
    os.makedirs(radar_dir, exist_ok=True)

    for metric in metrics:
        plot_radar_places_axes(
            df, metric,
            out_path=os.path.join(radar_dir, f"{metric}_radar_axes_places.png"),
            models=args.models,
            places=args.places,
            radar_norm=args.radar_norm,
            save_pdf=args.save_pdf,
            dataset_name=args.dataset_name,
            variation_name=args.variation_name,
        )

        plot_radar_models_axes(
            df, metric,
            out_path=os.path.join(radar_dir, f"{metric}_radar_axes_models.png"),
            models=args.models,
            places=args.places,
            radar_norm=args.radar_norm,
            save_pdf=args.save_pdf,
            dataset_name=args.dataset_name,
            variation_name=args.variation_name,
        )

    metrics_for_radar_c = [m for m in metrics if m in ALLOWED_METRICS]
    plot_radar_metrics_axes(
        df,
        metrics=metrics_for_radar_c,
        out_path=os.path.join(radar_dir, "all_metrics_radar_axes_metrics.png"),
        models=args.models,
        places=args.places,
        radar_norm=args.radar_norm,
        save_pdf=args.save_pdf,
        dataset_name=args.dataset_name,
        variation_name=args.variation_name,
    )

    # ── Ridge plots ───────────────────────────────────────────────────────────
    ridge_dir = os.path.join(args.out_dir, "ridge_plots")
    os.makedirs(ridge_dir, exist_ok=True)

    for metric in metrics:
        plot_metric_ridge(
            df, metric,
            out_path=os.path.join(ridge_dir, f"{metric}_ridge.png"),
            models=args.models,
            places=args.places,
            save_pdf=args.save_pdf,
            dataset_name=args.dataset_name,
            variation_name=args.variation_name,
        )

    _maybe_generate_structuredness_plots(args)

    print(f"\n[DONE] All plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()