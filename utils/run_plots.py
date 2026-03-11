"""
Usage:
  python utils/run_plots.py \
      --runs results/politeness/run_*/summary.csv \
      --out_dir results/combined_plots/polite \
      --dataset_name "TruthfulQA" \
      --style_name "politeness"

  python utils/run_plots.py \
      --runs results/combined_plots/safety_polite/combined_means_by_model_place_strength.csv \
      --out_dir results/combined_plots/safety_polite

  # length_variation with extra structuredness plot:
  python utils/run_plots.py \
      --runs results/length_variation/run_a/summary.csv results/length_variation/run_b/summary.csv \
      --row_runs results/length_variation/run_a/full_results_all_models.csv results/length_variation/run_b/full_results_all_models.csv \
      --out_dir results/combined_plots/length_variation \
      --dataset_name "TruthfulQA" \
      --style_name "length_variation"

  # BERTScore prompt-preservation check (no --runs needed):
  python utils/run_plots.py \
      --prompt_check \
      --out_dir results/prompt_bertscore_check

Plots produced in --prompt_check mode (strengths & datasets from config.yaml):
  - bertscore_prompt_politeness.png        : line plot, all datasets × places
  - bertscore_prompt_spacing_case_punct.png: 3 subplots (spacing|letter_case|punctuation)
  - bertscore_prompt_llm_styles.png        : 2 subplots (length_variation|inter_vs_imper)
                                             uses data/llm_style_cache/ for LLM styles
  - *.csv: raw BERTScore means per subplot

Extra behavior:
  - When --style_name length_variation is used, this script also tries to
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
    if args.style_name != "length_variation":
        return

    from utils.plots import load_results_csvs, make_structuredness_plots

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
        Single line plot, all datasets × places.  Politeness style only.

    Plot 2 – bertscore_prompt_spacing_case_punct.png
        One figure, 3 side-by-side subplots: spacing | letter_case | punctuation.
        Shared legend.  Style applied deterministically on-the-fly.

    Plot 3 – bertscore_prompt_llm_styles.png
        One figure, 2 side-by-side subplots: length_variation | inter_vs_imper.
        Prompts and their styled counterparts loaded from data/llm_style_cache/.
    """
    import json
    import numpy as np
    import torch
    import yaml

    from utils.data import load_dataset_by_name
    from utils.styles import apply_politeness, apply_spacing, apply_letter_case, apply_punctuation
    from utils.plots import (
        plot_bertscore_prompt_lines,
        plot_bertscore_prompt_subplots,
    )

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ── Load config ───────────────────────────────────────────────────────────
    config_path = os.path.join(repo_root, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    style_levels = cfg.get("style_levels", {})
    style_positions = cfg.get("style_positions", {})
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
        with torch.inference_mode():
            _, _, F1 = _scorer.score(cands, refs, batch_size=bert_batch, verbose=False)
        return float(np.nanmean(F1.cpu().numpy()))

    # ── Dataset loader ────────────────────────────────────────────────────────
    def _get_prompt(item):
        if isinstance(item, dict):
            v = item.get("question") or item.get("prompt") or item.get("instruction") or item.get("text")
            if v is not None:
                if isinstance(v, dict):
                    v = v.get("text", "")
                return str(v).strip()
        return str(item).strip()

    def _load_prompts(ds_name):
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
    pol_strengths = style_levels.get("politeness", list(range(-10, 11, 2)))
    pol_places    = style_positions.get("politeness", ["prefix", "suffix", "global"])

    rows_pol = []
    for ds, prompts in dataset_prompts.items():
        for place in pol_places:
            for s in pol_strengths:
                styled = [apply_politeness(p, int(s), place=place) for p in prompts]
                rows_pol.append({
                    "dataset": ds, "place": place, "strength": s,
                    "bertscore_prompt": _bertscore_f1(prompts, styled),
                })
                print(f"  politeness | {ds} | {place} | s={s}")

    import pandas as pd
    df_pol = pd.DataFrame(rows_pol)
    csv1 = os.path.join(args.out_dir, "bertscore_prompt_politeness.csv")
    df_pol.to_csv(csv1, index=False)
    png1 = os.path.join(args.out_dir, "bertscore_prompt_politeness.png")
    plot_bertscore_prompt_lines(
        df_pol, png1,
        threshold=threshold, save_pdf=save_pdf, style_name="Politeness",
    )
    print(f"[SAVE] {png1}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 2: Spacing | Letter Case | Punctuation — 3-subplot figure
    # ─────────────────────────────────────────────────────────────────────────
    print("[PROMPT CHECK] Plot 2: spacing | letter_case | punctuation …")
    det_styles = {
        "spacing":     (apply_spacing,      style_levels.get("spacing",     [0, 1, 5, 20, 50, 100]),
                        style_positions.get("spacing",     ["prefix", "suffix", "global"])),
        "letter_case": (apply_letter_case,  style_levels.get("letter_case", [0, 10, 25, 50, 75, 100]),
                        style_positions.get("letter_case", ["prefix", "suffix", "global"])),
        "punctuation": (apply_punctuation,  style_levels.get("punctuation", [0, 1, 3, 5, 10, 20]),
                        style_positions.get("punctuation", ["prefix", "suffix", "global"])),
    }

    dfs_det = []
    titles_det = []
    for style_key, (apply_fn, strengths, places) in det_styles.items():
        rows = []
        for ds, prompts in dataset_prompts.items():
            for place in places:
                for s in strengths:
                    styled = [apply_fn(p, s, place=place) for p in prompts]
                    rows.append({
                        "dataset": ds, "place": place, "strength": s,
                        "bertscore_prompt": _bertscore_f1(prompts, styled),
                    })
                    print(f"  {style_key} | {ds} | {place} | s={s}")
        dfs_det.append(pd.DataFrame(rows))
        titles_det.append(style_key.replace("_", " ").title())

    csv2 = os.path.join(args.out_dir, "bertscore_prompt_spacing_case_punct.csv")
    pd.concat(
        [df.assign(style=t) for df, t in zip(dfs_det, titles_det)],
        ignore_index=True,
    ).to_csv(csv2, index=False)

    png2 = os.path.join(args.out_dir, "bertscore_prompt_spacing_case_punct.png")
    plot_bertscore_prompt_subplots(
        dfs_det, titles_det, png2,
        threshold=threshold, save_pdf=save_pdf,
        suptitle="BERTScore (Prompt) vs. Style Strength — Surface Noise",
    )
    print(f"[SAVE] {png2}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 3: Length Variation | Inter vs Imper — 2-subplot figure (from cache)
    # ─────────────────────────────────────────────────────────────────────────
    print("[PROMPT CHECK] Plot 3: length_variation | inter_vs_imper (from cache) …")
    cache_root = os.path.join(repo_root, "data", "llm_style_cache")
    lv_strengths  = style_levels.get("length_variation", [0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    imper_params  = style_levels.get("inter_vs_imper",   ["interrogative", "imperative"])

    def _load_llm_cache(ds_name, style_prefix, param):
        """Load (orig, styled) pairs from the jsonl cache file for a given style param."""
        fname = os.path.join(cache_root, ds_name, f"{style_prefix}__{param}.jsonl")
        if not os.path.isfile(fname):
            return [], []
        origs, styled_list = [], []
        with open(fname) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                origs.append(rec.get("prompt_orig", ""))
                styled_list.append(rec.get("prompt_styled", ""))
        return origs, styled_list

    dfs_llm = []
    titles_llm = []

    # Length variation
    rows_lv = []
    for ds in DATASETS_CFG:
        for s in lv_strengths:
            origs, styled_list = _load_llm_cache(ds, "length_variation", s)
            if not origs:
                print(f"  [WARN] No cache for length_variation | {ds} | s={s} — skipping")
                continue
            rows_lv.append({
                "dataset": ds, "place": "global", "strength": s,
                "bertscore_prompt": _bertscore_f1(origs, styled_list),
            })
            print(f"  length_variation | {ds} | s={s}")
    dfs_llm.append(pd.DataFrame(rows_lv))
    titles_llm.append("Length Variation")

    # Inter vs Imper
    rows_iv = []
    for ds in DATASETS_CFG:
        for param in imper_params:
            origs, styled_list = _load_llm_cache(ds, "inter_vs_imper", param)
            if not origs:
                print(f"  [WARN] No cache for inter_vs_imper | {ds} | param={param} — skipping")
                continue
            rows_iv.append({
                "dataset": ds, "place": "global", "strength": param,
                "bertscore_prompt": _bertscore_f1(origs, styled_list),
            })
            print(f"  inter_vs_imper | {ds} | param={param}")
    dfs_llm.append(pd.DataFrame(rows_iv))
    titles_llm.append("Interrogative vs Imperative")

    csv3 = os.path.join(args.out_dir, "bertscore_prompt_llm_styles.csv")
    pd.concat(
        [df.assign(style=t) for df, t in zip(dfs_llm, titles_llm)],
        ignore_index=True,
    ).to_csv(csv3, index=False)

    png3 = os.path.join(args.out_dir, "bertscore_prompt_llm_styles.png")
    plot_bertscore_prompt_subplots(
        dfs_llm, titles_llm, png3,
        threshold=threshold, save_pdf=save_pdf,
        suptitle="BERTScore (Prompt) vs. Style — Structured Rewriting",
    )
    print(f"[SAVE] {png3}\n")

    print(f"\n[DONE] All prompt-check plots saved to: {args.out_dir}")


def main():
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
            "when --style_name length_variation. Typically full_results_all_models.csv "
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
    parser.add_argument("--style_name", default=None,
                        help="Style type (e.g. politeness, spacing, letter_case, "
                             "punctuation, length_variation, inter_vs_imper). "
                             "Added to all plot titles; also controls which strength "
                             "value is excluded from radar averages. "
                             "In --prompt_check mode use 'all' to run every non-LLM style.")
    parser.add_argument("--save_pdf", action="store_true",
                        help="Also save each plot as PDF.")

    # ── BERTScore prompt-preservation check mode ─────────────────────────────
    parser.add_argument(
        "--prompt_check",
        action="store_true",
        help=(
            "Generate BERTScore(prompt) preservation plots for all styles. "
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

    from utils.plots import (
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
            style_name=args.style_name,
        )

        plot_metric_lines_per_model(
            df, metric,
            out_dir=os.path.join(args.out_dir, "line_per_model"),
            models=args.models,
            places=args.places,
            save_pdf=args.save_pdf,
            dataset_name=args.dataset_name,
            style_name=args.style_name,
        )

        plot_metric_lines_per_place(
            df, metric,
            out_dir=os.path.join(args.out_dir, "line_per_place"),
            models=args.models,
            places=args.places,
            save_pdf=args.save_pdf,
            dataset_name=args.dataset_name,
            style_name=args.style_name,
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
            style_name=args.style_name,
        )

        plot_radar_models_axes(
            df, metric,
            out_path=os.path.join(radar_dir, f"{metric}_radar_axes_models.png"),
            models=args.models,
            places=args.places,
            radar_norm=args.radar_norm,
            save_pdf=args.save_pdf,
            dataset_name=args.dataset_name,
            style_name=args.style_name,
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
        style_name=args.style_name,
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
            style_name=args.style_name,
        )

    _maybe_generate_structuredness_plots(args)

    print(f"\n[DONE] All plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()