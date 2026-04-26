"""
Generate BERTScore(prompt) preservation plots for all styles using the 6
paper datasets at sample_size=16 (or --sample_size N).

Produces (in --out_dir):
  bertscore_prompt_spacing_case_punct.png  – 3 subplots: Spacing | Letter Case | Punctuation
  bertscore_prompt_llm_styles.png          – 3 subplots: Politeness | Length Variation | Interrogative vs Imperative
  bertscore_prompt_legend.png              – standalone legend image
  *.csv                                    – raw BERTScore means per figure

Usage:
  python plots/run_bertscore_check_paper.py --out_dir results/prompt_bertscore_check_paper
  python plots/run_bertscore_check_paper.py --out_dir results/prompt_bertscore_check_paper --sample_size 16 --save_pdf
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_dataset_by_name
from utils.styles import apply_politeness, apply_spacing, apply_letter_case, apply_punctuation
from plots.plots import (
    plot_bertscore_prompt_subplots,
    save_bertscore_legend_image,
    build_dataset_color_map,
)


def main():
    """Compute and save BERTScore prompt-preservation plots for all styles and datasets."""
    parser = argparse.ArgumentParser(
        description="BERTScore prompt-preservation check for the 6 paper datasets."
    )
    parser.add_argument("--out_dir",          default="results/prompt_bertscore_check_paper")
    parser.add_argument("--sample_size",      type=int,   default=16)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--threshold",        type=float, default=0.85)
    parser.add_argument("--bert_batch_size",  type=int,   default=32)
    parser.add_argument("--bert_model_type",  default="roberta-large")
    parser.add_argument("--save_pdf",         action="store_true")
    args = parser.parse_args()

    repo_root   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(repo_root, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    style_levels    = cfg.get("style_levels",    {})
    style_positions = cfg.get("style_positions", {})
    ds_cfg          = cfg.get("datasets",        {})

    # ── 6 paper datasets (display label → (loader_key, extra_kwargs)) ─────────
    # config_name / split read from config.yaml with sensible fallbacks.
    DATASETS = {
        "TruthfulQA":        ("truthful_qa", {
            "config_name": ds_cfg.get("truthful_qa",       {}).get("config_name", "generation"),
            "split":       ds_cfg.get("truthful_qa",       {}).get("split",       "validation"),
        }),
        "Natural Questions": ("natural_questions", {
            "config_name": ds_cfg.get("natural_questions", {}).get("config_name", "default"),
            "split":       ds_cfg.get("natural_questions", {}).get("split",       "validation"),
        }),
        "Alpaca":            ("alpaca", {
            "config_name": ds_cfg.get("alpaca",           {}).get("config_name", "default"),
            "split":       ds_cfg.get("alpaca",           {}).get("split",       "train"),
        }),
        "SimpleQA Verified": ("simpleqa_verified", {
            "config_name": ds_cfg.get("simpleqa_verified", {}).get("config_name", "default"),
            "split":       ds_cfg.get("simpleqa_verified", {}).get("split",       "eval"),
        }),
        "TriviaQA":          ("trivia_qa", {
            "config_name": ds_cfg.get("trivia_qa",         {}).get("config_name", "rc"),
            "split":       ds_cfg.get("trivia_qa",         {}).get("split",       "validation"),
        }),
        "HotpotQA":          ("hotpot_qa", {
            "config_name": ds_cfg.get("hotpot_qa",         {}).get("config_name", "distractor"),
            "split":       ds_cfg.get("hotpot_qa",         {}).get("split",       "validation"),
        }),
    }

    # ── BERTScore ─────────────────────────────────────────────────────────────
    try:
        from bert_score import BERTScorer
    except ImportError as e:
        raise ImportError("Install bert-score:  pip install bert-score") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BERT] Loading BERTScorer ({args.bert_model_type}) on {device} …")
    _scorer = BERTScorer(
        model_type=args.bert_model_type,
        lang="en",
        device=device,
        rescale_with_baseline=False,
    )
    print("[BERT] Ready.\n")

    def _bertscore_f1(refs, cands):
        """Compute mean BERTScore F1 between reference and candidate lists."""
        with torch.inference_mode():
            _, _, F1 = _scorer.score(cands, refs, batch_size=args.bert_batch_size, verbose=False)
        return float(np.nanmean(F1.cpu().numpy()))

    # ── Extract prompt text from standardised dataset dicts ───────────────────
    def _get_prompt(item):
        """Extract a prompt string from a dataset item dict or raw string."""
        if isinstance(item, dict):
            v = (item.get("question") or item.get("prompt")
                 or item.get("instruction") or item.get("text"))
            if v is not None:
                if isinstance(v, dict):
                    v = v.get("text", "")
                return str(v).strip()
        return str(item).strip()

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("[DATA] Loading datasets …")
    dataset_prompts: dict[str, list[str]] = {}
    for label, (ds_name, kwargs) in DATASETS.items():
        items = load_dataset_by_name(
            ds_name, sample_size=args.sample_size, seed=args.seed, **kwargs
        )
        dataset_prompts[label] = [_get_prompt(x) for x in items]
        print(f"  {label}: {len(dataset_prompts[label])} prompts")
    print("[DATA] Done.\n")

    os.makedirs(args.out_dir, exist_ok=True)
    threshold = args.threshold
    save_pdf  = args.save_pdf

    # ─────────────────────────────────────────────────────────────────────────
    # Politeness — needed for combined figure (plot 2)
    # ─────────────────────────────────────────────────────────────────────────
    print("[CHECK] Politeness …")
    pol_strengths = style_levels.get("politeness", list(range(-10, 11, 2)))
    pol_places    = style_positions.get("politeness", ["prefix", "suffix", "global"])

    rows_pol = []
    for label, prompts in dataset_prompts.items():
        for place in pol_places:
            for s in pol_strengths:
                styled = [apply_politeness(p, int(s), place=place) for p in prompts]
                rows_pol.append({
                    "dataset": label, "place": place, "strength": s,
                    "bertscore_prompt": _bertscore_f1(prompts, styled),
                })
                print(f"  politeness | {label} | {place} | s={s}")
    df_pol = pd.DataFrame(rows_pol)

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 1: Spacing | Letter Case | Punctuation  (3 subplots)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[CHECK] Spacing / Letter Case / Punctuation …")
    det_styles = {
        "Spacing":      (apply_spacing,      style_levels.get("spacing",     [0, 1, 5, 20, 50, 100]),
                         style_positions.get("spacing",     ["prefix", "suffix", "global"])),
        "Letter Case":  (apply_letter_case,  style_levels.get("letter_case", [0, 10, 25, 50, 75, 100]),
                         style_positions.get("letter_case", ["prefix", "suffix", "global"])),
        "Punctuation":  (apply_punctuation,  style_levels.get("punctuation", [0, 1, 3, 5, 10, 20]),
                         style_positions.get("punctuation", ["prefix", "suffix", "global"])),
    }

    dfs_det, titles_det = [], []
    for title, (apply_fn, strengths, places) in det_styles.items():
        rows = []
        for label, prompts in dataset_prompts.items():
            for place in places:
                for s in strengths:
                    styled = [apply_fn(p, s, place=place) for p in prompts]
                    rows.append({
                        "dataset": label, "place": place, "strength": s,
                        "bertscore_prompt": _bertscore_f1(prompts, styled),
                    })
                    print(f"  {title} | {label} | {place} | s={s}")
        dfs_det.append(pd.DataFrame(rows))
        titles_det.append(title)

    csv1 = os.path.join(args.out_dir, "bertscore_prompt_spacing_case_punct.csv")
    pd.concat(
        [df.assign(style=t) for df, t in zip(dfs_det, titles_det)], ignore_index=True
    ).to_csv(csv1, index=False)

    png1 = os.path.join(args.out_dir, "bertscore_prompt_spacing_case_punct.png")
    plot_bertscore_prompt_subplots(
        dfs_det, titles_det, png1,
        threshold=threshold, save_pdf=save_pdf, force_xmin_zero=True,
    )
    print(f"[SAVE] {png1}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 2: Politeness | Length Variation | Interrogative vs Imperative
    # ─────────────────────────────────────────────────────────────────────────
    print("[CHECK] LLM styles (length_variation | inter_vs_imper) from cache …")
    cache_root   = os.path.join(repo_root, "data", "llm_style_cache")
    lv_strengths = style_levels.get("length_variation", [0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    imper_params = style_levels.get("inter_vs_imper",   ["interrogative", "imperative"])

    # Map display label → cache subfolder name
    DS_TO_CACHE = {
        "TruthfulQA":        "truthful_qa",
        "Natural Questions": "natural_questions",
        "Alpaca":            "alpaca",
        "SimpleQA Verified": "simpleqa_verified",
        "TriviaQA":          "trivia_qa",
        "HotpotQA":          "hotpot_qa",
    }

    def _load_llm_cache(ds_cache_name, style_prefix, param):
        """Load cached LLM outputs for a dataset/style/param combination, returning (originals, styled) lists."""
        fname = os.path.join(cache_root, ds_cache_name, f"{style_prefix}__{param}.jsonl")
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

    rows_lv = []
    for label, ds_cache in DS_TO_CACHE.items():
        for s in lv_strengths:
            origs, styled_list = _load_llm_cache(ds_cache, "length_variation", s)
            if not origs:
                print(f"  [WARN] No cache: length_variation | {label} | s={s} — skipping")
                continue
            rows_lv.append({
                "dataset": label, "place": "global", "strength": s,
                "bertscore_prompt": _bertscore_f1(origs, styled_list),
            })
            print(f"  length_variation | {label} | s={s}")

    rows_iv = []
    for label, ds_cache in DS_TO_CACHE.items():
        for param in imper_params:
            origs, styled_list = _load_llm_cache(ds_cache, "inter_vs_imper", param)
            if not origs:
                print(f"  [WARN] No cache: inter_vs_imper | {label} | param={param} — skipping")
                continue
            rows_iv.append({
                "dataset": label, "place": "global", "strength": param,
                "bertscore_prompt": _bertscore_f1(origs, styled_list),
            })
            print(f"  inter_vs_imper | {label} | param={param}")

    dfs_llm_pol    = [df_pol, pd.DataFrame(rows_lv), pd.DataFrame(rows_iv)]
    titles_llm_pol = ["Politeness", "Length Variation", "Interrogative vs Imperative"]

    csv2 = os.path.join(args.out_dir, "bertscore_prompt_llm_styles.csv")
    pd.concat(
        [df.assign(style=t) for df, t in zip(dfs_llm_pol, titles_llm_pol)], ignore_index=True
    ).to_csv(csv2, index=False)

    png2 = os.path.join(args.out_dir, "bertscore_prompt_llm_styles.png")
    plot_bertscore_prompt_subplots(
        dfs_llm_pol, titles_llm_pol, png2,
        threshold=threshold, save_pdf=save_pdf,
    )
    print(f"[SAVE] {png2}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Standalone legend image
    # ─────────────────────────────────────────────────────────────────────────
    all_dfs        = dfs_det + dfs_llm_pol
    all_datasets_l = sorted({ds for df in all_dfs for ds in df["dataset"].dropna().unique()})
    all_places_l   = sorted({pl for df in all_dfs for pl in df["place"].dropna().unique()})
    color_map_l    = build_dataset_color_map(all_datasets_l)

    png_leg = os.path.join(args.out_dir, "bertscore_prompt_legend.png")
    save_bertscore_legend_image(
        all_datasets_l, all_places_l, color_map_l, png_leg,
        threshold=threshold, save_pdf=save_pdf,
    )
    print(f"[SAVE] {png_leg}\n")

    print(f"\n[DONE] All plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
