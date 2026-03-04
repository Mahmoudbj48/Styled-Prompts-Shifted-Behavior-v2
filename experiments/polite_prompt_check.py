# experiments/polite_prompt_check.py
"""
BERTScore-based semantic preservation check for styled prompts.

Computes BERTScore(prompt) — the semantic similarity between the original
and styled prompt — across multiple datasets and politeness strengths.
This validates that stylistic changes do not corrupt the original prompt meaning.

Inputs:
    - Multiple datasets (e.g. truthful_qa, natural_questions) via utils.data.load_dataset_by_name
    - Politeness style applied via utils.styles.apply_politeness
    - config.yaml is not required; strength grid is configured via CLI flags

Outputs (saved to results/prompt_bertscore_all_datasets/run_YYYYMMDD_HHMMSS/):
    - combined_means_by_dataset_place_strength.csv: mean BERTScore per
      (dataset, place, strength) combination
    - bertscore_prompt_line.png (+ optional .pdf): line plot of BERTScore vs. strength
    - threshold_pass_rates_0.85.csv: fraction of prompts above the 0.85 preservation threshold

Run:
  python experiments/polite_prompt_check.py \\
    --datasets truthful_qa natural_questions \\
    --sample_size 128 \\
    --places global prefix suffix \\
    --strength_range -10 10 --strength_step 2

Important flags:
    --datasets        Space-separated dataset names to evaluate
    --sample_size     Number of prompts per dataset
    --places          Placement strategies to evaluate
    --strength_range  Integer range for politeness strength grid
    --strength_step   Step size within the range
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_dataset_by_name
from utils.styles import apply_politeness
from utils.plots import plot_bertscore_prompt_lines


# ============================================================
# Strengths
# ============================================================

def strength_grid(lo: int = -10, hi: int = 10, step: int = 2) -> List[int]:
    if step <= 0:
        raise ValueError("step must be >= 1")
    if lo > hi:
        lo, hi = hi, lo
    return list(range(int(lo), int(hi) + 1, int(step)))


# ============================================================
# Prompt extraction (standardized schema from utils.data loaders)
# ============================================================

def _get_prompt_text(item: dict) -> str:
    """
    Extract the prompt text from a standardised dataset item dict.

    All loaders produce items with a "question" key, but this function
    also handles fallback keys (prompt, instruction, text) for robustness.
    """
    if isinstance(item, dict):
        v = item.get("question", None)
        if v is not None:
            # Natural Questions loader may return a nested dict; extract text safely.
            if isinstance(v, dict):
                v = v.get("text", "")
            return str(v).strip()
        # fallback keys
        for k in ("prompt", "instruction", "text"):
            vv = item.get(k, None)
            if vv is None:
                continue
            if isinstance(vv, dict):
                vv = vv.get("text", "")
            s = str(vv).strip()
            if s:
                return s
    return str(item).strip()


# ============================================================
# BERTScore(prompt)
# ============================================================

def compute_bertscore_f1(
        refs: List[str],
        cands: List[str],
        *,
        batch_size: int = 32,
        device: Optional[str] = None,
        model_type: str = "roberta-large",
        lang: str = "en",
) -> np.ndarray:
    """
    Returns per-example BERTScore F1 as numpy array shape (N,).
    """
    try:
        from bert_score import score as bert_score
    except Exception as e:
        raise ImportError(
            "Missing dependency: bert-score. Install with:\n"
            "  pip install bert-score"
        ) from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.inference_mode():
        P, R, F1 = bert_score(
            cands,
            refs,
            model_type=model_type,
            lang=lang,
            batch_size=int(batch_size),
            device=device,
            verbose=False,
            rescale_with_baseline=False,
        )
    return F1.detach().cpu().numpy().astype(np.float32)


# ============================================================
# Dataset loading
# ============================================================

def load_dataset_prompts(
        dataset_name: str,
        *,
        sample_size: int,
        seed: int,
        harmbench_config: str,
        truthfulqa_config: str,
        truthfulqa_split: str,
        gsm8k_config: str,
        gsm8k_split: str,
        nq_config: str,
        nq_split: str,
) -> List[dict]:
    """
    Calls load_dataset_by_name with the appropriate kwargs for each dataset.
    """
    if dataset_name == "truthful_qa":
        return load_dataset_by_name(
            "truthful_qa",
            sample_size=int(sample_size),
            seed=int(seed),
            config_name=str(truthfulqa_config),
            split=str(truthfulqa_split),
        )

    if dataset_name == "alpaca":
        # The Alpaca loader auto-picks the available split when none is specified.
        return load_dataset_by_name(
            "alpaca",
            sample_size=int(sample_size),
            seed=int(seed),
        )

    if dataset_name == "harmbench":
        return load_dataset_by_name(
            "harmbench",
            sample_size=int(sample_size),
            seed=int(seed),
            config_name=str(harmbench_config),
        )

    if dataset_name == "gsm8k":
        return load_dataset_by_name(
            "gsm8k",
            sample_size=int(sample_size),
            seed=int(seed),
            config_name=str(gsm8k_config),
            split=str(gsm8k_split),
        )

    if dataset_name == "natural_questions":
        return load_dataset_by_name(
            "natural_questions",
            sample_size=int(sample_size),
            seed=int(seed),
            config_name=str(nq_config),
            split=str(nq_split),
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


# ============================================================
# Experiment
# ============================================================

DATASETS = ["truthful_qa", "alpaca", "harmbench", "gsm8k", "natural_questions"]


def run(
        *,
        sample_size: int,
        places: List[str],
        strengths: List[int],
        seed: int,
        bert_batch_size: int,
        bert_model_type: str,
        out_dir: str,
        save_pdf: bool,
        threshold: float,
        # dataset-specific configs
        harmbench_config: str,
        truthfulqa_config: str,
        truthfulqa_split: str,
        gsm8k_config: str,
        gsm8k_split: str,
        nq_config: str,
        nq_split: str,
):
    os.makedirs(out_dir, exist_ok=True)

    loaded: Dict[str, List[dict]] = {}
    print("[DATA] Loading datasets using utils.data.load_dataset_by_name ...")
    for ds in DATASETS:
        print(f"  - {ds}")
        loaded[ds] = load_dataset_prompts(
            ds,
            sample_size=sample_size,
            seed=seed,
            harmbench_config=harmbench_config,
            truthfulqa_config=truthfulqa_config,
            truthfulqa_split=truthfulqa_split,
            gsm8k_config=gsm8k_config,
            gsm8k_split=gsm8k_split,
            nq_config=nq_config,
            nq_split=nq_split,
        )
    print("[DATA] ✓ Done.\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    total = len(DATASETS) * len(places) * len(strengths)
    pbar = tqdm(total=total, desc="dataset×place×strength", unit="bucket")

    for ds in DATASETS:
        prompts = [_get_prompt_text(x) for x in loaded[ds]]

        for place in places:
            for s in strengths:
                s_int = int(s)

                styled = [apply_politeness(p, s_int, place=place) for p in prompts]

                f1 = compute_bertscore_f1(
                    refs=prompts,
                    cands=styled,
                    batch_size=int(bert_batch_size),
                    device=device,
                    model_type=str(bert_model_type),
                    lang="en",
                )
                mean_f1 = float(np.nanmean(f1))

                rows.append({
                    "dataset": ds,
                    "place": place,
                    "strength": s_int,
                    "bertscore_prompt": mean_f1,
                    "n": int(len(prompts)),
                })

                pbar.update(1)

    pbar.close()

    df = pd.DataFrame(rows).sort_values(["dataset", "place", "strength"]).reset_index(drop=True)

    csv_path = os.path.join(out_dir, "combined_means_by_dataset_place_strength.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[MAIN] ✓ Saved: {csv_path}")

    plot_path = os.path.join(out_dir, "bertscore_prompt_line.png")
    plot_bertscore_prompt_lines(
        df,
        out_path_png=plot_path,
        threshold=float(threshold),
        save_pdf=bool(save_pdf),
    )
    print(f"[MAIN] ✓ Saved: {plot_path}" + (" (+pdf)" if save_pdf else ""))

    thr_table = (
        df.assign(pass_thr=df["bertscore_prompt"] >= float(threshold))
        .groupby(["dataset", "place"], dropna=False)["pass_thr"]
        .mean()
        .reset_index()
        .rename(columns={"pass_thr": f"pass_rate_{threshold:.2f}"})
        .sort_values(["dataset", "place"])
    )
    thr_path = os.path.join(out_dir, f"threshold_pass_rates_{threshold:.2f}.csv")
    thr_table.to_csv(thr_path, index=False)
    print(f"[MAIN] ✓ Saved: {thr_path}")

    print(f"\n✓ Done. Results in: {out_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_size", type=int, default=256)
    parser.add_argument("--places", nargs="+", default=["prefix", "middle", "suffix", "global"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--strength_lo", type=int, default=-10)
    parser.add_argument("--strength_hi", type=int, default=10)
    parser.add_argument("--strength_step", type=int, default=2)

    parser.add_argument("--bert_batch_size", type=int, default=32)
    parser.add_argument("--bert_model_type", type=str, default="roberta-large")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--save_pdf", action="store_true")

    # Dataset-specific config args, matching the loaders in utils/data.py
    parser.add_argument("--harmbench_config", type=str, default="standard")

    parser.add_argument("--truthfulqa_config", type=str, default="generation")
    parser.add_argument("--truthfulqa_split", type=str, default="validation")

    parser.add_argument("--gsm8k_config", type=str, default="main")
    parser.add_argument("--gsm8k_split", type=str, default="test")

    parser.add_argument("--nq_config", type=str, default="default")
    parser.add_argument("--nq_split", type=str, default="validation")

    args = parser.parse_args()

    strengths = strength_grid(args.strength_lo, args.strength_hi, args.strength_step)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "results", "prompt_bertscore_all_datasets")
    out_dir = os.path.join(base, f"run_{timestamp}")

    run(
        sample_size=int(args.sample_size),
        places=list(args.places),
        strengths=strengths,
        seed=int(args.seed),
        bert_batch_size=int(args.bert_batch_size),
        bert_model_type=str(args.bert_model_type),
        out_dir=out_dir,
        save_pdf=bool(args.save_pdf),
        threshold=float(args.threshold),

        harmbench_config=str(args.harmbench_config),
        truthfulqa_config=str(args.truthfulqa_config),
        truthfulqa_split=str(args.truthfulqa_split),
        gsm8k_config=str(args.gsm8k_config),
        gsm8k_split=str(args.gsm8k_split),
        nq_config=str(args.nq_config),
        nq_split=str(args.nq_split),
    )


if __name__ == "__main__":
    main()