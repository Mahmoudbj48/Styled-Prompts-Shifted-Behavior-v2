# experiments/cot_generate_responses.py
"""
CoT Response Generation (Based on spacing.py structure)
========================================================

Generates model responses with "Let's think step by step" suffix.
NO metrics computation - just raw response generation and caching.

Analysis (step counting via LLM-as-judge) will be done in a separate script.

Key changes from spacing.py:
1. Accepts --style parameter (spacing, punctuation, letter_case, politeness)
2. Adds "Let's think step by step" suffix to all prompts
3. Removes all metrics computation (BERTScore, BLEU, activation, confidence)
4. Only saves: question, prompts, raw responses

Run:
  python experiments/cot_generate_responses.py \
    --models L3.2-3B \
    --dataset gsm8k \
    --sample_size 128 \
    --style spacing \
    --batch_size 16
"""

import argparse
import os
import sys
import yaml
import json
import gzip
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_dataset_by_name
from utils.models import load_model, generate_response
from utils.styles import apply_spacing, apply_punctuation, apply_letter_case, apply_politeness


VALID_STYLES = {"spacing", "punctuation", "letter_case", "politeness"}


# =============================================================================
# Config + CLI helpers
# =============================================================================

def load_config() -> Dict[str, any]:
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _normalize_models(models: List[str], config: dict) -> List[str]:
    available = list(config.get("models", {}).keys())
    if not available:
        raise ValueError("No models found in config.yaml under 'models'.")

    m = [x.strip() for x in models]
    if len(m) == 1 and m[0].lower() == "all":
        return available

    unknown = [x for x in m if x not in available]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Available: {available} or 'all'.")
    return m


def _get_places(config: dict, style: str) -> List[str]:
    """Get places for the given style from config."""
    places = config.get("style_positions", {}).get(style, ["global"])
    # Exclude 'middle' if present
    places = [p for p in places if p != "middle"]
    places = [p for p in places if p in {"prefix", "suffix", "global"}]
    if not places:
        places = ["global"]
    return places


def _num_batches(n: int, bs: int) -> int:
    return (n + bs - 1) // bs


def _select_strengths(
        *,
        config_strengths: List[int],
        explicit_strengths: Optional[List[int]],
        strength_range: Optional[Tuple[int, int]],
        strength_step: int,
) -> List[int]:
    """
    Priority:
      1) explicit_strengths
      2) strength_range (grid)
      3) config_strengths
    """
    if explicit_strengths:
        out, seen = [], set()
        for s in explicit_strengths:
            s = int(s)
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out

    if strength_range is not None:
        lo, hi = strength_range
        if strength_step <= 0:
            raise ValueError("--strength_step must be >= 1")
        if lo > hi:
            lo, hi = hi, lo
        return list(range(int(lo), int(hi) + 1, int(strength_step)))

    return [int(x) for x in config_strengths]


def _get_prompt_text(item: dict) -> str:
    if "question" in item and item["question"]:
        return str(item["question"])
    if "prompt" in item and item["prompt"]:
        return str(item["prompt"])
    return str(item)


def _get_style_function(style_name: str):
    """Map style name to style function."""
    style_map = {
        "spacing": apply_spacing,
        "punctuation": apply_punctuation,
        "letter_case": apply_letter_case,
        "politeness": apply_politeness,
    }
    if style_name not in style_map:
        raise ValueError(f"Unknown style: {style_name}. Available: {list(style_map.keys())}")
    return style_map[style_name]


# =============================================================================
# Data cache helpers
# =============================================================================

def _safe_name(x: Optional[str]) -> str:
    if x is None:
        return "none"
    x = str(x)
    return "".join([c if c.isalnum() or c in ("-", "_", ".", "=") else "_" for c in x])


def _sample_cache_dir(
        *,
        data_dir: str,
        dataset: str,
        config_name: Optional[str],
        split: str,
        seed: int,
        sample_size: int,
) -> str:
    return os.path.join(
        data_dir,
        "samples",
        _safe_name(dataset),
        f"config_{_safe_name(config_name)}",
        f"split_{_safe_name(split)}",
        f"seed_{seed}",
        f"n_{sample_size}",
    )


def _sample_cache_path(sample_dir: str) -> str:
    return os.path.join(sample_dir, "sample.jsonl")


def _meta_path(sample_dir: str) -> str:
    return os.path.join(sample_dir, "meta.yaml")


def _write_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_yaml(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _load_or_create_sample(
        *,
        data_dir: str,
        dataset: str,
        config_name: Optional[str],
        split: str,
        seed: int,
        sample_size: int,
        overwrite_sample_cache: bool,
) -> List[dict]:
    sdir = _sample_cache_dir(
        data_dir=data_dir,
        dataset=dataset,
        config_name=config_name,
        split=split,
        seed=seed,
        sample_size=sample_size,
    )
    spath = _sample_cache_path(sdir)
    mpath = _meta_path(sdir)

    if os.path.exists(spath) and (not overwrite_sample_cache):
        items = _read_jsonl(spath)
        if len(items) == sample_size:
            return items

    items = load_dataset_by_name(
        dataset,
        sample_size=sample_size,
        seed=seed,
        config_name=config_name,
        split=split,
    )
    os.makedirs(sdir, exist_ok=True)
    _write_jsonl(spath, items)
    _write_yaml(mpath, {
        "dataset": dataset,
        "config_name": config_name,
        "split": split,
        "seed": seed,
        "sample_size": sample_size,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    return items


def _outputs_cache_path(
        *,
        data_dir: str,
        dataset: str,
        config_name: Optional[str],
        split: str,
        seed: int,
        sample_size: int,
        model_name: str,
        style: str,
        place: str,
        strength: int,
) -> str:
    return os.path.join(
        data_dir,
        "cot_outputs_cache",
        _safe_name(dataset),
        f"config_{_safe_name(config_name)}",
        f"split_{_safe_name(split)}",
        f"seed_{seed}",
        f"n_{sample_size}",
        _safe_name(model_name),
        _safe_name(style),
        _safe_name(place),
        f"strength_{int(strength)}.jsonl.gz",
    )


def _read_jsonl_gz(path: str) -> List[dict]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonl_gz(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_or_generate_outputs_for_bucket(
        *,
        cache_path: str,
        overwrite_output_cache: bool,
        model,
        tokenizer,
        prompt_orig_list: List[str],
        prompt_styled_list: List[str],
        max_new_tokens: int,
        batch_size: int,
) -> Dict[str, List[str]]:
    """
    Returns dict with:
      - output_orig_raw
      - output_styled_raw
    Loads from cache if exists and not overwritten.
    """
    n = len(prompt_orig_list)

    if os.path.exists(cache_path) and (not overwrite_output_cache):
        rows = _read_jsonl_gz(cache_path)
        if len(rows) == n:
            return {
                "output_orig_raw": [r["output_orig_raw"] for r in rows],
                "output_styled_raw": [r["output_styled_raw"] for r in rows],
            }

    out_orig_raw = generate_response(
        model, tokenizer,
        prompts=prompt_orig_list,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    out_styled_raw = generate_response(
        model, tokenizer,
        prompts=prompt_styled_list,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    if len(out_orig_raw) != n or len(out_styled_raw) != n:
        raise RuntimeError("generate_response returned wrong number of outputs when caching outputs.")

    rows = []
    for i in range(n):
        rows.append({
            "prompt_id": i,
            "prompt_orig": prompt_orig_list[i],
            "output_orig_raw": out_orig_raw[i],
            "prompt_styled": prompt_styled_list[i],
            "output_styled_raw": out_styled_raw[i],
        })
    _write_jsonl_gz(cache_path, rows)

    return {
        "output_orig_raw": out_orig_raw,
        "output_styled_raw": out_styled_raw,
    }


# =============================================================================
# Main experiment per model (SIMPLIFIED - NO METRICS)
# =============================================================================

def run_for_one_model(
        model_name: str,
        model_path: str,
        items: List[dict],
        strength_levels: List[int],
        places: List[str],
        style_name: str,
        config: dict,
        run_dir: str,
        *,
        batch_size: int,
        max_new_tokens: int,
        data_dir: str,
        dataset_name: str,
        dataset_config_name: Optional[str],
        dataset_split: str,
        dataset_seed: int,
        dataset_sample_size: int,
        overwrite_output_cache: bool,
) -> pd.DataFrame:
    """
    Generate CoT responses with "Let's think step by step" suffix.
    NO metrics computation - just raw generation.
    """

    # Load model
    model, tokenizer = load_model(
        model_path,
        device_map="auto",
        dtype="float16",
    )

    style_fn_base = _get_style_function(style_name)

    rows: List[dict] = []

    prompts_text = [_get_prompt_text(it) for it in items]
    categories = [it.get("category", "Unknown") for it in items]

    n = len(prompts_text)
    n_batches = _num_batches(n, batch_size)

    total_groups = n_batches * len(places) * len(strength_levels)
    batch_pbar = tqdm(total=total_groups, desc=f"[{model_name}] batch-groups", unit="group")

    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n)

        batch_ids = list(range(start, end))
        batch_orig_prompts = prompts_text[start:end]
        batch_categories = categories[start:end]

        # =================================================================
        # ADD COT SUFFIX: "Let's think step by step"
        # =================================================================
        batch_orig_prompts_cot = [p + "\n\nLet's think step by step" for p in batch_orig_prompts]

        for place in places:
            for strength in strength_levels:
                
                # Apply style to CoT-prompted questions
                # Fix lambda closure with default arguments
                if strength == 0:
                    style_fn = None
                else:
                    if style_name == "politeness":
                        style_fn = lambda p, s=strength: style_fn_base(p, s)
                    else:
                        style_fn = lambda p, s=strength, pl=place: style_fn_base(p, s, place=pl)
                
                batch_styled_prompts_cot = []
                for prompt_cot in batch_orig_prompts_cot:
                    if style_fn is None:
                        batch_styled_prompts_cot.append(prompt_cot)
                    else:
                        batch_styled_prompts_cot.append(style_fn(prompt_cot))

                # =================================================================
                # GENERATE OR LOAD FROM CACHE
                # =================================================================
                
                cache_path = _outputs_cache_path(
                    data_dir=data_dir,
                    dataset=dataset_name,
                    config_name=dataset_config_name,
                    split=dataset_split,
                    seed=dataset_seed,
                    sample_size=dataset_sample_size,
                    model_name=model_name,
                    style=style_name,
                    place=place,
                    strength=int(strength),
                )
                
                out = _load_or_generate_outputs_for_bucket(
                    cache_path=cache_path,
                    overwrite_output_cache=overwrite_output_cache,
                    model=model,
                    tokenizer=tokenizer,
                    prompt_orig_list=batch_orig_prompts_cot,
                    prompt_styled_list=batch_styled_prompts_cot,
                    max_new_tokens=max_new_tokens,
                    batch_size=batch_size,
                )

                # =================================================================
                # SAVE METADATA (NO METRICS - JUST RAW DATA)
                # =================================================================
                
                for j in range(len(batch_orig_prompts)):
                    i = batch_ids[j]
                    
                    row = {
                        "model": model_name,
                        "style": style_name,
                        "problem_id": i,
                        "category": batch_categories[j],
                        "strength": int(strength),
                        "place": place,
                        
                        # Original question (without CoT suffix)
                        "question_original": batch_orig_prompts[j],
                        
                        # Prompts (with CoT suffix "Let's think step by step")
                        "prompt_original_cot": batch_orig_prompts_cot[j],
                        "prompt_styled_cot": batch_styled_prompts_cot[j],
                        
                        # Raw responses (no cleaning, no parsing)
                        "response_original": out["output_orig_raw"][j],
                        "response_styled": out["output_styled_raw"][j],
                    }
                    
                    rows.append(row)

                batch_pbar.update(1)

    batch_pbar.close()

    df = pd.DataFrame(rows)

    out_csv = os.path.join(run_dir, f"{model_name}_responses.csv")
    df.to_csv(out_csv, index=False)
    print(f"✓ Saved responses: {out_csv}")

    return df


# =============================================================================
# Global runner (multi-model)
# =============================================================================

def run_experiment(
        models: List[str],
        dataset_name: str,
        sample_size: Optional[int],
        style_name: str,
        *,
        batch_size: int,
        max_new_tokens: Optional[int],
        strengths_explicit: Optional[List[int]],
        strength_range: Optional[Tuple[int, int]],
        strength_step: int,
        data_dir: str,
        overwrite_sample_cache: bool,
        overwrite_output_cache: bool,
        places_override: Optional[List[str]],
) -> str:
    config = load_config()

    if dataset_name not in config["datasets"]:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(config['datasets'].keys())}")
    dataset_config = config["datasets"][dataset_name]

    if sample_size is None:
        sample_size = int(dataset_config["sample_size"])

    places = places_override if (places_override and len(places_override) > 0) else _get_places(config, style_name)

    strength_levels = _select_strengths(
        config_strengths=config["style_levels"][style_name],
        explicit_strengths=strengths_explicit,
        strength_range=strength_range,
        strength_step=strength_step,
    )

    if max_new_tokens is None:
        max_new_tokens = int(config["defaults"].get("max_new_tokens_cot", 200))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    style_dir = os.path.join(base_results_dir, "cot_responses")
    run_dir = os.path.join(style_dir, f"run_{dataset_name}_{style_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("COT RESPONSE GENERATION (NO METRICS)")
    print(f"{'='*80}")
    print(f"Models: {models}")
    print(f"Dataset: {dataset_name}")
    print(f"Style: {style_name}")
    print(f"Sample size: {sample_size}")
    print(f"Strengths: {strength_levels}")
    print(f"Places: {places}")
    print(f"Batch size: {batch_size}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"CoT suffix: 'Let's think step by step'")
    print(f"Data cache dir: {data_dir}")
    print(f"Output dir: {run_dir}")
    print(f"{'='*80}\n")

    items = _load_or_create_sample(
        data_dir=data_dir,
        dataset=dataset_name,
        config_name=dataset_config.get("config_name"),
        split=dataset_config.get("split", "validation"),
        seed=int(config["defaults"]["random_seed"]),
        sample_size=int(sample_size),
        overwrite_sample_cache=bool(overwrite_sample_cache),
    )

    all_rows = []
    for model_name in models:
        model_path = config["models"][model_name]
        df_model = run_for_one_model(
            model_name=model_name,
            model_path=model_path,
            items=items,
            strength_levels=strength_levels,
            places=places,
            style_name=style_name,
            config=config,
            run_dir=run_dir,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            data_dir=data_dir,
            dataset_name=dataset_name,
            dataset_config_name=dataset_config.get("config_name"),
            dataset_split=dataset_config.get("split", "validation"),
            dataset_seed=int(config["defaults"]["random_seed"]),
            dataset_sample_size=int(sample_size),
            overwrite_output_cache=bool(overwrite_output_cache),
        )
        if df_model is not None and not df_model.empty:
            all_rows.append(df_model)

    if not all_rows:
        print("No outputs produced.")
        return run_dir

    df_all = pd.concat(all_rows, ignore_index=True)

    full_path = os.path.join(run_dir, "all_models_responses.csv")
    df_all.to_csv(full_path, index=False)
    print(f"✓ Saved combined responses: {full_path}")
    
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total responses: {len(df_all)}")
    print(f"Cached in: {data_dir}/cot_outputs_cache/")
    print(f"Results CSV: {full_path}")
    print(f"\nNext step: Run LLM-as-judge script to count reasoning steps")
    print(f"{'='*80}\n")

    return run_dir


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate CoT responses (no metrics)")
    parser.add_argument("--models", nargs="+", default=["L3.2-3B"], help="Model keys from config or 'all'")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--style", type=str, required=True, choices=list(VALID_STYLES),
                        help="Style to apply: spacing, punctuation, letter_case, politeness")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=None)

    parser.add_argument("--places", nargs="+", default=None, help="Override places for run")

    parser.add_argument("--strengths", nargs="+", type=int, default=None)
    parser.add_argument("--strength_range", nargs=2, type=int, default=None, metavar=("LO", "HI"))
    parser.add_argument("--strength_step", type=int, default=1)

    parser.add_argument("--data_dir", type=str, default=None,
                        help="Base data dir for caches (default=../data)")
    parser.add_argument("--overwrite_sample_cache", action="store_true")
    parser.add_argument("--overwrite_output_cache", action="store_true")

    args = parser.parse_args()

    config = load_config()
    models = _normalize_models(args.models, config)

    data_dir = args.data_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    run_experiment(
        models=models,
        dataset_name=args.dataset,
        sample_size=args.sample_size,
        style_name=args.style,
        batch_size=int(args.batch_size),
        max_new_tokens=args.max_new_tokens,
        strengths_explicit=args.strengths,
        strength_range=tuple(args.strength_range) if args.strength_range else None,
        strength_step=int(args.strength_step),
        data_dir=data_dir,
        overwrite_sample_cache=bool(args.overwrite_sample_cache),
        overwrite_output_cache=bool(args.overwrite_output_cache),
        places_override=args.places,
    )


if __name__ == "__main__":
    main()