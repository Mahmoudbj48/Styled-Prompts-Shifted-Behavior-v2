# experiments/cot_reasoning_generate.py
"""
Chain-of-Thought Response Generation (EFFICIENT)
================================================

This script ONLY generates responses and caches them.
Parsing and analysis are done separately (see cot_reasoning_analyze.py).

Key optimizations:
1. ✅ Generates baseline (strength=0) ONCE per place
2. ✅ Reuses baseline for all strength comparisons
3. ✅ Batched inference for efficiency
4. ✅ Output caching (resume from crashes)
5. ✅ Identical pattern to spacing.py/politeness.py

Run:
  python experiments/cot_reasoning_generate.py \
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_dataset_by_name
from utils.models import load_model, generate_response
from utils.metrics import add_cot_prompt
from utils.styles import apply_spacing, apply_punctuation, apply_letter_case, apply_politeness


VALID_STYLES = {"spacing", "punctuation", "letter_case", "politeness"}


# =============================================================================
# Helper Functions (same as other experiments)
# =============================================================================

def load_config() -> Dict[str, any]:
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _safe_name(x: Optional[str]) -> str:
    if x is None:
        return "none"
    x = str(x)
    return "".join([c if c.isalnum() or c in ("-", "_", ".", "=") else "_" for c in x])


def _normalize_models(models: List[str], config: dict) -> List[str]:
    available = list(config.get("models", {}).keys())
    if not available:
        raise ValueError("No models found in config.yaml")
    
    m = [x.strip() for x in models]
    if len(m) == 1 and m[0].lower() == "all":
        return available
    
    unknown = [x for x in m if x not in available]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Available: {available}")
    return m


def _get_places(config: dict, style: str) -> List[str]:
    places = config.get("style_positions", {}).get(style, ["global"])
    places = [p for p in places if p in {"prefix", "suffix", "global"}]
    if not places:
        places = ["global"]
    return places


def _select_strengths(
        *,
        config: dict,
        style: str,
        explicit_strengths: Optional[List[int]],
        strength_range: Optional[Tuple[int, int]],
        strength_step: int,
) -> List[int]:
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
    
    config_strengths = config.get("style_levels", {}).get(style, [0, 20, 50, 100])
    return [int(x) for x in config_strengths]


def _get_prompt_text(item: dict) -> str:
    if "question" in item and item["question"]:
        return str(item["question"])
    if "prompt" in item and item["prompt"]:
        return str(item["prompt"])
    return str(item)


def _get_style_function(style_name: str):
    style_map = {
        "spacing": apply_spacing,
        "punctuation": apply_punctuation,
        "letter_case": apply_letter_case,
        "politeness": apply_politeness,
    }
    if style_name not in style_map:
        raise ValueError(f"Unknown style: {style_name}")
    return style_map[style_name]


# =============================================================================
# Sample Caching
# =============================================================================

def _sample_cache_dir(*, data_dir, dataset, config_name, split, seed, sample_size):
    return os.path.join(
        data_dir, "samples", _safe_name(dataset),
        f"config_{_safe_name(config_name)}",
        f"split_{_safe_name(split)}",
        f"seed_{seed}", f"n_{sample_size}",
    )


def _write_jsonl(path: str, rows: List[dict]):
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


def _write_yaml(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _load_or_create_sample(*, data_dir, dataset, config_name, split, seed, sample_size, overwrite_sample_cache):
    sdir = _sample_cache_dir(
        data_dir=data_dir, dataset=dataset, config_name=config_name,
        split=split, seed=seed, sample_size=sample_size
    )
    spath = os.path.join(sdir, "sample.jsonl")
    mpath = os.path.join(sdir, "meta.yaml")
    
    if os.path.exists(spath) and not overwrite_sample_cache:
        items = _read_jsonl(spath)
        if len(items) == sample_size:
            return items
    
    items = load_dataset_by_name(
        dataset, sample_size=sample_size, seed=seed,
        config_name=config_name, split=split
    )
    os.makedirs(sdir, exist_ok=True)
    _write_jsonl(spath, items)
    _write_yaml(mpath, {
        "dataset": dataset, "config_name": config_name,
        "split": split, "seed": seed, "sample_size": sample_size,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    return items


# =============================================================================
# Output Caching (SAME PATTERN AS SPACING.PY)
# =============================================================================

def _outputs_cache_path(*, data_dir, dataset, config_name, split, seed, sample_size,
                        model_name, style, place, strength):
    return os.path.join(
        data_dir, "cot_outputs_cache", _safe_name(dataset),
        f"config_{_safe_name(config_name)}",
        f"split_{_safe_name(split)}",
        f"seed_{seed}", f"n_{sample_size}",
        _safe_name(model_name), _safe_name(style),
        _safe_name(place), f"strength_{int(strength)}.jsonl.gz",
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


def _write_jsonl_gz(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_or_generate_outputs_for_bucket(
        *, cache_path, overwrite_output_cache, model, tokenizer,
        prompt_orig_list, prompt_styled_list, max_new_tokens, batch_size):
    """
    SAME FUNCTION AS SPACING.PY/POLITENESS.PY
    
    Returns:
        Dict with output_orig_raw, output_styled_raw
    """
    n = len(prompt_orig_list)
    
    # Try to load from cache
    if os.path.exists(cache_path) and not overwrite_output_cache:
        rows = _read_jsonl_gz(cache_path)
        if len(rows) == n:
            return {
                "output_orig_raw": [r["output_orig_raw"] for r in rows],
                "output_styled_raw": [r["output_styled_raw"] for r in rows],
            }
    
    # Generate new outputs
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
        raise RuntimeError("generate_response returned wrong number of outputs")
    
    # Cache results
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
# Main Experiment (FOLLOWS SPACING.PY PATTERN EXACTLY)
# =============================================================================

def run_for_one_model(
        model_name, model_path, items, strength_levels, places,
        style_name, config, run_dir, *,
        batch_size, max_new_tokens, data_dir,
        dataset_name, dataset_config_name, dataset_split,
        dataset_seed, dataset_sample_size, overwrite_output_cache):
    """
    Generate CoT responses following the EXACT pattern from spacing.py.
    
    Key optimization: Generates baseline ONCE per batch, reuses for comparisons.
    """
    
    # Load model
    model, tokenizer = load_model(
        model_path,
        device_map=config["defaults"].get("device_map", "auto"),
        dtype=config["defaults"].get("dtype", "float32"),
    )
    
    style_fn_base = _get_style_function(style_name)
    questions = [_get_prompt_text(it) for it in items]
    categories = [it.get("category", "Unknown") for it in items]
    
    n = len(questions)
    rows: List[dict] = []
    
    # =================================================================
    # BATCHED PATTERN (same as spacing.py)
    # =================================================================
    # Process in batches to avoid memory issues with large datasets
    # =================================================================
    
    def _num_batches(n: int, bs: int) -> int:
        return (n + bs - 1) // bs
    
    n_batches = _num_batches(n, batch_size)
    total_groups = n_batches * len(places) * len(strength_levels)
    
    batch_pbar = tqdm(total=total_groups, desc=f"[{model_name}] batch-groups", unit="group")
    
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n)
        
        batch_ids = list(range(start, end))
        batch_orig_prompts = questions[start:end]
        batch_categories = categories[start:end]
        
        # Add CoT instruction to all questions
        batch_orig_prompts_cot = [add_cot_prompt(q) for q in batch_orig_prompts]
        
        for place in places:
            for strength in strength_levels:
                
                # =================================================================
                # FIX: Lambda closure with default arguments
                # =================================================================
                style_fn = None
                if strength != 0:
                    if style_name == "politeness":
                        style_fn = lambda q, s=strength: style_fn_base(q, s)
                    else:
                        style_fn = lambda q, s=strength, p=place: style_fn_base(q, s, place=p)
                
                # Apply style to CoT-prompted questions
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
                # SAVE METADATA (for later analysis)
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
                        
                        # Original question
                        "question_original": batch_orig_prompts[j],
                        
                        # Prompts (with CoT instruction)
                        "prompt_original_cot": batch_orig_prompts_cot[j],
                        "prompt_styled_cot": batch_styled_prompts_cot[j],
                        
                        # Responses (raw, unparsed)
                        "response_original": out["output_orig_raw"][j],
                        "response_styled": out["output_styled_raw"][j],
                    }
                    
                    rows.append(row)
                
                batch_pbar.update(1)
    
    batch_pbar.close()
    
    # Save results
    df = pd.DataFrame(rows)
    out_csv = os.path.join(run_dir, f"{model_name}_responses.csv")
    df.to_csv(out_csv, index=False)
    print(f"✓ Saved responses: {out_csv}")
    
    return df


# =============================================================================
# Multi-Model Runner
# =============================================================================

def run_experiment(
        models, dataset_name, sample_size, style_name, *,
        batch_size, max_new_tokens, strengths_explicit,
        strength_range, strength_step, data_dir,
        overwrite_sample_cache, overwrite_output_cache, places_override):
    
    config = load_config()
    
    if dataset_name not in config["datasets"]:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    dataset_config = config["datasets"][dataset_name]
    
    if sample_size is None:
        sample_size = int(dataset_config["sample_size"])
    
    places = places_override if places_override else _get_places(config, style_name)
    
    strength_levels = _select_strengths(
        config=config, style=style_name,
        explicit_strengths=strengths_explicit,
        strength_range=strength_range,
        strength_step=strength_step,
    )
    
    if max_new_tokens is None:
        max_new_tokens = int(config["defaults"].get("max_new_tokens_cot", 100))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    style_dir = os.path.join(base_results_dir, "cot_reasoning")
    run_dir = os.path.join(style_dir, f"run_responses_{dataset_name}_{style_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("COT RESPONSE GENERATION (EFFICIENT)")
    print(f"{'='*80}")
    print(f"Models: {models}")
    print(f"Dataset: {dataset_name}")
    print(f"Style: {style_name}")
    print(f"Sample size: {sample_size}")
    print(f"Strengths: {strength_levels}")
    print(f"Places: {places}")
    print(f"Batch size: {batch_size}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Data cache dir: {data_dir}")
    print(f"Output dir: {run_dir}")
    print(f"{'='*80}\n")
    
    # Load sample
    items = _load_or_create_sample(
        data_dir=data_dir,
        dataset=dataset_name,
        config_name=dataset_config.get("config_name"),
        split=dataset_config.get("split", "test"),
        seed=int(config["defaults"]["random_seed"]),
        sample_size=int(sample_size),
        overwrite_sample_cache=bool(overwrite_sample_cache),
    )
    
    # Run for each model
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
            dataset_split=dataset_config.get("split", "test"),
            dataset_seed=int(config["defaults"]["random_seed"]),
            dataset_sample_size=int(sample_size),
            overwrite_output_cache=bool(overwrite_output_cache),
        )
        if df_model is not None and not df_model.empty:
            all_rows.append(df_model)
    
    if not all_rows:
        print("No outputs produced")
        return run_dir
    
    # Combine all models
    df_all = pd.concat(all_rows, ignore_index=True)
    full_path = os.path.join(run_dir, "all_models_responses.csv")
    df_all.to_csv(full_path, index=False)
    print(f"✓ Saved combined responses: {full_path}")
    
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total responses generated: {len(df_all)}")
    print(f"Cached in: {data_dir}/cot_outputs_cache/")
    print(f"Results CSV: {full_path}")
    print(f"\nNext step: Run cot_reasoning_analyze.py to parse and analyze")
    print(f"{'='*80}\n")
    
    return run_dir


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate CoT responses (efficient)")
    parser.add_argument("--models", nargs="+", default=["L3.2-3B"])
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--style", type=str, required=True, choices=list(VALID_STYLES))
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    
    parser.add_argument("--places", nargs="+", default=None)
    parser.add_argument("--strengths", nargs="+", type=int, default=None)
    parser.add_argument("--strength_range", nargs=2, type=int, default=None)
    parser.add_argument("--strength_step", type=int, default=1)
    
    parser.add_argument("--data_dir", type=str, default=None)
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