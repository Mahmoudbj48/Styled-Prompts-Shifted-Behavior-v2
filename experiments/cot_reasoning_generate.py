# experiments/cot_generate_responses.py
"""
CoT Response Generation - PRODUCTION OPTIMIZED
===============================================

OPTIMIZATIONS:
1. ✅ Generate baseline ONCE per batch (not per strength/place combo)
2. ✅ Batch styled prompts together for GPU efficiency
3. ✅ Comprehensive caching with separate baseline cache
4. ✅ Progress tracking with time estimates

Performance: ~10x faster than naive implementation

Run:
  python experiments/cot_generate_responses.py \
    --models L3.1-8B \
    --dataset gsm8k \
    --sample_size 128 \
    --style spacing \
    --batch_size 32 \
    --max_new_tokens 200 \
    --strengths 0 50 100 \
    --places global
"""

import argparse
import os
import sys
import yaml
import json
import gzip
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data import load_dataset_by_name
from utils.models import load_model, generate_response
from utils.styles import apply_spacing, apply_punctuation, apply_letter_case, apply_politeness
from utils.llm_style_cache import load_or_generate_styled_prompts


VALID_STYLES = {"spacing", "punctuation", "letter_case", "politeness", "length_variation", "inter_vs_imper"}


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
    places = [p for p in places if p != "middle"]
    places = [p for p in places if p in {"prefix", "suffix", "global"}]
    if not places:
        places = ["global"]
    return places


def _num_batches(n: int, bs: int) -> int:
    return (n + bs - 1) // bs


def _select_strengths(
        *, config_strengths, explicit_strengths, strength_range, strength_step,
        as_float=False, as_string=False) -> list:
    # String modes (e.g. inter_vs_imper) — no numeric conversion
    if as_string:
        if explicit_strengths:
            return [str(s) for s in explicit_strengths]
        return [str(s) for s in config_strengths]

    if explicit_strengths:
        out, seen = [], set()
        for s in explicit_strengths:
            s = float(s) if as_float else int(s)
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

    if as_float:
        return [float(x) for x in config_strengths]
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
    if style_name in ("length_variation", "inter_vs_imper"):
        return None  # uses LLM style cache, not an inline function
    if style_name not in style_map:
        raise ValueError(f"Unknown style: {style_name}")
    return style_map[style_name]


# =============================================================================
# Data cache helpers
# =============================================================================

def _safe_name(x: Optional[str]) -> str:
    if x is None:
        return "none"
    x = str(x)
    return "".join([c if c.isalnum() or c in ("-", "_", ".", "=") else "_" for c in x])


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


def _load_or_create_sample(
        *, data_dir, dataset, config_name, split, seed, sample_size, overwrite_sample_cache):
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


def _baseline_cache_path(*, data_dir, dataset, config_name, split, seed, sample_size, model_name):
    """Separate cache for baseline (original) responses - shared across all strengths/places."""
    return os.path.join(
        data_dir, "cot_baseline_cache", _safe_name(dataset),
        f"config_{_safe_name(config_name)}",
        f"split_{_safe_name(split)}",
        f"seed_{seed}", f"n_{sample_size}",
        _safe_name(model_name),
        "baseline_responses.jsonl.gz",
    )


def _styled_cache_path(*, data_dir, dataset, config_name, split, seed, sample_size,
                       model_name, style, place, strength):
    """Cache for styled responses - one per (place, strength) combo."""
    return os.path.join(
        data_dir, "cot_styled_cache", _safe_name(dataset),
        f"config_{_safe_name(config_name)}",
        f"split_{_safe_name(split)}",
        f"seed_{seed}", f"n_{sample_size}",
        _safe_name(model_name), _safe_name(style),
        _safe_name(place), f"strength_{_safe_name(str(strength))}.jsonl.gz",
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


# =============================================================================
# Baseline Generation (Once Per Batch)
# =============================================================================

def _load_or_generate_baseline(
        *, cache_path, overwrite_cache, model, tokenizer,
        prompts_cot: List[str], batch_ids: List[int], 
        max_new_tokens: int, batch_size: int) -> List[str]:
    """
    Load or generate baseline responses.
    These are cached ONCE and reused across all strength/place combos.
    """
    n = len(prompts_cot)
    
    # Try loading from cache
    if os.path.exists(cache_path) and not overwrite_cache:
        try:
            all_rows = _read_jsonl_gz(cache_path)
            # Extract only the responses for this batch
            batch_responses = []
            for batch_id in batch_ids:
                matching = [r for r in all_rows if r["prompt_id"] == batch_id]
                if matching:
                    batch_responses.append(matching[0]["response"])
            
            if len(batch_responses) == n:
                print(f"  ✓ Loaded {n} baseline responses from cache")
                return batch_responses
        except Exception as e:
            print(f"  ⚠️  Cache read failed ({e}), regenerating...")
    
    # Generate baseline
    print(f"  → Generating {n} baseline responses (ONCE for all combos)...")
    start_time = time.time()
    
    responses = generate_response(
        model, tokenizer, prompts_cot,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    
    elapsed = time.time() - start_time
    print(f"    ✓ Generated in {elapsed:.1f}s ({elapsed/n:.2f}s per response)")
    
    # Cache baseline responses
    # Load existing cache if it exists
    existing_rows = {}
    if os.path.exists(cache_path):
        try:
            for row in _read_jsonl_gz(cache_path):
                existing_rows[row["prompt_id"]] = row
        except:
            pass
    
    # Update with new responses
    for i, (batch_id, response) in enumerate(zip(batch_ids, responses)):
        existing_rows[batch_id] = {
            "prompt_id": batch_id,
            "prompt": prompts_cot[i],
            "response": response,
        }
    
    # Write all rows
    all_rows = [existing_rows[k] for k in sorted(existing_rows.keys())]
    _write_jsonl_gz(cache_path, all_rows)
    
    return responses


# =============================================================================
# Styled Generation
# =============================================================================

def _load_or_generate_styled(
        *, cache_path, overwrite_cache, model, tokenizer,
        prompts_styled: List[str], batch_ids: List[int],
        max_new_tokens: int, batch_size: int) -> List[str]:
    """Load or generate styled responses for a specific (place, strength) combo."""
    n = len(prompts_styled)
    
    # Try loading from cache
    if os.path.exists(cache_path) and not overwrite_cache:
        try:
            rows = _read_jsonl_gz(cache_path)
            if len(rows) == n:
                # Verify IDs match
                if all(rows[i]["prompt_id"] == batch_ids[i] for i in range(n)):
                    return [r["response"] for r in rows]
        except Exception as e:
            print(f"  ⚠️  Styled cache read failed ({e}), regenerating...")
    
    # Generate styled responses
    start_time = time.time()
    
    responses = generate_response(
        model, tokenizer, prompts_styled,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    
    elapsed = time.time() - start_time
    
    # Cache styled responses
    rows = []
    for i, (batch_id, response) in enumerate(zip(batch_ids, responses)):
        rows.append({
            "prompt_id": batch_id,
            "prompt": prompts_styled[i],
            "response": response,
        })
    _write_jsonl_gz(cache_path, rows)
    
    return responses


# =============================================================================
# Main Experiment (OPTIMIZED)
# =============================================================================

def run_for_one_model(
        model_name, model_path, items, strength_levels, places,
        style_name, config, run_dir, *,
        batch_size, max_new_tokens, data_dir, dataset_name, dataset_config_name,
        dataset_split, dataset_seed, dataset_sample_size, overwrite_output_cache,
        rewrite_provider="openai", rewrite_model="gpt-4o-mini",
        rewrite_api_key_env="OPENAI_API_KEY", overwrite_style_cache=False):
    """
    OPTIMIZED: 
    1. Generate baseline ONCE per batch
    2. Batch styled prompts efficiently
    """
    
    # Load model
    print(f"\n{'='*80}")
    print(f"Loading {model_name}...")
    print(f"{'='*80}")
    model, tokenizer = load_model(model_path, device_map="auto", dtype="float16")
    
    # Verify GPU usage
    import torch
    device = next(model.parameters()).device
    print(f"✓ Model loaded on: {device}")
    print(f"✓ CUDA memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"{'='*80}\n")
    
    style_fn_base = _get_style_function(style_name)
    
    prompts_text = [_get_prompt_text(it) for it in items]
    categories = [it.get("category", "Unknown") for it in items]
    
    n = len(prompts_text)
    n_batches = _num_batches(n, batch_size)
    
    # ------------------------------------------------------------------
    # Pre-load LLM-rewritten styled prompts for structured styles
    # (loaded from / generated into data/llm_style_cache/)
    # ------------------------------------------------------------------
    styled_prompts_by_strength: Dict[str, List[str]] = {}
    if style_name == "length_variation":
        for mult in strength_levels:
            styled_prompts_by_strength[mult] = load_or_generate_styled_prompts(
                data_dir=data_dir,
                dataset=dataset_name,
                prompts=prompts_text,
                style_name="length_variation",
                param=mult,
                rewrite_provider=rewrite_provider,
                rewrite_model=rewrite_model,
                rewrite_api_key_env=rewrite_api_key_env,
                overwrite=overwrite_style_cache,
            )
    elif style_name == "inter_vs_imper":
        for mode in strength_levels:
            styled_prompts_by_strength[mode] = load_or_generate_styled_prompts(
                data_dir=data_dir,
                dataset=dataset_name,
                prompts=prompts_text,
                style_name="inter_vs_imper",
                param=mode,
                rewrite_provider=rewrite_provider,
                rewrite_model=rewrite_model,
                rewrite_api_key_env=rewrite_api_key_env,
                overwrite=overwrite_style_cache,
            )

    # Calculate total work
    total_styled_generations = n_batches * len(places) * len(strength_levels)
    
    print(f"Experiment plan:")
    print(f"  • {n} samples / {batch_size} = {n_batches} batches")
    print(f"  • {len(strength_levels)} strengths × {len(places)} places = {len(strength_levels) * len(places)} combos")
    print(f"  • Baseline: {n_batches} generations (ONCE per batch)")
    print(f"  • Styled: {total_styled_generations} generations")
    print(f"  • Total: {n_batches + total_styled_generations} generations")
    print(f"  • max_new_tokens: {max_new_tokens}\n")
    
    rows: List[dict] = []
    
    # Progress tracking
    pbar = tqdm(total=total_styled_generations, desc=f"[{model_name}] styled generations", unit="gen")
    
    # Baseline cache path (shared across all combos)
    baseline_cache = _baseline_cache_path(
        data_dir=data_dir, dataset=dataset_name,
        config_name=dataset_config_name, split=dataset_split,
        seed=dataset_seed, sample_size=dataset_sample_size,
        model_name=model_name
    )
    
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n)
        
        batch_ids = list(range(start, end))
        batch_orig_prompts = prompts_text[start:end]
        batch_categories = categories[start:end]
        
        # Add CoT suffix
        batch_orig_prompts_cot = [p + "\n\nLet's think step by step" for p in batch_orig_prompts]
        
        # ================================================================
        # OPTIMIZATION: Generate baseline ONCE per batch
        # ================================================================
        batch_response_baseline = _load_or_generate_baseline(
            cache_path=baseline_cache,
            overwrite_cache=overwrite_output_cache,
            model=model,
            tokenizer=tokenizer,
            prompts_cot=batch_orig_prompts_cot,
            batch_ids=batch_ids,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )
        
        # ================================================================
        # Loop through combos - generate ONLY styled responses
        # ================================================================
        for place in places:
            for strength in strength_levels:
                
                # Apply style (with fixed lambda closure)
                if style_name in ("length_variation", "inter_vs_imper"):
                    # Use pre-loaded LLM-rewritten prompts + CoT suffix
                    all_styled = styled_prompts_by_strength[strength]
                    batch_styled_prompts_cot = [
                        sp + "\n\nLet's think step by step"
                        for sp in all_styled[start:end]
                    ]
                elif strength == 0:
                    batch_styled_prompts_cot = batch_orig_prompts_cot
                else:
                    if style_name == "politeness":
                        style_fn = lambda p, s=strength: style_fn_base(p, s)
                    else:
                        style_fn = lambda p, s=strength, pl=place: style_fn_base(p, s, place=pl)
                    
                    batch_styled_prompts_cot = [style_fn(p) for p in batch_orig_prompts_cot]
                
                # Generate styled responses
                styled_cache = _styled_cache_path(
                    data_dir=data_dir, dataset=dataset_name,
                    config_name=dataset_config_name, split=dataset_split,
                    seed=dataset_seed, sample_size=dataset_sample_size,
                    model_name=model_name, style=style_name,
                    place=place, strength=strength
                )
                
                batch_response_styled = _load_or_generate_styled(
                    cache_path=styled_cache,
                    overwrite_cache=overwrite_output_cache,
                    model=model,
                    tokenizer=tokenizer,
                    prompts_styled=batch_styled_prompts_cot,
                    batch_ids=batch_ids,
                    max_new_tokens=max_new_tokens,
                    batch_size=batch_size,
                )
                
                # Save rows
                for j in range(len(batch_orig_prompts)):
                    i = batch_ids[j]
                    
                    row = {
                        "model": model_name,
                        "style": style_name,
                        "problem_id": i,
                        "category": batch_categories[j],
                        "strength": strength,
                        "place": place,
                        "question_original": batch_orig_prompts[j],
                        "prompt_original_cot": batch_orig_prompts_cot[j],
                        "prompt_styled_cot": batch_styled_prompts_cot[j],
                        "response_original": batch_response_baseline[j],
                        "response_styled": batch_response_styled[j],
                    }
                    rows.append(row)
                
                pbar.update(1)
    
    pbar.close()
    
    # Save results
    df = pd.DataFrame(rows)
    out_csv = os.path.join(run_dir, f"{model_name}_responses.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✓ Saved responses: {out_csv}")
    
    return df


# =============================================================================
# Global runner
# =============================================================================

def run_experiment(
        models, dataset_name, sample_size, style_name, *,
        batch_size, max_new_tokens, strengths_explicit, strength_range, strength_step,
        data_dir, overwrite_sample_cache, overwrite_output_cache, places_override,
        rewrite_provider="openai", rewrite_model="gpt-4o-mini",
        rewrite_api_key_env="OPENAI_API_KEY", overwrite_style_cache=False):

    config = load_config()
    
    if dataset_name not in config["datasets"]:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    dataset_config = config["datasets"][dataset_name]
    
    if sample_size is None:
        sample_size = int(dataset_config["sample_size"])
    
    places = places_override if places_override else _get_places(config, style_name)
    
    as_float = (style_name == "length_variation")
    as_string = (style_name == "inter_vs_imper")
    strength_levels = _select_strengths(
        config_strengths=config["style_levels"][style_name],
        explicit_strengths=strengths_explicit,
        strength_range=strength_range,
        strength_step=strength_step,
        as_float=as_float,
        as_string=as_string,
    )
    
    if max_new_tokens is None:
        max_new_tokens = int(config["defaults"].get("max_new_tokens_cot", 200))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    style_dir = os.path.join(base_results_dir, "cot_responses")
    run_dir = os.path.join(style_dir, f"run_{dataset_name}_{style_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("COT RESPONSE GENERATION - PRODUCTION OPTIMIZED")
    print(f"{'='*80}")
    print(f"Models: {models}")
    print(f"Dataset: {dataset_name}")
    print(f"Style: {style_name}")
    print(f"Sample size: {sample_size}")
    print(f"Strengths: {strength_levels}")
    print(f"Places: {places}")
    print(f"Batch size: {batch_size}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Data cache: {data_dir}")
    print(f"Output: {run_dir}")
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
            rewrite_provider=rewrite_provider,
            rewrite_model=rewrite_model,
            rewrite_api_key_env=rewrite_api_key_env,
            overwrite_style_cache=overwrite_style_cache,
        )
        if df_model is not None and not df_model.empty:
            all_rows.append(df_model)
    
    if not all_rows:
        print("No outputs produced")
        return run_dir
    
    # Combine results
    df_all = pd.concat(all_rows, ignore_index=True)
    full_path = os.path.join(run_dir, "all_models_responses.csv")
    df_all.to_csv(full_path, index=False)
    print(f"✓ Saved combined responses: {full_path}")
    
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total responses: {len(df_all)}")
    print(f"Baseline cache: {data_dir}/cot_baseline_cache/")
    print(f"Styled cache: {data_dir}/cot_styled_cache/")
    print(f"Results: {full_path}")
    print(f"{'='*80}\n")
    
    return run_dir


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate CoT responses (production optimized)")
    parser.add_argument("--models", nargs="+", default=["L3.2-3B"])
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--style", type=str, required=True, choices=list(VALID_STYLES))
    
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for generation (default: 32)")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Maximum tokens to generate (default: from config, typically 200)")
    
    parser.add_argument("--places", nargs="+", default=None)
    parser.add_argument("--strengths", nargs="+", type=str, default=None,
                        help="Override strengths/multipliers/modes "
                             "(e.g. --strengths 0.5 1.5 2.0 or --strengths interrogative imperative)")
    parser.add_argument("--strength_range", nargs=2, type=int, default=None)
    parser.add_argument("--strength_step", type=int, default=1)
    
    # LLM for rewriting prompts (length_variation / structured styles)
    parser.add_argument("--rewrite_provider", type=str, default="openai",
                        choices=["openai", "gemini"],
                        help="Provider for prompt rewriting LLM (length_variation)")
    parser.add_argument("--rewrite_model", type=str, default="gpt-4o-mini",
                        help="Model name for prompt rewriting")
    parser.add_argument("--rewrite_api_key_env", type=str, default=None,
                        help="Env var for rewrite API key (default: auto from provider)")
    parser.add_argument("--overwrite_style_cache", action="store_true",
                        help="Re-generate LLM-rewritten prompts even if cached")

    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--overwrite_sample_cache", action="store_true")
    parser.add_argument("--overwrite_output_cache", action="store_true")
    
    args = parser.parse_args()
    
    config = load_config()
    models = _normalize_models(args.models, config)
    data_dir = args.data_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    
    rewrite_api_key_env = args.rewrite_api_key_env or (
        "OPENAI_API_KEY" if args.rewrite_provider == "openai" else "GEMINI_API_KEY"
    )

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
        rewrite_provider=args.rewrite_provider,
        rewrite_model=args.rewrite_model,
        rewrite_api_key_env=rewrite_api_key_env,
        overwrite_style_cache=bool(args.overwrite_style_cache),
    )


if __name__ == "__main__":
    main()