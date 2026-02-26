# experiments/cot_reasoning.py
"""
Chain-of-Thought Reasoning Trace Experiment (OPTIMIZED)
========================================================

Measures the number of reasoning steps produced by models when given
style-perturbed prompts with CoT instructions.

Key metric: Number of reasoning steps (step_1, step_2, ..., step_N)

Run:
  python experiments/cot_reasoning.py \
    --models L3.2-3B \
    --dataset gsm8k \
    --sample_size 128 \
    --style spacing \
    --strengths 0 50 100 \
    --places global \
    --batch_size 16
"""

import argparse
import os
import sys
import yaml
import json
import gzip
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_dataset_by_name
from utils.models import load_model, generate_response
from utils.metrics import add_cot_prompt, parse_cot_response
from utils.styles import apply_spacing, apply_punctuation, apply_letter_case, apply_politeness


VALID_STYLES = {"spacing", "punctuation", "letter_case", "politeness"}


def _safe_name(x: Optional[str]) -> str:
    """Sanitize strings for safe filesystem paths."""
    if x is None:
        return "none"
    x = str(x)
    return "".join([c if c.isalnum() or c in ("-", "_", ".", "=") else "_" for c in x])


def _cot_outputs_cache_path(
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
    """
    Path for cached CoT outputs.
    
    Structure: data/cot_outputs_cache/{dataset}/{model}/{style}/{place}/strength_{N}.jsonl.gz
    """
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
    """Read gzipped JSONL file."""
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonl_gz(path: str, rows: List[dict]) -> None:
    """Write gzipped JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_or_generate_cot_outputs(
        *,
        cache_path: str,
        overwrite_cache: bool,
        model,
        tokenizer,
        questions: List[str],
        prompts_original: List[str],
        prompts_styled: List[str],
        max_new_tokens: int,
        batch_size: int,
) -> Dict[str, List[str]]:
    """
    Load cached CoT outputs or generate new ones.
    
    Cache format:
    {
      "question_id": int,
      "question": str,
      "prompt_original": str,
      "prompt_styled": str,
      "response_original": str,
      "response_styled": str
    }
    
    Returns:
    {
      "responses_original": List[str],
      "responses_styled": List[str]
    }
    """
    n = len(questions)
    
    # Try to load from cache
    if os.path.exists(cache_path) and not overwrite_cache:
        try:
            rows = _read_jsonl_gz(cache_path)
            if len(rows) == n:
                print(f"  ✓ Loaded {n} cached outputs from {os.path.basename(cache_path)}")
                return {
                    "responses_original": [r["response_original"] for r in rows],
                    "responses_styled": [r["response_styled"] for r in rows],
                }
        except Exception as e:
            print(f"  ⚠ Cache read failed ({e}), regenerating...")
    
    # Generate new outputs
    print(f"  → Generating {n} original responses (batch_size={batch_size})...")
    responses_original = generate_response(
        model, tokenizer, prompts_original,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    
    print(f"  → Generating {n} styled responses (batch_size={batch_size})...")
    responses_styled = generate_response(
        model, tokenizer, prompts_styled,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    
    # Cache results
    rows = []
    for i in range(n):
        rows.append({
            "question_id": i,
            "question": questions[i],
            "prompt_original": prompts_original[i],
            "prompt_styled": prompts_styled[i],
            "response_original": responses_original[i],
            "response_styled": responses_styled[i],
        })
    
    _write_jsonl_gz(cache_path, rows)
    print(f"  ✓ Cached {n} outputs to {os.path.basename(cache_path)}")
    
    return {
        "responses_original": responses_original,
        "responses_styled": responses_styled,
    }


# =============================================================================
# Config + CLI Helpers
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
    """Select strength levels based on priority: explicit > range > config."""
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

    # Get from config
    config_strengths = config.get("style_levels", {}).get(style, [0, 20, 50, 100])
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
# Data Cache Helpers (Sample Caching)
# =============================================================================

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
    Run CoT experiment for one model.
    
    OPTIMIZATIONS:
    1. Output caching per (strength, place) combo
    2. Incremental CSV saving after each combo
    3. Fixed lambda closure bug with default arguments
    4. Progress tracking with tqdm
    """
    
    # Load model
    model, tokenizer = load_model(
        model_path,
        device_map=config["defaults"].get("device_map", "auto"),
        dtype=config["defaults"].get("dtype", "float32"),
    )
    
    style_fn_base = _get_style_function(style_name)
    questions = [_get_prompt_text(it) for it in items]
    
    rows: List[dict] = []
    
    total_runs = len(places) * len(strength_levels)
    pbar = tqdm(total=total_runs, desc=f"[{model_name}] CoT runs", unit="run")
    
    out_csv = os.path.join(run_dir, f"{model_name}_results.csv")
    
    for place in places:
        for strength in strength_levels:
            style_fn = None
            if strength != 0:
                if style_name == "politeness":
                    # Capture strength VALUE with default argument
                    style_fn = lambda q, s=strength: style_fn_base(q, s)
                else:
                    # Capture both strength and place VALUES
                    style_fn = lambda q, s=strength, p=place: style_fn_base(q, s, place=p)
            
            # Prepare prompts
            prompts_original = [add_cot_prompt(q) for q in questions]
            
            prompts_styled = []
            for q in questions:
                if style_fn is None:
                    prompts_styled.append(add_cot_prompt(q))
                else:
                    # Apply style to CoT-prompted question
                    prompt_with_cot = add_cot_prompt(q)
                    styled_prompt = style_fn(prompt_with_cot)
                    prompts_styled.append(styled_prompt)
        
            
            cache_path = _cot_outputs_cache_path(
                data_dir=data_dir,
                dataset=dataset_name,
                config_name=dataset_config_name,
                split=dataset_split,
                seed=dataset_seed,
                sample_size=dataset_sample_size,
                model_name=model_name,
                style=style_name,
                place=place,
                strength=strength,
            )
            
            outputs = _load_or_generate_cot_outputs(
                cache_path=cache_path,
                overwrite_cache=overwrite_output_cache,
                model=model,
                tokenizer=tokenizer,
                questions=questions,
                prompts_original=prompts_original,
                prompts_styled=prompts_styled,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )
            
            # Parse responses
            print(f"  → Parsing {len(questions)} responses...")
            for i in range(len(questions)):
                parsed_orig = parse_cot_response(outputs["responses_original"][i])
                parsed_styled = parse_cot_response(outputs["responses_styled"][i])
                
                steps_diff = parsed_styled["num_steps"] - parsed_orig["num_steps"]
                
                result = {
                    "model": model_name,
                    "style": style_name,
                    "problem_id": i,
                    "category": items[i].get("category", "unknown"),
                    "strength": strength,
                    "place": place,
                    
                    # Questions and prompts
                    "question_original": questions[i],
                    "prompt_original_cot": prompts_original[i],
                    "prompt_styled_cot": prompts_styled[i],
                    
                    # Responses
                    "response_original": outputs["responses_original"][i],
                    "response_styled": outputs["responses_styled"][i],
                    "response_original_cleaned": parsed_orig.get("response_cleaned", ""),
                    "response_styled_cleaned": parsed_styled.get("response_cleaned", ""),
                    
                    # Original metrics
                    "num_steps_original": parsed_orig["num_steps"],
                    "parse_success_original": parsed_orig["parse_success"],
                    "parse_error_original": parsed_orig.get("parse_error", ""),
                    "avg_step_length_original": parsed_orig["avg_step_length"],
                    "total_reasoning_length_original": parsed_orig["total_reasoning_length"],
                    "answer_original": parsed_orig.get("answer", ""),
                    "steps_original": " ||| ".join(parsed_orig["steps"]) if parsed_orig["steps"] else "",
                    
                    # Styled metrics
                    "num_steps_styled": parsed_styled["num_steps"],
                    "parse_success_styled": parsed_styled["parse_success"],
                    "parse_error_styled": parsed_styled.get("parse_error", ""),
                    "avg_step_length_styled": parsed_styled["avg_step_length"],
                    "total_reasoning_length_styled": parsed_styled["total_reasoning_length"],
                    "answer_styled": parsed_styled.get("answer", ""),
                    "steps_styled": " ||| ".join(parsed_styled["steps"]) if parsed_styled["steps"] else "",
                    
                    # Comparison
                    "steps_diff": steps_diff,
                    "steps_change_pct": (steps_diff / parsed_orig["num_steps"] * 100) if parsed_orig["num_steps"] > 0 else 0.0,
                    "both_parsed": parsed_orig["parse_success"] and parsed_styled["parse_success"],
                }
                
                # Add transparency logs
                if "cleaning_log" in parsed_orig:
                    result["cleaning_log_original"] = json.dumps(parsed_orig["cleaning_log"])
                if "cleaning_log" in parsed_styled:
                    result["cleaning_log_styled"] = json.dumps(parsed_styled["cleaning_log"])
                
                rows.append(result)
            
            
            df_partial = pd.DataFrame(rows)
            df_partial.to_csv(out_csv, index=False)
            
            pbar.update(1)
    
    pbar.close()
    
    print(f"✓ Saved final results: {out_csv}")
    
    # Print summary statistics
    df_final = pd.DataFrame(rows)
    total = len(df_final)
    both_parsed = df_final["both_parsed"].sum()
    
    orig_steps_avg = df_final["num_steps_original"].mean()
    styled_steps_avg = df_final["num_steps_styled"].mean()
    
    parse_success_orig = df_final["parse_success_original"].sum()
    parse_success_styled = df_final["parse_success_styled"].sum()
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*70}")
    print(f"Total questions:           {total}")
    print(f"Parse success (original):  {parse_success_orig}/{total} ({parse_success_orig/total*100:.1f}%)")
    print(f"Parse success (styled):    {parse_success_styled}/{total} ({parse_success_styled/total*100:.1f}%)")
    print(f"Both parsed:               {both_parsed}/{total} ({both_parsed/total*100:.1f}%)")
    print(f"Avg steps (original):      {orig_steps_avg:.2f}")
    print(f"Avg steps (styled):        {styled_steps_avg:.2f}")
    print(f"Difference:                {styled_steps_avg - orig_steps_avg:+.2f}")
    print(f"{'='*70}\n")
    
    return df_final


# =============================================================================
# Global Runner (Multi-Model)
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
        config=config,
        style=style_name,
        explicit_strengths=strengths_explicit,
        strength_range=strength_range,
        strength_step=strength_step,
    )
    
    if max_new_tokens is None:
        max_new_tokens = int(config["defaults"].get("max_new_tokens_cot", 512))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    style_dir = os.path.join(base_results_dir, "cot_reasoning")
    run_dir = os.path.join(style_dir, f"run_multi_{dataset_name}_{style_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("COT REASONING TRACE EXPERIMENT (OPTIMIZED)")
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
    print(f"Overwrite output cache: {overwrite_output_cache}")
    print(f"Output dir (results): {run_dir}")
    print(f"{'='*80}\n")
    
    items = _load_or_create_sample(
        data_dir=data_dir,
        dataset=dataset_name,
        config_name=dataset_config.get("config_name"),
        split=dataset_config.get("split", "test"),
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
            dataset_split=dataset_config.get("split", "test"),
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
    
    full_path = os.path.join(run_dir, "full_results_all_models.csv")
    df_all.to_csv(full_path, index=False)
    print(f"✓ Saved combined full results: {full_path}")
    
    # Generate plots automatically
    try:
        from utils.cot_plots import make_all_plots_from_csvs
        
        plot_dir = os.path.join(run_dir, "plots_metrics")
        os.makedirs(plot_dir, exist_ok=True)
        
        make_all_plots_from_csvs(
            plot_inputs=[run_dir],
            out_dir=plot_dir,
            strengths=strength_levels,
            places_filter=places,
            models_filter=models,
            dataset_name=dataset_name,
            style_name=style_name,
            save_pdf=False,
            include_title=False,
            legend_outside=True,
        )
        print(f"✓ Generated plots: {plot_dir}")
    except ImportError:
        print("⚠️ Plotting module not found - skipping plots")
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")
    
    return run_dir


# =============================================================================
# CLI Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run CoT reasoning trace experiment (optimized)")
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
    parser.add_argument("--overwrite_output_cache", action="store_true",
                        help="Regenerate cached outputs (useful for testing parser changes)")

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