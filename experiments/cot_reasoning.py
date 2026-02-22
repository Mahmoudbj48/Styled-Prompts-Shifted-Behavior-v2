# experiments/cot_reasoning.py
"""
Chain-of-Thought Reasoning Trace Experiment (BATCHED) + DATA CACHING

Measures the number of reasoning steps produced by models when given
style-perturbed prompts with CoT instructions.

Key metric: Number of reasoning steps (step_1, step_2, ..., step_N)

Run:
  python experiments/cot_reasoning.py \
    --models L3.2-1B \
    --dataset gsm8k \
    --sample_size 128 \
    --style spacing \
    --strengths 0 20 50 100 \
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
from utils.models import load_model
from utils.metrics import evaluate_cot_reasoning_comparison
from utils.styles import apply_spacing, apply_punctuation, apply_letter_case, apply_politeness


VALID_STYLES = {"spacing", "punctuation", "letter_case", "politeness"}


# --------------------------
# Config + CLI helpers
# --------------------------
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


# --------------------------
# Data cache helpers
# --------------------------
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


# --------------------------
# Main experiment per model
# --------------------------
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
) -> pd.DataFrame:
    
    model, tokenizer = load_model(
        model_path,
        device_map=config["defaults"].get("device_map", "auto"),
        dtype=config["defaults"].get("dtype", "float32"),
    )
    
    style_fn_base = _get_style_function(style_name)
    
    questions = [_get_prompt_text(it) for it in items]
    
    rows: List[dict] = []
    
    total_runs = len(places) * len(strength_levels)
    pbar = tqdm(total=total_runs, desc=f"[{model_name}] CoT evaluation", unit="run")
    
    for place in places:
        for strength in strength_levels:
            # Create style function for this strength/place
            if strength == 0:
                style_fn = None
            else:
                if style_name == "politeness":
                    style_fn = lambda q: style_fn_base(q, strength)
                else:
                    style_fn = lambda q: style_fn_base(q, strength, place=place)
            
            # Evaluate CoT reasoning
            results = evaluate_cot_reasoning_comparison(
                model=model,
                tokenizer=tokenizer,
                questions=questions,
                style_fn=style_fn,
                strength=strength,
                place=place,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                apply_cot_before_style=True,
            )
            
            # Add model name and style to each result
            for i, result in enumerate(results):
                result["model"] = model_name
                result["style"] = style_name
                result["problem_id"] = i
                result["category"] = items[i].get("category", "unknown")
                rows.append(result)
            
            pbar.update(1)
    
    pbar.close()
    
    df = pd.DataFrame(rows)
    
    out_csv = os.path.join(run_dir, f"{model_name}_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"✓ Saved per-example results: {out_csv}")
    
    return df


# --------------------------
# Global runner (multi-model)
# --------------------------
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
    print("COT REASONING TRACE EXPERIMENT (MULTI-MODEL, BATCHED)")
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


def main():
    parser = argparse.ArgumentParser(description="Run CoT reasoning trace experiment (multi-model, batched)")
    parser.add_argument("--models", nargs="+", default=["L3.2-1B"], help="Model keys from config or 'all'")
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
        places_override=args.places,
    )


if __name__ == "__main__":
    main()