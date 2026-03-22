# experiments/mirroring.py
"""
Style Mirroring Experiment (BATCHED, dataset-loaded prompts) — multi-model + CSV + line plots

What this script does:
- Loads prompts via utils.data.load_dataset_by_name
- Samples N prompts
- For EACH target model under test:
    - Generates baseline outputs for all prompts (batched)
    - For each (place, strength):
        - Builds styled prompts
        - Generates styled outputs (batched)
        - Cleans outputs (removes prompt echoes / chat transcripts)
        - Judges each example using OpenAI OR Gemini judge
    - Computes mirroring rate per (place, strength): YES / judged_total
    - Optionally stores ONE YES and ONE NO example per (place,strength)
- Saves:
    1) per-example CSV (optional; can be very large)
    2) per-bucket summary CSV (rates)
    3) plots: rate vs strength (one line per place; separate figure per model)

Strength selection options:
A) Explicit list:
    --strengths -10 -5 0 5 10
B) Choose K from config.yaml:
    --num_strengths 5 --strength_strategy extremes|even
C) Choose a numeric range (integers):
    --strength_range -10 10 --strength_step 1
   (This overrides config + num_strengths unless you also pass --clip_to_config.)

Run (OpenAI judge):
  export OPENAI_API_KEY="..."
  python experiments/mirroring.py \
    --models llama mistral \
    --dataset truthful_qa \
    --sample_size 128 \
    --style politeness \
    --places prefix suffix global \
    --strength_range 0 10 --strength_step 1 \
    --judge_provider openai \
    --judge_model gpt-4o-mini \
    --batch_size 16 \
    --print_examples_per_bucket \
    --save_per_example_csv

Run (Gemini judge):
  export GEMINI_API_KEY="..."
  python experiments/mirroring.py \
    --models llama \
    --dataset truthful_qa \
    --sample_size 128 \
    --style politeness \
    --places global \
    --strengths 1 3 8 \
    --judge_provider gemini \
    --judge_model gemini-2.5-flash \
    --batch_size 16

Notes:
- Per-example CSV can be huge: O(n_prompts * places * strengths).
  Use --save_per_example_csv only when needed.
"""

import argparse
import gzip
import json
import os
import sys
import yaml
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_dataset_by_name
from utils.models import load_model, generate_response
from utils.styles import apply_politeness
from plots.plots import apply_neurips_style

from utils.metrics import (
    clean_chatty_generation,
    judge_with_retries,
)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ExampleRecord:
    """Stores one (original, styled) example pair and its mirroring verdict."""
    prompt_orig: str
    output_orig: str
    prompt_styled: str
    output_styled: str
    judge_raw: str
    verdict: bool
    prompt_id: int


# =============================================================================
# Config + helpers
# =============================================================================

def load_config() -> Dict[str, Any]:
    """Load the project-level config.yaml from the repository root."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _normalize_models(models: List[str], config: Dict[str, Any]) -> List[str]:
    """Validate and expand model keys against config; 'all' returns every available model."""
    available = list(config.get("models", {}).keys())
    if not available:
        raise ValueError("No models found in config.yaml under 'models'.")

    m = [x.strip() for x in models]
    if len(m) == 1 and m[0] == "all":
        return available

    unknown = [x for x in m if x not in available]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Available: {available} or 'all'.")
    return m


def apply_style(prompt: str, style: str, strength: int, place: str) -> str:
    """Apply the named style to a prompt. Currently only 'politeness' is supported."""
    if style == "politeness":
        return apply_politeness(prompt, strength, place=place)
    raise ValueError(f"Unknown style: {style}")


def _get_prompt_text(item: Dict[str, Any]) -> str:
    """Extract the prompt string from a dataset item dict, trying 'question' then 'prompt'."""
    if "question" in item and item["question"]:
        return str(item["question"])
    if "prompt" in item and item["prompt"]:
        return str(item["prompt"])
    return str(item)


def select_strengths(
        *,
        config_strengths: List[int],
        explicit_strengths: Optional[List[int]],
        num_strengths: Optional[int],
        strategy: str,
        strength_range: Optional[Tuple[int, int]],
        strength_step: int,
        clip_to_config: bool,
) -> List[int]:
    """
    Priority:
    1) explicit_strengths (if provided)
    2) strength_range (if provided) => integer grid [lo..hi] step
       - if clip_to_config: intersect with config_strengths
    3) num_strengths from config_strengths using strategy
    4) all config_strengths
    """
    if explicit_strengths:
        seen = set()
        out = []
        for s in explicit_strengths:
            if s not in seen:
                out.append(int(s))
                seen.add(s)
        return out

    if strength_range is not None:
        lo, hi = strength_range
        if strength_step <= 0:
            raise ValueError("--strength_step must be >= 1")
        if lo > hi:
            lo, hi = hi, lo
        grid = list(range(int(lo), int(hi) + 1, int(strength_step)))
        if clip_to_config:
            cfg = set(int(x) for x in config_strengths)
            grid = [s for s in grid if s in cfg]
        # keep unique
        seen = set()
        out = []
        for s in grid:
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out

    strengths = sorted(set(int(x) for x in config_strengths))
    if not num_strengths or num_strengths >= len(strengths):
        return strengths

    if strategy == "even":
        k = num_strengths
        if k == 1:
            return [strengths[len(strengths) // 2]]
        idxs = [round(i * (len(strengths) - 1) / (k - 1)) for i in range(k)]
        chosen, seen = [], set()
        for idx in idxs:
            s = strengths[idx]
            if s not in seen:
                chosen.append(s)
                seen.add(s)
        # Fill if duplicates due to rounding
        i = 0
        while len(chosen) < k and i < len(strengths):
            for cand in (strengths[i], strengths[-1 - i]):
                if len(chosen) >= k:
                    break
                if cand not in seen:
                    chosen.append(cand)
                    seen.add(cand)
            i += 1
        return chosen

    # extremes-first
    strengths_sorted = sorted(strengths, key=lambda x: (-abs(x), x))
    return strengths_sorted[:num_strengths]


# =============================================================================
# Output caching helpers
# =============================================================================
# Uses the SAME cache path layout and row format as the base experiments
# (punctuation.py, length_variation.py, etc.) so that outputs generated by
# either experiment are found and reused by the other.
#
# Path:  data/outputs_cache/{dataset}/.../  {model}/{style}/{place}/strength_{s}.jsonl.gz
# Row:   {prompt_id, prompt_orig, output_orig_raw, output_orig_clean,
#          prompt_styled, output_styled_raw, output_styled_clean}
# =============================================================================

def _safe_name(x) -> str:
    if x is None:
        return "none"
    x = str(x)
    return "".join(c if c.isalnum() or c in ("-", "_", ".", "=") else "_" for c in x)


def _outputs_cache_path(
        *, data_dir: str, dataset: str, config_name: Optional[str],
        split: str, seed: int, sample_size: int,
        model_name: str, style: str, place: str, strength,
) -> str:
    """
    Canonical cache file for one (model, style, place, strength) bucket.

    Matches the path used by punctuation.py / length_variation.py so both
    the base experiments and the mirroring experiment share the same cache.
    """
    return os.path.join(
        data_dir, "outputs_cache",
        _safe_name(dataset),
        f"config_{_safe_name(config_name)}",
        f"split_{_safe_name(split)}",
        f"seed_{seed}", f"n_{sample_size}",
        _safe_name(model_name),
        _safe_name(style),
        _safe_name(place),
        f"strength_{_safe_name(str(strength))}.jsonl.gz",
    )


def _read_jsonl_gz(path: str) -> List[dict]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
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
        overwrite: bool,
        model,
        tokenizer,
        prompt_orig_list: List[str],
        prompt_styled_list: List[str],
        max_new_tokens: int,
        batch_size: int,
) -> Dict[str, List[str]]:
    """
    Return dict with output_orig_raw, output_orig_clean,
    output_styled_raw, output_styled_clean.

    Same signature and row format as the base experiments so the
    cache files are interchangeable.
    """
    n = len(prompt_orig_list)

    if os.path.exists(cache_path) and not overwrite:
        rows = _read_jsonl_gz(cache_path)
        if len(rows) == n:
            return {
                "output_orig_raw": [r["output_orig_raw"] for r in rows],
                "output_orig_clean": [r["output_orig_clean"] for r in rows],
                "output_styled_raw": [r["output_styled_raw"] for r in rows],
                "output_styled_clean": [r["output_styled_clean"] for r in rows],
            }

    out_orig_raw = generate_response(
        model, tokenizer, prompts=prompt_orig_list,
        max_new_tokens=max_new_tokens, batch_size=batch_size,
    )
    out_styled_raw = generate_response(
        model, tokenizer, prompts=prompt_styled_list,
        max_new_tokens=max_new_tokens, batch_size=batch_size,
    )
    if len(out_orig_raw) != n or len(out_styled_raw) != n:
        raise RuntimeError("generate_response returned wrong number of outputs.")

    out_orig_clean = [clean_chatty_generation(o, prompt_text=p) for p, o in zip(prompt_orig_list, out_orig_raw)]
    out_styled_clean = [clean_chatty_generation(o, prompt_text=sp) for sp, o in zip(prompt_styled_list, out_styled_raw)]

    rows = []
    for i in range(n):
        rows.append({
            "prompt_id": i,
            "prompt_orig": prompt_orig_list[i],
            "output_orig_raw": out_orig_raw[i],
            "output_orig_clean": out_orig_clean[i],
            "prompt_styled": prompt_styled_list[i],
            "output_styled_raw": out_styled_raw[i],
            "output_styled_clean": out_styled_clean[i],
        })
    _write_jsonl_gz(cache_path, rows)
    print(f"  ✓ Cached outputs ({n} rows) → {cache_path}")

    return {
        "output_orig_raw": out_orig_raw,
        "output_orig_clean": out_orig_clean,
        "output_styled_raw": out_styled_raw,
        "output_styled_clean": out_styled_clean,
    }


def _all_outputs_cached(
        *, places: List[str], strengths: List[int],
        data_dir: str, dataset: str, config_name: Optional[str],
        split: str, seed: int, sample_size: int,
        model_name: str, style: str,
) -> bool:
    """Check whether every (place, strength) bucket is already cached."""
    for place in places:
        for strength in strengths:
            sp = _outputs_cache_path(
                data_dir=data_dir, dataset=dataset, config_name=config_name,
                split=split, seed=seed, sample_size=sample_size,
                model_name=model_name, style=style, place=place, strength=strength,
            )
            if not os.path.exists(sp):
                return False
    return True


# =============================================================================
# Plotting
# =============================================================================

def plot_rate_lines(df_summary: pd.DataFrame, *, model_name: str, out_path: str, title: str):
    """
    df_summary columns required:
      - strength (int)
      - place (str)
      - rate (float)
    One line per place.
    """
    if df_summary.empty:
        return

    strengths_sorted = sorted(df_summary["strength"].unique().tolist())

    apply_neurips_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    for place in sorted(df_summary["place"].unique().tolist()):
        sub = df_summary[df_summary["place"] == place].copy()
        if sub.empty:
            continue
        sub = sub.set_index("strength").reindex(strengths_sorted)
        y = sub["rate"].values
        ax.plot(strengths_sorted, y, marker="o", label=f"{place}")

    ax.set_xlabel("Strength")
    ax.set_ylabel("Mirroring rate (YES / total)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main per model
# =============================================================================

def run_for_one_model(
        *,
        model_name: str,
        model_path: str,
        prompts: List[str],
        places: List[str],
        strengths: List[int],
        style: str,
        config: Dict[str, Any],
        batch_size: int,
        max_new_tokens: int,
        judge_provider: str,
        judge_model: str,
        openai_key_env: str,
        gemini_key_env: str,
        judge_max_output_tokens: int,
        max_judge_calls: int,
        print_every: int,
        print_examples_per_bucket: bool,
        save_per_example_csv: bool,
        disable_guard: bool,
        # --- caching parameters ---
        data_dir: Optional[str] = None,
        dataset: str = "",
        dataset_config_name: Optional[str] = None,
        dataset_split: str = "validation",
        dataset_seed: int = 42,
        dataset_sample_size: int = 128,
        overwrite_output_cache: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[Tuple[str, int], Optional[ExampleRecord]], Dict[Tuple[str, int], Optional[ExampleRecord]], int]:
    """
    Returns:
      - df_summary (per place,strength)
      - df_examples (optional per-example; None if not saved)
      - example_yes dict
      - example_no dict
      - judge_calls_used
    """
    n = len(prompts)
    use_cache = bool(data_dir)

    # --- Decide whether the model must be loaded ---
    need_model = True
    if use_cache:
        need_model = not _all_outputs_cached(
            places=places, strengths=strengths,
            data_dir=data_dir, dataset=dataset,
            config_name=dataset_config_name, split=dataset_split,
            seed=dataset_seed, sample_size=dataset_sample_size,
            model_name=model_name, style=style,
        )
        if not need_model:
            print(f"  ✓ All outputs for {model_name} are cached — skipping model load")

    model, tokenizer = None, None
    if need_model:
        model, tokenizer = load_model(
            model_path,
            device_map=config["defaults"].get("device_map", "auto"),
            dtype=config["defaults"].get("dtype", "float32"),
        )

    counts: Dict[Tuple[str, int], Dict[str, int]] = defaultdict(lambda: {"yes": 0, "total": 0})

    example_yes: Dict[Tuple[str, int], Optional[ExampleRecord]] = {(p, s): None for p in places for s in strengths}
    example_no: Dict[Tuple[str, int], Optional[ExampleRecord]] = {(p, s): None for p in places for s in strengths}

    per_example_rows: List[dict] = []

    planned = len(places) * len(strengths) * n
    pbar = tqdm(total=min(planned, max_judge_calls), desc=f"[{model_name}] judging", unit="ex")

    judge_calls = 0
    printed = 0

    for place in places:
        for strength in strengths:
            key = (place, strength)

            styled_prompts = [apply_style(p, style, strength, place) for p in prompts]

            # --- Load or generate outputs (baseline + styled together, with caching) ---
            if use_cache:
                cache_path = _outputs_cache_path(
                    data_dir=data_dir, dataset=dataset,
                    config_name=dataset_config_name, split=dataset_split,
                    seed=dataset_seed, sample_size=dataset_sample_size,
                    model_name=model_name, style=style,
                    place=place, strength=strength,
                )
                out = _load_or_generate_outputs_for_bucket(
                    cache_path=cache_path, overwrite=overwrite_output_cache,
                    model=model, tokenizer=tokenizer,
                    prompt_orig_list=prompts,
                    prompt_styled_list=styled_prompts,
                    max_new_tokens=max_new_tokens, batch_size=batch_size,
                )
                base_clean = out["output_orig_clean"]
                styled_clean = out["output_styled_clean"]
            else:
                base_raw = generate_response(
                    model, tokenizer, prompts=prompts,
                    max_new_tokens=max_new_tokens, batch_size=batch_size,
                )
                base_clean = [clean_chatty_generation(o, prompt_text=p) for p, o in zip(prompts, base_raw)]
                styled_raw = generate_response(
                    model, tokenizer, prompts=styled_prompts,
                    max_new_tokens=max_new_tokens, batch_size=batch_size,
                )
                styled_clean = [clean_chatty_generation(o, prompt_text=sp) for sp, o in zip(styled_prompts, styled_raw)]

            for i in range(n):
                if judge_calls >= max_judge_calls:
                    break

                verdict, judge_raw, _ = judge_with_retries(
                    judge_provider=judge_provider,
                    original_prompt=prompts[i],
                    original_output=base_clean[i],
                    styled_prompt=styled_prompts[i],
                    styled_output=styled_clean[i],
                    style_name=style,
                    strength=strength,
                    place=place,
                    judge_model=judge_model,
                    openai_key_env=openai_key_env,
                    gemini_key_env=gemini_key_env,
                    max_output_tokens=judge_max_output_tokens,

                )

                judge_calls += 1
                pbar.update(1)

                if verdict is None:
                    # optionally keep row for debugging
                    if save_per_example_csv:
                        per_example_rows.append({
                            "model": model_name,
                            "prompt_id": i,
                            "place": place,
                            "strength": int(strength),
                            "prompt_orig": prompts[i],
                            "output_orig": base_clean[i],
                            "prompt_styled": styled_prompts[i],
                            "output_styled": styled_clean[i],
                            "judge_raw": judge_raw,
                            "verdict": "",
                        })
                    continue

                counts[key]["total"] += 1
                if verdict:
                    counts[key]["yes"] += 1

                if save_per_example_csv:
                    per_example_rows.append({
                        "model": model_name,
                        "prompt_id": i,
                        "place": place,
                        "strength": int(strength),
                        "prompt_orig": prompts[i],
                        "output_orig": base_clean[i],
                        "prompt_styled": styled_prompts[i],
                        "output_styled": styled_clean[i],
                        "judge_raw": judge_raw,
                        "verdict": "YES" if verdict else "NO",
                    })

                if print_examples_per_bucket:
                    if verdict and example_yes[key] is None:
                        example_yes[key] = ExampleRecord(
                            prompt_orig=prompts[i],
                            output_orig=base_clean[i],
                            prompt_styled=styled_prompts[i],
                            output_styled=styled_clean[i],
                            judge_raw=judge_raw,
                            verdict=True,
                            prompt_id=i,
                        )
                    if (not verdict) and example_no[key] is None:
                        example_no[key] = ExampleRecord(
                            prompt_orig=prompts[i],
                            output_orig=base_clean[i],
                            prompt_styled=styled_prompts[i],
                            output_styled=styled_clean[i],
                            judge_raw=judge_raw,
                            verdict=False,
                            prompt_id=i,
                        )

                if print_every and (judge_calls % print_every == 0):
                    printed += 1
                    print("\n" + "-" * 110)
                    print(f"[DEBUG #{printed}] model={model_name} place={place} strength={strength} i={i}")
                    print("PROMPT:", prompts[i])
                    print("BASE (clean):", base_clean[i])
                    print("STYLED PROMPT:", styled_prompts[i])
                    print("STYLED (clean):", styled_clean[i])
                    print("JUDGE RAW:", judge_raw)
                    print("VERDICT:", "YES" if verdict else "NO")
                    print("-" * 110 + "\n")

            if judge_calls >= max_judge_calls:
                break
        if judge_calls >= max_judge_calls:
            break

    pbar.close()

    # Build summary DF
    summary_rows = []
    for place in places:
        for strength in strengths:
            key = (place, strength)
            yes = counts[key]["yes"]
            total = counts[key]["total"]
            rate = (yes / total) if total > 0 else np.nan
            summary_rows.append({
                "model": model_name,
                "place": place,
                "strength": int(strength),
                "yes": int(yes),
                "total": int(total),
                "rate": float(rate) if np.isfinite(rate) else np.nan,
            })
    df_summary = pd.DataFrame(summary_rows)

    df_examples = pd.DataFrame(per_example_rows) if (save_per_example_csv and per_example_rows) else None
    return df_summary, df_examples, example_yes, example_no, judge_calls


# =============================================================================
# Examples printing (optional)
# =============================================================================

def _print_example_bucket(model_name: str, place: str, strength: int, yes_ex: Optional[ExampleRecord], no_ex: Optional[ExampleRecord]):
    print("\n" + "=" * 110)
    print(f"EXAMPLES for bucket: model={model_name} | place={place} | strength={strength}")
    print("=" * 110)

    def _print_one(title: str, ex: ExampleRecord):
        print("\n" + "-" * 110)
        print(title)
        print("-" * 110)
        print(f"prompt_id: {ex.prompt_id}")

        print("\nORIGINAL PROMPT:")
        print(ex.prompt_orig)
        print("\nORIGINAL OUTPUT (clean):")
        print(ex.output_orig)

        print("\nSTYLED PROMPT:")
        print(ex.prompt_styled)
        print("\nSTYLED OUTPUT (clean):")
        print(ex.output_styled)

        print("\nJUDGE RAW OUTPUT:")
        print(ex.judge_raw)
        print("\nVERDICT:", "YES" if ex.verdict else "NO")
        print("-" * 110)

    if yes_ex is None:
        print("\n[YES] Not found for this bucket (within judge limits / failures).")
    else:
        _print_one("[YES] Mirroring present", yes_ex)

    if no_ex is None:
        print("\n[NO] Not found for this bucket (within judge limits / failures).")
    else:
        _print_one("[NO] Mirroring absent", no_ex)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    # Target models under test (multi-model)
    parser.add_argument("--models", nargs="+", default=["llama"], help="Model keys from config.yaml or 'all'")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=None)

    # Dataset
    parser.add_argument("--dataset", type=str, default="truthful_qa")
    parser.add_argument("--sample_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    # Style
    parser.add_argument("--style", type=str, default="politeness")
    parser.add_argument("--places", nargs="+", default=["prefix", "suffix", "global"])

    # Strength selection
    parser.add_argument("--strengths", nargs="+", type=int, default=None)
    parser.add_argument("--num_strengths", type=int, default=None)
    parser.add_argument("--strength_strategy", type=str, default="extremes", choices=["extremes", "even"])

    parser.add_argument("--strength_range", nargs=2, type=int, default=None, metavar=("LO", "HI"),
                        help="Use integer strengths from LO..HI (inclusive). Overrides config selection.")
    parser.add_argument("--strength_step", type=int, default=1, help="Step for --strength_range (default=1).")
    parser.add_argument("--clip_to_config", action="store_true",
                        help="If using --strength_range, intersect with config strengths.")

    # Judge
    parser.add_argument("--judge_provider", type=str, default="openai", choices=["openai", "gemini"])
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openai_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--gemini_key_env", type=str, default="GEMINI_API_KEY")
    parser.add_argument("--judge_max_output_tokens", type=int, default=16)
    parser.add_argument("--max_judge_calls", type=int, default=200000)
    parser.add_argument("--disable_guard", action="store_true", help="Disable light false-positive guard.")

    # Output / logging
    parser.add_argument("--results_dir", type=str, default=None, help="Override base results dir (default=../results)")
    parser.add_argument("--run_name", type=str, default=None, help="Optional custom run name (folder).")
    parser.add_argument("--save_per_example_csv", action="store_true",
                        help="Save per-example rows (can be huge).")
    parser.add_argument("--print_every", type=int, default=0)
    parser.add_argument("--print_examples_per_bucket", action="store_true")

    # Caching
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Base data dir for output caches (default: ../data)")
    parser.add_argument("--overwrite_output_cache", action="store_true",
                        help="Re-generate model outputs even if cached")

    args = parser.parse_args()

    # Key check
    if args.judge_provider == "openai":
        if not os.environ.get(args.openai_key_env):
            raise SystemExit(f"ERROR: {args.openai_key_env} not set. export {args.openai_key_env}='...'")
    else:
        if not os.environ.get(args.gemini_key_env):
            raise SystemExit(f"ERROR: {args.gemini_key_env} not set. export {args.gemini_key_env}='...'")

    config = load_config()

    # normalize models
    models = _normalize_models(args.models, config)

    # validate dataset + style
    if args.dataset not in config["datasets"]:
        raise ValueError(f"Dataset '{args.dataset}' not in config. Available: {list(config['datasets'].keys())}")
    if args.style not in config["style_levels"]:
        raise ValueError(f"Style '{args.style}' not in config['style_levels'].")

    # strengths
    config_strengths = list(config["style_levels"][args.style])
    strengths = select_strengths(
        config_strengths=config_strengths,
        explicit_strengths=args.strengths,
        num_strengths=args.num_strengths,
        strategy=args.strength_strategy,
        strength_range=tuple(args.strength_range) if args.strength_range else None,
        strength_step=int(args.strength_step),
        clip_to_config=bool(args.clip_to_config),
    )
    if not strengths:
        raise SystemExit("No strengths selected (check your args / --clip_to_config).")

    max_new_tokens = int(args.max_new_tokens or config["defaults"].get("max_new_tokens", 128))

    # Data cache directory
    data_dir = args.data_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    # Load dataset prompts once
    dataset_cfg = config["datasets"][args.dataset]
    items = load_dataset_by_name(
        args.dataset,
        sample_size=args.sample_size,
        seed=args.seed,
        config_name=dataset_cfg.get("config_name"),
        split=dataset_cfg.get("split", "validation"),
    )
    prompts = [_get_prompt_text(it) for it in items]
    if not prompts:
        raise SystemExit("No prompts loaded from dataset.")
    n = len(prompts)

    # Results directory
    base_results_dir = args.results_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    style_dir = os.path.join(base_results_dir, "mirroring", args.style)
    os.makedirs(style_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"run_{args.dataset}_{timestamp}"
    run_dir = os.path.join(style_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print("\n" + "=" * 110)
    print("MIRRORING EXPERIMENT (MULTI-MODEL, DATASET-LOADED)")
    print("=" * 110)
    print(f"Run dir: {run_dir}")
    print(f"Dataset: {args.dataset} | n={n} | seed={args.seed}")
    print(f"Models: {models}")
    print(f"Batch size: {args.batch_size} | max_new_tokens: {max_new_tokens}")
    print(f"Style: {args.style} | places={args.places} | strengths={strengths}")
    print(f"Judge: {args.judge_provider} / {args.judge_model} | judge_max_tokens={args.judge_max_output_tokens}")
    print(f"Guard enabled: {not args.disable_guard}")
    print(f"Save per-example CSV: {bool(args.save_per_example_csv)}")
    print(f"Data cache dir: {data_dir}")
    print(f"Overwrite output cache: {args.overwrite_output_cache}")
    print("=" * 110 + "\n")

    # Run per model
    all_summary = []
    total_judge_calls = 0

    for model_name in models:
        if model_name not in config["models"]:
            raise ValueError(f"Model '{model_name}' not in config. Available: {list(config['models'].keys())}")
        model_path = config["models"][model_name]

        df_summary, df_examples, ex_yes, ex_no, used = run_for_one_model(
            model_name=model_name,
            model_path=model_path,
            prompts=prompts,
            places=args.places,
            strengths=strengths,
            style=args.style,
            config=config,
            batch_size=int(args.batch_size),
            max_new_tokens=int(max_new_tokens),
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            openai_key_env=args.openai_key_env,
            gemini_key_env=args.gemini_key_env,
            judge_max_output_tokens=int(args.judge_max_output_tokens),
            max_judge_calls=int(args.max_judge_calls),
            print_every=int(args.print_every),
            print_examples_per_bucket=bool(args.print_examples_per_bucket),
            save_per_example_csv=bool(args.save_per_example_csv),
            disable_guard=bool(args.disable_guard),
            # --- caching ---
            data_dir=data_dir,
            dataset=args.dataset,
            dataset_config_name=dataset_cfg.get("config_name"),
            dataset_split=dataset_cfg.get("split", "validation"),
            dataset_seed=args.seed,
            dataset_sample_size=n,
            overwrite_output_cache=bool(args.overwrite_output_cache),
        )

        total_judge_calls += used
        all_summary.append(df_summary)

        # Save per-model summary
        summary_path = os.path.join(run_dir, f"{model_name}_mirroring_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"✓ Saved summary CSV: {summary_path}")

        # Save per-example if requested
        if args.save_per_example_csv and df_examples is not None and not df_examples.empty:
            ex_path = os.path.join(run_dir, f"{model_name}_mirroring_per_example.csv")
            df_examples.to_csv(ex_path, index=False)
            print(f"✓ Saved per-example CSV: {ex_path}")

        # Plot per model
        plot_path = os.path.join(plot_dir, f"{model_name}_rate_vs_strength.png")
        plot_rate_lines(
            df_summary=df_summary[["place", "strength", "rate"]],
            model_name=model_name,
            out_path=plot_path,
            title=f"Mirroring rate vs Strength | model={model_name} | dataset={args.dataset}",
        )
        print(f"✓ Saved plot: {plot_path}")

        # Print examples if requested
        if args.print_examples_per_bucket:
            print("\n" + "=" * 110)
            print(f"ONE YES + ONE NO EXAMPLE PER (PLACE × STRENGTH) — model={model_name}")
            print("=" * 110)
            for place in args.places:
                for strength in strengths:
                    key = (place, strength)
                    _print_example_bucket(model_name, place, strength, ex_yes.get(key), ex_no.get(key))

    # Combined summary
    df_all = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    combined_path = os.path.join(run_dir, "all_models_mirroring_summary.csv")
    df_all.to_csv(combined_path, index=False)
    print(f"\n✓ Saved combined summary CSV: {combined_path}")

    print("\n" + "=" * 110)
    print(f"Total judge calls attempted (all models): {total_judge_calls}")
    print("Done.\n")


if __name__ == "__main__":
    main()
