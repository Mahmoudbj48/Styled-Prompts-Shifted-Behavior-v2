# experiments/mirroring.py
"""
Style Mirroring Experiment (BATCHED, dataset-loaded prompts) + Examples per (place,strength)

What this script does:
- Loads prompts using your database loader: utils.data.load_dataset_by_name
- Samples N prompts (sample_size)
- Generates baseline outputs for all prompts (batched)
- For each (place, strength):
    - Builds styled prompts
    - Generates styled outputs (batched)
    - Cleans outputs to remove prompt echoes / chat transcripts
    - Judges each example using OpenAI OR Gemini judge
- Reports mirroring rate per (place, strength):
    YES / judged_total
- OPTIONAL: prints ONE YES and ONE NO example per (place,strength):
    including:
      - original prompt + baseline output
      - styled prompt + styled output
      - judge raw output

Run:
  export OPENAI_API_KEY="..."
  python experiments/mirroring.py \
    --model llama \
    --dataset truthful_qa \
    --sample_size 128 \
    --style politeness \
    --places prefix suffix global \
    --num_strengths 5 \
    --judge_provider openai \
    --judge_model gpt-4o-mini \
    --batch_size 16 \
    --max_judge_calls 20000 \
    --print_examples_per_bucket

Or Gemini:
  export GEMINI_API_KEY="..."
  python experiments/mirroring.py \
    --model llama \
    --dataset truthful_qa \
    --sample_size 128 \
    --style politeness \
    --places global \
    --strengths -10 -5 0 5 10 \
    --judge_provider gemini \
    --judge_model gemini-2.5-flash \
    --batch_size 16 \
    --print_examples_per_bucket
"""

import argparse
import os
import sys
import yaml
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_dataset_by_name
from utils.models import load_model, generate_response
from utils.styles import apply_politeness


from utils.metrics import (
    clean_chatty_generation,
    judge_with_retries,
)

# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ExampleRecord:
    prompt_orig: str
    output_orig: str
    prompt_styled: str
    output_styled: str
    judge_raw: str
    verdict: bool
    prompt_id: int


# =============================================================================
# Config helpers
# =============================================================================

def load_config() -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def select_strengths(
        *,
        config_strengths: List[int],
        explicit_strengths: Optional[List[int]],
        num_strengths: Optional[int],
        strategy: str = "extremes",
) -> List[int]:
    """
    - If explicit_strengths provided: use them (unique, preserve order).
    - Else if num_strengths provided: select K from config_strengths using strategy.
    - Else: use config_strengths as-is.
    """
    if explicit_strengths:
        seen = set()
        out = []
        for s in explicit_strengths:
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out

    strengths = sorted(set(config_strengths))
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

    # Default: extremes-first by |s|
    strengths_sorted = sorted(strengths, key=lambda x: (-abs(x), x))
    return strengths_sorted[:num_strengths]


def apply_style(prompt: str, style: str, strength: int, place: str) -> str:
    if style == "politeness":
        return apply_politeness(prompt, strength, place=place)
    raise ValueError(f"Unknown style: {style}")


def _get_prompt_text(item: Dict[str, Any]) -> str:
    """
    Unify access across datasets: prefer 'question', else 'prompt'.
    Extend if your loader uses different keys.
    """
    if "question" in item and item["question"]:
        return str(item["question"])
    if "prompt" in item and item["prompt"]:
        return str(item["prompt"])
    return str(item)


# =============================================================================
# Printing helpers
# =============================================================================

def _print_example_bucket(place: str, strength: int, yes_ex: Optional[ExampleRecord], no_ex: Optional[ExampleRecord]):
    print("\n" + "=" * 110)
    print(f"EXAMPLES for bucket: place={place} | strength={strength}")
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

    # Target model under test
    parser.add_argument("--model", type=str, default="llama", help="Model key from config.yaml")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for target LLM generation")
    parser.add_argument("--max_new_tokens", type=int, default=None)

    # Dataset (database-loaded prompts)
    parser.add_argument("--dataset", type=str, default="truthful_qa", help="Dataset name from config.yaml")
    parser.add_argument("--sample_size", type=int, default=128, help="Number of prompts to sample")
    parser.add_argument("--seed", type=int, default=42)

    # Style params
    parser.add_argument("--style", type=str, default="politeness")
    parser.add_argument("--places", nargs="+", default=["prefix", "suffix", "global"])

    parser.add_argument("--strengths", nargs="+", type=int, default=None)
    parser.add_argument("--num_strengths", type=int, default=None)
    parser.add_argument("--strength_strategy", type=str, default="extremes", choices=["extremes", "even"])

    # Judge params
    parser.add_argument("--judge_provider", type=str, default="openai", choices=["openai", "gemini"])
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openai_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--gemini_key_env", type=str, default="GEMINI_API_KEY")
    parser.add_argument("--judge_max_output_tokens", type=int, default=16)
    parser.add_argument("--max_judge_calls", type=int, default=200000)

    # Verbosity / Examples
    parser.add_argument("--print_every", type=int, default=0,
                        help="If >0, print debug details every N judged examples.")
    parser.add_argument("--print_examples_per_bucket", action="store_true",
                        help="Print ONE YES and ONE NO example per (place,strength) bucket (if found).")

    args = parser.parse_args()

    # Key check
    if args.judge_provider == "openai":
        if not os.environ.get(args.openai_key_env):
            raise SystemExit(f"ERROR: {args.openai_key_env} not set. export {args.openai_key_env}='...'")
    else:
        if not os.environ.get(args.gemini_key_env):
            raise SystemExit(f"ERROR: {args.gemini_key_env} not set. export {args.gemini_key_env}='...'")

    config = load_config()

    if args.model not in config["models"]:
        raise ValueError(f"Model '{args.model}' not in config. Available: {list(config['models'].keys())}")
    model_path = config["models"][args.model]

    if args.dataset not in config["datasets"]:
        raise ValueError(f"Dataset '{args.dataset}' not in config. Available: {list(config['datasets'].keys())}")

    if args.style not in config["style_levels"]:
        raise ValueError(f"Style '{args.style}' not in config['style_levels'].")

    strengths = select_strengths(
        config_strengths=config["style_levels"][args.style],
        explicit_strengths=args.strengths,
        num_strengths=args.num_strengths,
        strategy=args.strength_strategy,
    )

    max_new_tokens = int(args.max_new_tokens or config["defaults"].get("max_new_tokens", 128))

    # Load dataset prompts via DB loader
    dataset_cfg = config["datasets"][args.dataset]
    items = load_dataset_by_name(
        args.dataset,
        sample_size=args.sample_size,
        seed=args.seed,
        config_name=dataset_cfg.get("config_name"),
        split=dataset_cfg.get("split", "validation"),
    )
    prompts = [_get_prompt_text(it) for it in items]
    n = len(prompts)
    if n == 0:
        raise SystemExit("No prompts loaded from dataset.")

    print("\n" + "=" * 110)
    print("MIRRORING EXPERIMENT (DATASET-LOADED)")
    print("=" * 110)
    print(f"Dataset: {args.dataset} | n={n} | seed={args.seed}")
    print(f"Target model: {args.model} | batch_size={args.batch_size} | max_new_tokens={max_new_tokens}")
    print(f"Style: {args.style} | places={args.places} | strengths={strengths}")
    print(f"Judge: {args.judge_provider} / {args.judge_model} | judge_max_tokens={args.judge_max_output_tokens}")
    print(f"Print examples per bucket: {bool(args.print_examples_per_bucket)}")
    print("=" * 110 + "\n")

    # Load target LLM
    model, tokenizer = load_model(
        model_path,
        device_map=config["defaults"].get("device_map", "auto"),
        dtype=config["defaults"].get("dtype", "float32"),
    )

    # ---- Baseline outputs (batched) ----
    base_raw = generate_response(
        model, tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        batch_size=args.batch_size,
    )
    base_clean = [clean_chatty_generation(o, prompt_text=p) for p, o in zip(prompts, base_raw)]

    # ---- Metrics accumulators per (place,strength) ----
    counts: Dict[Tuple[str, int], Dict[str, int]] = defaultdict(lambda: {"yes": 0, "total": 0})

    # ---- Example holders per bucket (store first YES and first NO) ----
    example_yes: Dict[Tuple[str, int], Optional[ExampleRecord]] = {(p, s): None for p in args.places for s in strengths}
    example_no: Dict[Tuple[str, int], Optional[ExampleRecord]] = {(p, s): None for p in args.places for s in strengths}

    # Progress bar = per-example judge attempts (capped)
    planned = len(args.places) * len(strengths) * n
    pbar = tqdm(total=min(planned, args.max_judge_calls), desc="judging", unit="ex")

    judge_calls = 0
    printed = 0

    for place in args.places:
        for strength in strengths:
            key = (place, strength)

            # Build styled prompts
            styled_prompts = [apply_style(p, args.style, strength, place) for p in prompts]

            # Styled outputs (batched)
            styled_raw = generate_response(
                model, tokenizer,
                prompts=styled_prompts,
                max_new_tokens=max_new_tokens,
                batch_size=args.batch_size,
            )

            # Clean styled outputs; remove echo of *styled prompt*
            styled_clean = [clean_chatty_generation(o, prompt_text=sp) for sp, o in zip(styled_prompts, styled_raw)]

            # Judge each example for this bucket
            for i in range(n):
                if judge_calls >= args.max_judge_calls:
                    break

                verdict, judge_raw, _judge_prompt = judge_with_retries(
                    judge_provider=args.judge_provider,
                    original_prompt=prompts[i],
                    original_output=base_clean[i],
                    styled_prompt=styled_prompts[i],
                    styled_output=styled_clean[i],
                    style_name=args.style,
                    strength=strength,
                    place=place,
                    judge_model=args.judge_model,
                    openai_key_env=args.openai_key_env,
                    gemini_key_env=args.gemini_key_env,
                    max_output_tokens=args.judge_max_output_tokens,
                )

                judge_calls += 1
                pbar.update(1)

                if verdict is None:
                    continue  # skip from denom

                # Update counts
                counts[key]["total"] += 1
                if verdict:
                    counts[key]["yes"] += 1

                # Save one YES and one NO example per bucket (if requested)
                if args.print_examples_per_bucket:
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

                # Optional debug printing
                if args.print_every and (judge_calls % args.print_every == 0):
                    printed += 1
                    print("\n" + "-" * 110)
                    print(f"[DEBUG #{printed}] place={place} strength={strength} i={i}")
                    print("PROMPT:", prompts[i])
                    print("BASE (clean):", base_clean[i])
                    print("STYLED PROMPT:", styled_prompts[i])
                    print("STYLED (clean):", styled_clean[i])
                    print("JUDGE RAW:", judge_raw)
                    print("VERDICT:", "YES" if verdict else "NO")
                    print("-" * 110 + "\n")

                # If we already have both examples for this bucket, we can stop judging this bucket early.
                # This does NOT change the metric unless you want it to; it would reduce total judged.
                # So we DO NOT early-stop here by default.

            if judge_calls >= args.max_judge_calls:
                break
        if judge_calls >= args.max_judge_calls:
            break

    pbar.close()

    # ---- Summary table per (place,strength) ----
    print("\n" + "=" * 110)
    print("MIRRORING RATE PER (PLACE × STRENGTH)")
    print("=" * 110)

    for place in args.places:
        print(f"\nPLACE: {place}")
        print("-" * 110)
        for strength in strengths:
            key = (place, strength)
            yes = counts[key]["yes"]
            total = counts[key]["total"]
            rate = (yes / total) if total > 0 else float("nan")
            print(f"  strength={strength:>4} | YES={yes:>5} / {total:<5} | rate={rate:.3f}")

    # ---- Print ONE YES and ONE NO per bucket (if requested) ----
    if args.print_examples_per_bucket:
        print("\n" + "=" * 110)
        print("ONE YES + ONE NO EXAMPLE PER (PLACE × STRENGTH)  (if found)")
        print("=" * 110)

        for place in args.places:
            for strength in strengths:
                key = (place, strength)
                _print_example_bucket(place, strength, example_yes[key], example_no[key])

    print("\n" + "=" * 110)
    print(f"Total judge calls attempted: {judge_calls}")
    print("Done.\n")


if __name__ == "__main__":
    main()
