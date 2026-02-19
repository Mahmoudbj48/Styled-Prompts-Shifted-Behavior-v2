"""
Trace-length metric experiment (sample of prompts) — AVG per place & strength


Run:
  python experiments/trace.py --model llama --dataset truthful_qa --sample_size 128 --places prefix suffix global --batch_size 64 --max_new_tokens 128
"""

from __future__ import annotations

import argparse
import os
import sys
import yaml
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_dataset_by_name
from utils.models import load_model, generate_response
from utils.styles import apply_politeness

# Uses your FIXED metrics code (always-valid JSON + optional second-pass fixer)
from utils.metrics import compute_trace_metric_batched


# --------------------------
# Helpers
# --------------------------
def load_config() -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def apply_style(prompt: str, style: str, strength: int, place: str) -> str:
    if style == "politeness":
        return apply_politeness(prompt, strength, place=place)
    raise ValueError(f"Unknown style: {style}")


def plot_metric_lines_place(
        df_mean: pd.DataFrame,
        *,
        metric: str,
        strengths: List[int],
        places: List[str],
        out_path: str,
        title: str,
):
    """
    Single plot:
      X = strength
      Lines = place
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    strengths_sorted = sorted([int(s) for s in strengths])

    for place in places:
        sub = df_mean[df_mean["place"] == place].copy()
        if sub.empty:
            continue
        sub = sub.set_index("strength").reindex(strengths_sorted)
        y = sub[metric].values
        ax.plot(strengths_sorted, y, marker="o", label=f"{place}")

    ax.set_xlabel("Strength")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama", help="Model key from config.yaml")
    parser.add_argument("--dataset", type=str, default="truthful_qa", help="Dataset name (as in config.yaml)")

    # requested defaults
    parser.add_argument("--sample_size", type=int, default=128, help="Number of prompts to test (default=128)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for LLM generation (default=64)")

    parser.add_argument("--style", type=str, default="politeness", help="Style name (politeness)")
    parser.add_argument("--places", nargs="+", default=["prefix", "suffix", "global"], help="Places to test")

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Override max_new_tokens. If None, uses max(384, config_default).",
    )

    parser.add_argument(
        "--fix_bad_outputs",
        action="store_true",
        help="If set, run a second-pass JSON fixer when output is not valid JSON.",
    )
    parser.add_argument(
        "--fix_max_new_tokens",
        type=int,
        default=256,
        help="Max tokens for the second-pass fixer (only if --fix_bad_outputs).",
    )

    parser.add_argument("--out_dir", type=str, default=None, help="Optional output directory")

    # saving behavior
    parser.add_argument(
        "--save_per_prompt",
        action="store_true",
        help="If set, saves per-prompt rows (larger file). Aggregate CSV is always saved.",
    )

    args = parser.parse_args()

    config = load_config()

    # ---- Validate config keys ----
    if args.model not in config.get("models", {}):
        raise ValueError(f"Model '{args.model}' not in config. Available: {list(config.get('models', {}).keys())}")
    model_path = config["models"][args.model]

    if args.dataset not in config.get("datasets", {}):
        raise ValueError(f"Dataset '{args.dataset}' not in config. Available: {list(config.get('datasets', {}).keys())}")
    dataset_cfg = config["datasets"][args.dataset]

    if args.style not in config.get("style_levels", {}):
        raise ValueError(
            f"Style '{args.style}' not in config['style_levels']. "
            f"Available: {list(config.get('style_levels', {}).keys())}"
        )
    strengths: List[int] = list(config["style_levels"][args.style])
    strengths = [int(s) for s in strengths]

    # ---- Fix max tokens ----
    cfg_default = _safe_int(config.get("defaults", {}).get("max_new_tokens", 128), default=128)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else max(384, cfg_default)

    # ---- Output dir ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    run_dir = args.out_dir or os.path.join(base_results_dir, "trace", f"run_{args.dataset}_{args.model}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("\n" + "=" * 90)
    print("TRACE-LENGTH EXPERIMENT (SAMPLE) — AVG TRACE LENGTH PER PLACE & STRENGTH")
    print("=" * 90)
    print(f"Model: {args.model} -> {model_path}")
    print(f"Dataset: {args.dataset} | sample_size={args.sample_size}")
    print(f"Style: {args.style}")
    print(f"Places: {args.places}")
    print(f"Strengths: {strengths}")
    print(f"batch_size: {args.batch_size}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"fix_bad_outputs: {args.fix_bad_outputs} | fix_max_new_tokens: {args.fix_max_new_tokens}")
    print(f"Output dir: {run_dir}")
    print("=" * 90 + "\n")

    # ---- Load prompts sample ----
    prompts_items = load_dataset_by_name(
        args.dataset,
        sample_size=args.sample_size,
        seed=_safe_int(config.get("defaults", {}).get("random_seed", 42), default=42),
        config_name=dataset_cfg.get("config_name"),
        split=dataset_cfg.get("split", "validation"),
    )
    if not prompts_items:
        raise RuntimeError("Dataset loader returned zero items.")

    prompts: List[str] = [x["question"] for x in prompts_items]
    n_prompts = len(prompts)

    # sanity: confirm sample size is what you asked for (some datasets may return fewer)
    if n_prompts != args.sample_size:
        print(f"WARNING: requested sample_size={args.sample_size}, but loader returned n={n_prompts}")

    # ---- Load LLM ----
    model, tokenizer = load_model(
        model_path,
        device_map=config.get("defaults", {}).get("device_map", "auto"),
        dtype=config.get("defaults", {}).get("dtype", "float32"),
    )

    # ---- Aggregate + optional per-prompt rows ----
    agg_rows: List[Dict[str, Any]] = []
    per_rows: List[Dict[str, Any]] = []

    # tqdm counts groups (place x strength) correctly
    pbar = tqdm(total=len(args.places) * len(strengths), desc="Trace groups", unit="group")

    for place in args.places:
        for s in strengths:
            styled_prompts = [apply_style(p, args.style, int(s), place) for p in prompts]

            # NOTE: compute_trace_metric_batched should internally build_trace_prompt
            # If your implementation expects raw prompts and does build_trace_prompt inside, keep this.
            # If it expects already-trace-instructed prompts, change in utils.metrics accordingly.
            result = compute_trace_metric_batched(
                model,
                tokenizer,
                styled_prompts,
                max_new_tokens=int(max_new_tokens),
                batch_size=int(args.batch_size),
                do_sample=False,
                use_cache=False,
                generate_fn=generate_response,
                fix_bad_outputs=bool(args.fix_bad_outputs),
                fix_max_new_tokens=int(args.fix_max_new_tokens),
            )

            trace_lengths = result["trace_lengths"]
            parsed_flags = result["parsed_flags"]

            # IMPORTANT: mean should ignore failures (NaN), so use nanmean
            avg_trace_len = float(np.nanmean(trace_lengths)) if len(trace_lengths) else float("nan")
            parsed_rate = float(np.mean(parsed_flags)) if len(parsed_flags) else float("nan")

            agg_rows.append(
                {
                    "model": args.model,
                    "dataset": args.dataset,
                    "place": place,
                    "strength": int(s),
                    "n": int(n_prompts),
                    "avg_trace_len": avg_trace_len,
                    "parsed_rate": parsed_rate,
                }
            )

            if args.save_per_prompt:
                raw_outputs = result.get("raw_outputs", [None] * n_prompts)
                json_outputs = result.get("json_outputs", [None] * n_prompts)
                # if your metric returns "trace_json_prompts", store them too (optional)
                trace_prompts = result.get("trace_prompts", None)

                for i in range(n_prompts):
                    per_rows.append(
                        {
                            "model": args.model,
                            "dataset": args.dataset,
                            "place": place,
                            "strength": int(s),
                            "prompt_id": int(i),
                            "prompt_orig": prompts[i],
                            "prompt_styled": styled_prompts[i],
                            "trace_len": float(trace_lengths[i]),
                            "parsed_ok": bool(parsed_flags[i]),
                            "trace_prompt": trace_prompts[i] if trace_prompts is not None else None,
                            "raw_output": raw_outputs[i],
                            "json_output": json_outputs[i],
                        }
                    )

            pbar.update(1)

    pbar.close()

    # ---- Aggregate DF ----
    df_agg = pd.DataFrame(agg_rows).sort_values(["place", "strength"]).reset_index(drop=True)

    print("\n" + "=" * 90)
    print("AVG TRACE LENGTH (by place, strength)")
    print("=" * 90)
    print(df_agg.to_string(index=False))
    print("=" * 90 + "\n")

    # ---- SAVE CSVs (ALWAYS for aggregate) ----
    out_agg = os.path.join(run_dir, "trace_avg_by_place_strength.csv")
    df_agg.to_csv(out_agg, index=False)
    print("Saved aggregate CSV:", out_agg)

    if args.save_per_prompt:
        df_per = pd.DataFrame(per_rows)
        out_per = os.path.join(run_dir, "trace_per_prompt.csv")
        df_per.to_csv(out_per, index=False)
        print("Saved per-prompt CSV:", out_per)

    # ---- PLOT: avg_trace_len vs strength (lines by place) ----
    plot_dir = os.path.join(run_dir, "plots_metrics")
    os.makedirs(plot_dir, exist_ok=True)

    # If you later add multiple models, you'd group by model too.
    df_mean = (
        df_agg.groupby(["place", "strength"], dropna=False)[["avg_trace_len", "parsed_rate"]]
        .mean()
        .reset_index()
        .sort_values(["place", "strength"])
    )

    out_png = os.path.join(plot_dir, "avg_trace_len_vs_strength.png")
    plot_metric_lines_place(
        df_mean=df_mean,
        metric="avg_trace_len",
        strengths=strengths,
        places=args.places,
        out_path=out_png,
        title=f"avg_trace_len vs Strength ({args.dataset}) - model={args.model}",
    )
    print("Saved plot:", out_png)

    out_png2 = os.path.join(plot_dir, "parsed_rate_vs_strength.png")
    plot_metric_lines_place(
        df_mean=df_mean,
        metric="parsed_rate",
        strengths=strengths,
        places=args.places,
        out_path=out_png2,
        title=f"parsed_rate vs Strength ({args.dataset}) - model={args.model}",
    )
    print("Saved plot:", out_png2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
