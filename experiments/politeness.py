"""
Politeness Style Experiment (BATCHED, progress fixed)

Key fixes:
- Batching is correct and deterministic.
- tqdm now reflects what you care about:
  * batch_pbar counts (batch_chunk, place, strength) groups
  * row_pbar optionally counts per-example rows
- Ensures batch_size comes from CLI (e.g., 21) and is used consistently.

Run:
  python experiments/politeness.py --models llama --dataset truthful_qa --sample_size 128 --experiments all --batch_size 21
"""

import argparse
import os
import sys
import yaml
from datetime import datetime
from typing import Dict, List, Set, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_dataset_by_name
from utils.models import load_model, generate_response  # batched: prompts -> list[str]
from utils.metrics import (
    compute_bleu,
    compute_bertscore,
    compute_confidence,
    get_layer_activations_batch,
    reduce_activations_2d,
)
from utils.styles import apply_politeness


VALID_EXPERIMENTS = {"prompt", "response", "activation", "confidence"}


# --------------------------
# Config + CLI helpers
# --------------------------
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _normalize_experiments(experiments: Optional[List[str]]) -> Set[str]:
    if experiments is None:
        return set(VALID_EXPERIMENTS)

    exp = [e.strip().lower() for e in experiments]
    if "all" in exp:
        return set(VALID_EXPERIMENTS)

    unknown = [e for e in exp if e not in VALID_EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}. Valid: {sorted(VALID_EXPERIMENTS)} or 'all'.")
    return set(exp)


def _normalize_models(models: List[str], config: dict) -> List[str]:
    available = list(config.get("models", {}).keys())
    if not available:
        raise ValueError("No models found in config.yaml under 'models'.")

    m = [x.strip().lower() for x in models]
    if len(m) == 1 and m[0] == "all":
        return available

    unknown = [x for x in m if x not in available]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Available: {available} or 'all'.")
    return m


def _get_places(config: dict) -> List[str]:
    places = [p for p in config.get("style_positions", {}).get("politeness", []) if p != "middle"]
    if not places:
        places = ["prefix", "suffix", "global"]
    places = [p for p in places if p in {"prefix", "suffix", "global"}]
    if not places:
        raise ValueError("No valid politeness positions after excluding 'middle'. Use prefix/suffix/global.")
    return places


def _num_batches(n: int, bs: int) -> int:
    return (n + bs - 1) // bs


# --------------------------
# Plotting helpers (unchanged)
# --------------------------
def plot_metric_lines(df_mean: pd.DataFrame, metric: str, strengths: List[int], places: List[str],
                      models: List[str], out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    strengths_sorted = sorted(strengths)

    for model in models:
        for place in places:
            sub = df_mean[(df_mean["model"] == model) & (df_mean["place"] == place)]
            if sub.empty:
                continue
            sub = sub.set_index("strength").reindex(strengths_sorted)
            y = sub[metric].values
            ax.plot(strengths_sorted, y, marker="o", label=f"{model}/{place}")

    ax.set_xlabel("Strength")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _radar_angles(n: int):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    return angles


def plot_radar(radar_df: pd.DataFrame, metrics: List[str], out_path: str, title: str):
    if radar_df.empty:
        return

    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)

    labels = metrics[:]
    angles = _radar_angles(len(labels))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)

    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1)

    for _, row in radar_df.iterrows():
        vals = [float(row[m]) for m in labels]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=f"{row['model']}/{row['place']}")
        ax.fill(angles, vals, alpha=0.08)

    ax.set_title(title, y=1.10)
    ax.legend(bbox_to_anchor=(1.25, 1.10), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_radar_table(df_mean: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    base = df_mean.groupby(["model", "place"], dropna=False)[metrics].mean().reset_index()

    norm = base.copy()
    for m in metrics:
        col = pd.to_numeric(base[m], errors="coerce")
        mn, mx = np.nanmin(col.values), np.nanmax(col.values)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
            norm[m] = 0.5
        else:
            norm[m] = (col - mn) / (mx - mn)
    return norm


# --------------------------
# Main experiment per model (FIXED PROGRESS + BATCHING)
# --------------------------
def run_for_one_model(
        model_name: str,
        model_path: str,
        prompts: List[dict],
        strength_levels: List[int],
        places: List[str],
        config: dict,
        run_dir: str,
        experiments: Set[str],
        *,
        batch_size: int,
        max_new_tokens: int,
        show_row_pbar: bool = False,
) -> pd.DataFrame:

    llm_experiments = {"response", "activation", "confidence"}
    run_llm_phase = len(experiments.intersection(llm_experiments)) > 0

    # Only load model if needed
    model = None
    tokenizer = None
    if run_llm_phase:
        model, tokenizer = load_model(
            model_path,
            device_map=config["defaults"].get("device_map", "auto"),
            dtype=config["defaults"].get("dtype", "float32"),
        )

    seed = int(config["defaults"].get("random_seed", 42))

    activations_cache = {(place, s): [] for place in places for s in strength_levels} if "activation" in experiments else None

    rows: List[dict] = []

    n = len(prompts)
    n_batches = _num_batches(n, batch_size)

    # Each “group” = (batch_chunk, place, strength)
    total_groups = n_batches * len(places) * len(strength_levels)
    batch_pbar = tqdm(total=total_groups, desc=f"[{model_name}] batch-groups", unit="group")

    row_pbar = None
    if show_row_pbar:
        total_rows = n * len(places) * len(strength_levels)
        row_pbar = tqdm(total=total_rows, desc=f"[{model_name}] rows", unit="row", leave=False)

    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n)
        batch_items = prompts[start:end]
        batch_ids = list(range(start, end))
        batch_orig_prompts = [it["question"] for it in batch_items]
        batch_categories = [it.get("category", "Unknown") for it in batch_items]

        # Baseline responses for this batch (computed ONCE per batch)
        batch_response_orig = None
        if run_llm_phase and ("response" in experiments or "confidence" in experiments):
            batch_response_orig = generate_response(
                model, tokenizer,
                prompts=batch_orig_prompts,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,   # your requested generation batch size (e.g. 21)
            )
            if len(batch_response_orig) != len(batch_orig_prompts):
                raise RuntimeError("generate_response returned wrong number of baseline outputs.")

        # Baseline activations for this batch (computed ONCE per batch)
        batch_act_orig = None
        if run_llm_phase and ("activation" in experiments):
            batch_act_orig = get_layer_activations_batch(
                model, tokenizer,
                prompts=batch_orig_prompts,
                layer_idx=-1,
            )
            if batch_act_orig.shape[0] != len(batch_orig_prompts):
                raise RuntimeError("get_layer_activations_batch returned wrong batch size.")

        # Now iterate place/strength, but update progress PER GROUP (not per example)
        for place in places:
            for strength in strength_levels:
                batch_pert_prompts = [apply_politeness(p, strength, place=place) for p in batch_orig_prompts]

                # Prompt BERTScore (per-example)
                batch_prompt_bs = None
                if "prompt" in experiments:
                    batch_prompt_bs = [
                        compute_bertscore(o, s, device="cpu")
                        for o, s in zip(batch_orig_prompts, batch_pert_prompts)
                    ]

                # Styled responses (batched)
                batch_response_pert = None
                if run_llm_phase and ("response" in experiments or "confidence" in experiments):
                    batch_response_pert = generate_response(
                        model, tokenizer,
                        prompts=batch_pert_prompts,
                        max_new_tokens=max_new_tokens,
                        batch_size=batch_size,
                    )
                    if len(batch_response_pert) != len(batch_pert_prompts):
                        raise RuntimeError("generate_response returned wrong number of styled outputs.")

                # Styled activations (batched)
                batch_act_pert = None
                if run_llm_phase and ("activation" in experiments):
                    batch_act_pert = get_layer_activations_batch(
                        model, tokenizer,
                        prompts=batch_pert_prompts,
                        layer_idx=-1,
                    )
                    if batch_act_pert.shape[0] != len(batch_pert_prompts):
                        raise RuntimeError("get_layer_activations_batch returned wrong batch size (styled).")

                # Per-example rows
                for j in range(len(batch_items)):
                    i = batch_ids[j]
                    orig = batch_orig_prompts[j]
                    pert = batch_pert_prompts[j]
                    cat = batch_categories[j]

                    row = {
                        "model": model_name,
                        "prompt_id": i,
                        "place": place,
                        "strength": int(strength),
                        "category": cat,
                        "prompt_orig": orig,
                        "prompt_pert": pert,
                    }

                    if "prompt" in experiments:
                        row["bertscore_prompt"] = float(batch_prompt_bs[j])

                    if run_llm_phase and ("response" in experiments or "confidence" in experiments):
                        row["response_orig"] = batch_response_orig[j]
                        row["response_pert"] = batch_response_pert[j]

                    if run_llm_phase and ("response" in experiments):
                        ro = batch_response_orig[j]
                        rp = batch_response_pert[j]
                        row["bleu"] = float(compute_bleu(ro, rp))
                        row["bertscore_response"] = float(compute_bertscore(ro, rp, device=str(model.device)))

                    if run_llm_phase and ("activation" in experiments):
                        a0 = batch_act_orig[j]
                        a1 = batch_act_pert[j]
                        row["activation_similarity"] = float(F.cosine_similarity(a0.unsqueeze(0), a1.unsqueeze(0)).item())
                        activations_cache[(place, strength)].append(a1.detach())

                    if run_llm_phase and ("confidence" in experiments):
                        conf = compute_confidence(model, tokenizer, orig, pert, batch_response_orig[j])
                        row.update(conf)

                    rows.append(row)
                    if row_pbar is not None:
                        row_pbar.update(1)

                # Update once per (batch, place, strength)
                batch_pbar.update(1)

    batch_pbar.close()
    if row_pbar is not None:
        row_pbar.close()

    df = pd.DataFrame(rows)
    out_csv = os.path.join(run_dir, f"{model_name}_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"✓ Saved per-example results: {out_csv}")

    # t-SNE plots if activation enabled
    if "activation" in experiments and activations_cache is not None:
        plot_dir = os.path.join(run_dir, "plots_tsne", model_name)
        os.makedirs(plot_dir, exist_ok=True)
        viz_sample_size = min(50, len(prompts))

        for place in places:
            fig, ax = plt.subplots(figsize=(10, 7))
            for strength in sorted(strength_levels):
                acts = activations_cache[(place, strength)][:viz_sample_size]
                if len(acts) < 2:
                    continue
                coords = reduce_activations_2d(acts, method="tsne", seed=seed)
                ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=35, label=f"s={strength}")

            ax.set_title(f"t-SNE Activations (last layer) - {model_name} - {place}")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"tsne_{place}.png"), dpi=250, bbox_inches="tight")
            plt.close(fig)

    return df


# --------------------------
# Global runner (multi-model)
# --------------------------
def run_experiment(models: List[str], dataset_name: str, sample_size: Optional[int], experiments: Optional[List[str]],
                   *, batch_size: int, max_new_tokens: Optional[int], show_row_pbar: bool):
    config = load_config()
    experiments_set = _normalize_experiments(experiments)

    if dataset_name not in config["datasets"]:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(config['datasets'].keys())}")
    dataset_config = config["datasets"][dataset_name]
    if sample_size is None:
        sample_size = dataset_config["sample_size"]

    strength_levels = config["style_levels"]["politeness"]
    places = _get_places(config)

    # tokens default
    if max_new_tokens is None:
        max_new_tokens = int(config["defaults"].get("max_new_tokens", 256))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    style_dir = os.path.join(base_results_dir, "politeness")
    run_dir = os.path.join(style_dir, f"run_multi_{dataset_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("POLITENESS EXPERIMENT (MULTI-MODEL, BATCHED)")
    print(f"{'='*80}")
    print(f"Models: {models}")
    print(f"Dataset: {dataset_name}")
    print(f"Experiments: {sorted(experiments_set)}")
    print(f"Sample size: {sample_size}")
    print(f"Strengths: {strength_levels}")
    print(f"Places (middle skipped): {places}")
    print(f"Batch size (requested): {batch_size}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Output dir: {run_dir}")
    print(f"{'='*80}\n")

    prompts = load_dataset_by_name(
        dataset_name,
        sample_size=sample_size,
        seed=config["defaults"]["random_seed"],
        config_name=dataset_config.get("config_name"),
        split=dataset_config.get("split", "validation"),
    )

    # Sanity print: confirm batching math for your case
    n_batches = _num_batches(len(prompts), batch_size)
    print(f"Batching sanity: n_prompts={len(prompts)}, batch_size={batch_size} -> n_batches={n_batches}")

    all_rows = []
    for model_name in models:
        model_path = config["models"][model_name]
        df_model = run_for_one_model(
            model_name=model_name,
            model_path=model_path,
            prompts=prompts,
            strength_levels=strength_levels,
            places=places,
            config=config,
            run_dir=run_dir,
            experiments=experiments_set,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            show_row_pbar=show_row_pbar,
        )
        if df_model is not None and not df_model.empty:
            all_rows.append(df_model)

    if not all_rows:
        print("No outputs produced (check experiments selection).")
        return run_dir

    df_all = pd.concat(all_rows, ignore_index=True)
    full_path = os.path.join(run_dir, "full_results_all_models.csv")
    df_all.to_csv(full_path, index=False)
    print(f"✓ Saved combined full results: {full_path}")

    metric_cols = [c for c in [
        "bertscore_prompt",
        "bleu",
        "bertscore_response",
        "delta_log_prob",
        "entropy_shift",
        "jsd_drift",
        "activation_similarity",
    ] if c in df_all.columns]

    df_mean = (
        df_all.groupby(["model", "place", "strength"], dropna=False)[metric_cols]
        .mean()
        .reset_index()
        .sort_values(["model", "place", "strength"])
    )
    mean_path = os.path.join(run_dir, "means_by_model_place_strength.csv")
    df_mean.to_csv(mean_path, index=False)
    print(f"✓ Saved means table: {mean_path}")

    plot_dir = os.path.join(run_dir, "plots_metrics")
    os.makedirs(plot_dir, exist_ok=True)

    for metric in metric_cols:
        out_png = os.path.join(plot_dir, f"{metric}_vs_strength.png")
        plot_metric_lines(
            df_mean=df_mean,
            metric=metric,
            strengths=strength_levels,
            places=places,
            models=models,
            out_path=out_png,
            title=f"{metric} vs Strength ({dataset_name})",
        )
        print(f"✓ Plot saved: {out_png}")

    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Run politeness style experiment (multi-model, batched)")
    parser.add_argument("--models", nargs="+", default=["llama"], help="Model names from config or 'all'")
    parser.add_argument("--dataset", type=str, default="truthful_qa", help="Dataset name (as in config)")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of prompts (default from config)")
    parser.add_argument("--experiments", nargs="+", default=["all"], help="all OR subset of: prompt response activation confidence")

    # IMPORTANT: explicitly control batch_size from CLI
    parser.add_argument("--batch_size", type=int, default=21, help="Generation + activation batch size (e.g., 21)")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Override max_new_tokens")
    parser.add_argument("--show_row_pbar", action="store_true", help="Also show per-row progress bar")

    args = parser.parse_args()

    config = load_config()
    models = _normalize_models(args.models, config)

    run_experiment(
        models=models,
        dataset_name=args.dataset,
        sample_size=args.sample_size,
        experiments=args.experiments,
        batch_size=int(args.batch_size),
        max_new_tokens=args.max_new_tokens,
        show_row_pbar=bool(args.show_row_pbar),
    )


if __name__ == "__main__":
    main()
