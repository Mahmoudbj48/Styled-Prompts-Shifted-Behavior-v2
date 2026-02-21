"""
Politeness Safety Axes Experiment (Activations Clustering: HarmBench vs Alpaca)

This experiment is for the *politeness style* (NOT padding).

For each (place, strength):
  - Load harmless prompts from Alpaca (tatsu-lab/alpaca)
  - Load harmful prompts from HarmBench (walledai/HarmBench)
  - Apply politeness style to BOTH sets (same place+strength)
  - Extract last-layer activations
  - Compute:
      (1) t-SNE 2D on combined activations
      (2) PCA  2D on combined activations
      (3) UMAP 2D on combined activations
  - Plot 2D scatter (harmless vs harmful)
  - Compute silhouette score on high-D activations (cosine) using utils.metrics.compute_silhouette_score
  - Save:
      - per-example coords csv for tsne/pca/umap (cached path)
      - silhouette summary csv
      - plots tsne/pca/umap per strength/place
      - silhouette line plots: x=strength, y=silhouette (per model and combined)

Example:
  python experiments/safety_polite.py \
    --models L3-8B \
    --alpaca_sample_size 128 \
    --harmbench_sample_size 128 \
    --harmbench_config standard \
    --places prefix suffix global \
    --strength_range -10 10 --strength_step 2 \
    --batch_size 16
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_dataset_by_name
from utils.models import load_model
from utils.styles import apply_politeness
from utils.metrics import (
    get_layer_activations_batch,
    reduce_activations_2d,
    compute_silhouette_score,
)

# plotting + caching utilities
from utils.politeness_plots import (
    apply_neurips_style,
    save_coords2d,
    plot_2d_scatter_two_clusters,
    plot_silhouette_vs_strength,
)


# --------------------------
# Config helpers
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


def _select_strengths(
        *,
        explicit_strengths: Optional[List[int]],
        strength_range: Optional[Tuple[int, int]],
        strength_step: int,
        default_strengths: Optional[List[int]] = None,
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

    if default_strengths is not None:
        return [int(x) for x in default_strengths]

    return [-10, -6, -2, 0, 2, 6, 10]


def _get_prompt_text(item: dict) -> str:
    if "question" in item and item["question"]:
        return str(item["question"])
    if "prompt" in item and item["prompt"]:
        return str(item["prompt"])
    if "instruction" in item and item["instruction"]:
        return str(item["instruction"])
    if "text" in item and item["text"]:
        return str(item["text"])
    return str(item)


# --------------------------
# Core per-model runner
# --------------------------
def run_for_one_model(
        *,
        model_name: str,
        model_path: str,
        alpaca_items: List[dict],
        harmbench_items: List[dict],
        strengths: List[int],
        places: List[str],
        run_dir: str,
        coords_cache_dir: str,
        batch_size: int,
        seed: int,
        device_map: str,
        dtype: str,
        style_name: str = "politeness",
) -> pd.DataFrame:
    """
    Returns df_summary with one row per (model, place, strength).
    Also writes cached coords and plots.
    """

    apply_neurips_style()

    model, tokenizer = load_model(
        model_path,
        device_map=device_map,
        dtype=dtype,
    )

    harmless_prompts = [_get_prompt_text(x) for x in alpaca_items]
    harmful_prompts = [_get_prompt_text(x) for x in harmbench_items]

    n0 = len(harmless_prompts)
    n1 = len(harmful_prompts)

    harmless_ids = list(range(n0))
    harmful_ids = list(range(n1))

    summary_rows: List[dict] = []

    total = len(places) * len(strengths)
    pbar = tqdm(total=total, desc=f"[{model_name}] buckets", unit="bucket")

    for place in places:
        for s in strengths:
            # Apply politeness style to BOTH sets
            harmless_styled = [apply_politeness(p, int(s), place=place) for p in harmless_prompts]
            harmful_styled = [apply_politeness(p, int(s), place=place) for p in harmful_prompts]

            # Extract last-layer activations (NOTE: your get_layer_activations_batch likely
            # already batches internally; batch_size is kept for compatibility, but not used here
            # unless your function supports it.)
            acts0 = get_layer_activations_batch(
                model, tokenizer,
                prompts=harmless_styled,
                layer_idx=-1,
            )
            acts1 = get_layer_activations_batch(
                model, tokenizer,
                prompts=harmful_styled,
                layer_idx=-1,
            )

            # Convert to numpy
            import torch
            if isinstance(acts0, torch.Tensor):
                acts0_np = acts0.detach().cpu().numpy()
            else:
                acts0_np = np.asarray(acts0)

            if isinstance(acts1, torch.Tensor):
                acts1_np = acts1.detach().cpu().numpy()
            else:
                acts1_np = np.asarray(acts1)

            X = np.concatenate([acts0_np, acts1_np], axis=0)
            y = np.array([0] * acts0_np.shape[0] + [1] * acts1_np.shape[0], dtype=int)

            # Silhouette on high-D activations
            sil = compute_silhouette_score(X, y, metric="cosine")

            # reduce_activations_2d expects LIST of tensors
            X_t = torch.from_numpy(X).float()
            X_list = [X_t[i] for i in range(X_t.shape[0])]

            coords_tsne = reduce_activations_2d(X_list, method="tsne", seed=seed)  # (N,2)
            coords_pca = reduce_activations_2d(X_list, method="pca", seed=seed)    # (N,2)
            coords_umap = reduce_activations_2d(X_list, method="umap", seed=seed)  # (N,2)

            # Save coords to deterministic cache paths
            common_meta = {"labeling": "0=harmless,1=harmful"}

            tsne_coords_path = save_coords2d(
                cache_dir=coords_cache_dir,
                dataset_name="alpaca_vs_harmbench",
                style_name=style_name,
                model=model_name,
                place=place,
                strength=int(s),
                method="tsne",
                prompt_ids=(harmless_ids + harmful_ids),
                coords=coords_tsne,
                meta=common_meta,
            )
            pca_coords_path = save_coords2d(
                cache_dir=coords_cache_dir,
                dataset_name="alpaca_vs_harmbench",
                style_name=style_name,
                model=model_name,
                place=place,
                strength=int(s),
                method="pca",
                prompt_ids=(harmless_ids + harmful_ids),
                coords=coords_pca,
                meta=common_meta,
            )
            umap_coords_path = save_coords2d(
                cache_dir=coords_cache_dir,
                dataset_name="alpaca_vs_harmbench",
                style_name=style_name,
                model=model_name,
                place=place,
                strength=int(s),
                method="umap",
                prompt_ids=(harmless_ids + harmful_ids),
                coords=coords_umap,
                meta=common_meta,
            )

            # Plot paths
            tsne_plot_path = os.path.join(run_dir, "plots_tsne", model_name, f"tsne_{place}_s{int(s)}.png")
            pca_plot_path = os.path.join(run_dir, "plots_pca", model_name, f"pca_{place}_s{int(s)}.png")
            umap_plot_path = os.path.join(run_dir, "plots_umap", model_name, f"umap_{place}_s{int(s)}.png")

            # Plot combined 2D with two clusters
            plot_2d_scatter_two_clusters(
                coords=coords_tsne,
                labels=y,
                out_path_png=tsne_plot_path,
                title=f"t-SNE last-layer | {model_name} | place={place} | strength={int(s)}",
                xlabel="t-SNE 1",
                ylabel="t-SNE 2",
                legend_labels=("harmless", "harmful"),
            )
            plot_2d_scatter_two_clusters(
                coords=coords_pca,
                labels=y,
                out_path_png=pca_plot_path,
                title=f"PCA last-layer | {model_name} | place={place} | strength={int(s)}",
                xlabel="PCA 1",
                ylabel="PCA 2",
                legend_labels=("harmless", "harmful"),
            )
            plot_2d_scatter_two_clusters(
                coords=coords_umap,
                labels=y,
                out_path_png=umap_plot_path,
                title=f"UMAP last-layer | {model_name} | place={place} | strength={int(s)}",
                xlabel="UMAP 1",
                ylabel="UMAP 2",
                legend_labels=("harmless", "harmful"),
            )

            summary_rows.append({
                "model": model_name,
                "place": place,
                "strength": int(s),
                "n_alpaca": int(n0),
                "n_harmbench": int(n1),
                "silhouette_cosine": float(sil) if np.isfinite(sil) else np.nan,
                "tsne_plot_path": tsne_plot_path,
                "pca_plot_path": pca_plot_path,
                "umap_plot_path": umap_plot_path,
                "tsne_coords_path": tsne_coords_path,
                "pca_coords_path": pca_coords_path,
                "umap_coords_path": umap_coords_path,
            })

            pbar.update(1)

    pbar.close()

    df_summary = pd.DataFrame(summary_rows)

    out_sum = os.path.join(run_dir, f"{model_name}_politeness_safety_summary.csv")
    df_summary.to_csv(out_sum, index=False)
    print(f"✓ Saved summary: {out_sum}")

    # silhouette line plot per model (lines=place)
    metrics_dir = os.path.join(run_dir, "plots_metrics", model_name)
    os.makedirs(metrics_dir, exist_ok=True)

    out_line = os.path.join(metrics_dir, "silhouette_vs_strength_by_place.png")
    plot_silhouette_vs_strength(
        df_summary=df_summary,
        out_path_png=out_line,
        group_by=["place"],
        title=f"Silhouette vs Strength | {model_name} | politeness",
        legend_outside=True,
    )
    print(f"✓ Plot saved: {out_line}")

    return df_summary


# --------------------------
# Multi-model runner
# --------------------------
def run_experiment(
        *,
        models: List[str],
        alpaca_sample_size: int,
        harmbench_sample_size: int,
        harmbench_config: str,
        places: List[str],
        strengths: List[int],
        batch_size: int,
        seed: int,
        coords_cache_dir: str,
) -> str:
    config = load_config()
    models = _normalize_models(models, config)

    device_map = config["defaults"].get("device_map", "auto")
    dtype = config["defaults"].get("dtype", "float32")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    run_dir = os.path.join(base_results_dir, "politeness_safety", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("POLITENESS SAFETY AXES (t-SNE + PCA + UMAP + Silhouette)")
    print(f"{'='*80}")
    print(f"Models: {models}")
    print(f"Alpaca sample_size: {alpaca_sample_size}")
    print(f"HarmBench sample_size: {harmbench_sample_size} (config={harmbench_config})")
    print(f"Places: {places}")
    print(f"Strengths: {strengths}")
    print(f"Batch size: {batch_size}")
    print(f"Seed: {seed}")
    print(f"Coords cache dir: {coords_cache_dir}")
    print(f"Output dir: {run_dir}")
    print(f"{'='*80}\n")

    # Load datasets via your factory
    alpaca_items = load_dataset_by_name(
        "alpaca",
        sample_size=int(alpaca_sample_size),
        seed=int(seed),
    )
    harmbench_items = load_dataset_by_name(
        "harmbench",
        sample_size=int(harmbench_sample_size),
        seed=int(seed),
        config_name=str(harmbench_config),
    )

    all_sum = []
    for model_name in models:
        model_path = config["models"][model_name]
        df_s = run_for_one_model(
            model_name=model_name,
            model_path=model_path,
            alpaca_items=alpaca_items,
            harmbench_items=harmbench_items,
            strengths=strengths,
            places=places,
            run_dir=run_dir,
            coords_cache_dir=coords_cache_dir,
            batch_size=int(batch_size),
            seed=int(seed),
            device_map=str(device_map),
            dtype=str(dtype),
            style_name="politeness",
        )
        all_sum.append(df_s)

    df_all = pd.concat(all_sum, ignore_index=True) if all_sum else pd.DataFrame()

    out_all = os.path.join(run_dir, "politeness_safety_summary_all_models.csv")
    df_all.to_csv(out_all, index=False)
    print(f"✓ Saved all-model summary: {out_all}")

    # Combined silhouette plot: lines = model/place
    metrics_dir = os.path.join(run_dir, "plots_metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    out_line_all = os.path.join(metrics_dir, "silhouette_vs_strength_by_model_place.png")
    plot_silhouette_vs_strength(
        df_summary=df_all,
        out_path_png=out_line_all,
        group_by=["model", "place"],
        title="Silhouette vs Strength | politeness | all models/places",
        legend_outside=True,
    )
    print(f"✓ Plot saved: {out_line_all}")

    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Politeness safety axes: t-SNE + PCA + UMAP + silhouette on Alpaca (harmless) vs HarmBench (harmful)."
    )

    parser.add_argument("--models", nargs="+", default=["L3-8B"], help="Model keys from config.yaml or 'all'.")

    parser.add_argument("--alpaca_sample_size", type=int, default=128)
    parser.add_argument("--harmbench_sample_size", type=int, default=128)
    parser.add_argument("--harmbench_config", type=str, default="standard")

    parser.add_argument("--places", nargs="+", default=["prefix"], help="prefix suffix global")

    parser.add_argument("--strengths", nargs="+", type=int, default=None)
    parser.add_argument("--strength_range", nargs=2, type=int, default=None, metavar=("LO", "HI"))
    parser.add_argument("--strength_step", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--coords_cache_dir",
        type=str,
        default=None,
        help="Where to save/load cached 2D coords CSVs (default: ../data/embeddings_2d_cache)."
    )

    args = parser.parse_args()

    config = load_config()
    default_strengths = config.get("style_levels", {}).get("politeness", None)

    strengths = _select_strengths(
        explicit_strengths=args.strengths,
        strength_range=tuple(args.strength_range) if args.strength_range else None,
        strength_step=int(args.strength_step),
        default_strengths=default_strengths,
    )

    coords_cache_dir = args.coords_cache_dir
    if coords_cache_dir is None:
        coords_cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "embeddings_2d_cache")

    run_experiment(
        models=args.models,
        alpaca_sample_size=int(args.alpaca_sample_size),
        harmbench_sample_size=int(args.harmbench_sample_size),
        harmbench_config=str(args.harmbench_config),
        places=args.places,
        strengths=strengths,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        coords_cache_dir=str(coords_cache_dir),
    )


if __name__ == "__main__":
    main()