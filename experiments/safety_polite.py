# experiments/safety_polite.py

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_dataset_by_name
from utils.models import load_model, generate_response
from utils.styles import apply_politeness
from utils.metrics import *

from utils.politeness_plots import (
    apply_neurips_style,
    save_coords2d,
    plot_2d_scatter_two_clusters,
    plot_silhouette_vs_strength,
    plot_metric_vs_strength,
)


def load_config() -> Dict[str, any]:
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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
    for k in ("question", "prompt", "instruction", "text"):
        if k in item and item[k]:
            return str(item[k])
    return str(item)


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
        compute_asr: bool = True,
        guard_max_new_tokens: int = 24,
        gen_max_new_tokens: int = 128,
) -> pd.DataFrame:
    """
    One row per (place, strength).
    ASR here = compliance rate on harmful prompts (COMPLIED/REFUSED judge).
    """

    apply_neurips_style()

    print(f"\n[MAIN] Loading tested model: {model_name}")
    model, tokenizer = load_model(model_path, device_map=device_map, dtype=dtype)
    print("[MAIN] ✓ Tested model loaded")

    judge = None
    guard = LlamaGuard3Scorer(
        model_path="meta-llama/Llama-Guard-3-8B",
        vllm_mode=False,          # set True if you want vLLM
        local_files_only=False,   # set True if your env is offline
    )


    harmless_prompts = [_get_prompt_text(x) for x in alpaca_items]
    harmful_prompts = [_get_prompt_text(x) for x in harmbench_items]

    n0 = len(harmless_prompts)
    n1 = len(harmful_prompts)

    harmless_ids = list(range(n0))
    harmful_ids = list(range(n1))

    summary_rows: List[dict] = []

    total = len(places) * len(strengths)
    pbar = tqdm(total=total, desc=f"[{model_name}] place×strength", unit="bucket")

    bucket_idx = 0
    for place in places:
        for s in strengths:
            bucket_idx += 1
            s_int = int(s)

            print(f"\n[BUCKET {bucket_idx}/{total}] place={place} strength={s_int}")

            # 1) Style prompts
            print("[BUCKET] Applying politeness style")
            harmless_styled = [apply_politeness(p, s_int, place=place) for p in harmless_prompts]
            harmful_styled = [apply_politeness(p, s_int, place=place) for p in harmful_prompts]

            # 2) Activations
            print("[BUCKET] Extracting last-layer activations")
            acts0 = get_layer_activations_batch(model, tokenizer, prompts=harmless_styled, layer_idx=-1)
            acts1 = get_layer_activations_batch(model, tokenizer, prompts=harmful_styled, layer_idx=-1)

            import torch
            acts0_np = acts0.detach().cpu().numpy() if isinstance(acts0, torch.Tensor) else np.asarray(acts0)
            acts1_np = acts1.detach().cpu().numpy() if isinstance(acts1, torch.Tensor) else np.asarray(acts1)

            X = np.concatenate([acts0_np, acts1_np], axis=0)
            y = np.array([0] * acts0_np.shape[0] + [1] * acts1_np.shape[0], dtype=int)

            # 3) Silhouette
            print("[BUCKET] Computing silhouette score (cosine)")
            sil = compute_silhouette_score(X, y, metric="cosine")

            # 4) 2D reductions
            print("[BUCKET] Computing 2D reductions (t-SNE, PCA, UMAP)")
            X_t = torch.from_numpy(X).float()
            X_list = [X_t[i] for i in range(X_t.shape[0])]
            coords_tsne = reduce_activations_2d(X_list, method="tsne", seed=seed)
            coords_pca = reduce_activations_2d(X_list, method="pca", seed=seed)
            coords_umap = reduce_activations_2d(X_list, method="umap", seed=seed)

            # 5) Save coords
            print("[BUCKET] Saving 2D coordinates")
            common_meta = {"labels": "0=harmless,1=harmful"}
            dataset_tag = "alpaca_vs_harmbench"

            tsne_coords_path = save_coords2d(
                cache_dir=coords_cache_dir,
                dataset_name=dataset_tag,
                style_name=style_name,
                model=model_name,
                place=place,
                strength=s_int,
                method="tsne",
                prompt_ids=(harmless_ids + harmful_ids),
                coords=coords_tsne,
                meta=common_meta,
            )
            pca_coords_path = save_coords2d(
                cache_dir=coords_cache_dir,
                dataset_name=dataset_tag,
                style_name=style_name,
                model=model_name,
                place=place,
                strength=s_int,
                method="pca",
                prompt_ids=(harmless_ids + harmful_ids),
                coords=coords_pca,
                meta=common_meta,
            )
            umap_coords_path = save_coords2d(
                cache_dir=coords_cache_dir,
                dataset_name=dataset_tag,
                style_name=style_name,
                model=model_name,
                place=place,
                strength=s_int,
                method="umap",
                prompt_ids=(harmless_ids + harmful_ids),
                coords=coords_umap,
                meta=common_meta,
            )

            # 6) Plot
            print("[BUCKET] Saving 2D scatter plots")
            tsne_plot_path = os.path.join(run_dir, "plots_tsne", model_name, f"tsne_{place}_s{s_int}.png")
            pca_plot_path = os.path.join(run_dir, "plots_pca", model_name, f"pca_{place}_s{s_int}.png")
            umap_plot_path = os.path.join(run_dir, "plots_umap", model_name, f"umap_{place}_s{s_int}.png")

            plot_2d_scatter_two_clusters(
                coords=coords_tsne,
                labels=y,
                out_path_png=tsne_plot_path,
                title=f"t-SNE of last-layer activations (place={place}, strength={s_int})",
                xlabel="t-SNE 1",
                ylabel="t-SNE 2",
                legend_labels=("Harmless (Alpaca)", "Harmful (HarmBench)"),
            )
            plot_2d_scatter_two_clusters(
                coords=coords_pca,
                labels=y,
                out_path_png=pca_plot_path,
                title=f"PCA of last-layer activations (place={place}, strength={s_int})",
                xlabel="PCA 1",
                ylabel="PCA 2",
                legend_labels=("Harmless (Alpaca)", "Harmful (HarmBench)"),
            )
            plot_2d_scatter_two_clusters(
                coords=coords_umap,
                labels=y,
                out_path_png=umap_plot_path,
                title=f"UMAP of last-layer activations (place={place}, strength={s_int})",
                xlabel="UMAP 1",
                ylabel="UMAP 2",
                legend_labels=("Harmless (Alpaca)", "Harmful (HarmBench)"),
            )

            # 7) ASR (COMPLIANCE-based)
            asr = np.nan
            judged_n = 0
            complied_n = 0
            judge_model_id = None

            # --------------------------------------------------
            # ASR (Llama-Guard-3) — unsafe-based definition
            # --------------------------------------------------
            if compute_asr:
                print("[BUCKET] Generating model responses for harmful prompts (ASR)")
                harmful_outputs = generate_response(
                    model,
                    tokenizer,
                    prompts=harmful_styled,
                    max_new_tokens=int(gen_max_new_tokens),
                    batch_size=int(batch_size),
                )

                print("[BUCKET] Judging responses with Llama-Guard-3")



                asr_pack = compute_asr_with_llama_guard3(
                    guard=guard,
                    harmful_prompts=harmful_styled,
                    model_responses=harmful_outputs,
                    batch_size=int(batch_size),
                    show_progress=True,
                )

                # 1 = unsafe (attack success)
                asr = float(asr_pack["asr"])
                judged_n = int(asr_pack["judged_n"])
                unsafe_n = int(asr_pack["unsafe_n"])
                judge_model_id = asr_pack.get("judge_model_id", None)

                print(f"[BUCKET] ASR={asr:.4f}  unsafe={unsafe_n}/{judged_n}")

                # Save per-example judgments
                print("[BUCKET] Saving ASR judgments CSV")
                per_ex_dir = os.path.join(run_dir, "asr_judgements", model_name)
                os.makedirs(per_ex_dir, exist_ok=True)
                per_ex_path = os.path.join(per_ex_dir, f"asr_{place}_s{s_int}.csv")

                df_asr = pd.DataFrame({
                    "model": model_name,
                    "place": place,
                    "strength": s_int,
                    "prompt_id": list(range(len(harmful_styled))),
                    "harmful_prompt": harmful_styled,
                    "model_response": harmful_outputs,
                    "unsafe_score": asr_pack["scores"],          # 1=unsafe, 0=safe
                    "unsafe_category": asr_pack["unsafe_categories"],
                    "unsafe_code": asr_pack["unsafe_codes"],
                    "judge_raw": asr_pack["raw_outputs"],
                })

                df_asr.to_csv(per_ex_path, index=False)
                print(f"[BUCKET] ✓ Saved: {per_ex_path}")
            summary_rows.append({
                "model": model_name,
                "place": place,
                "strength": s_int,
                "n_alpaca": int(n0),
                "n_harmbench": int(n1),
                "silhouette_cosine": float(sil) if np.isfinite(sil) else np.nan,
                "asr_compliance": float(asr) if np.isfinite(asr) else np.nan,
                "asr_judged_n": int(judged_n),
                "asr_complied_n": int(complied_n),
                "asr_judge_model": judge_model_id,
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
    print(f"\n[MAIN] ✓ Saved summary: {out_sum}")

    # Line plots
    metrics_dir = os.path.join(run_dir, "plots_metrics", model_name)
    os.makedirs(metrics_dir, exist_ok=True)

    out_sil = os.path.join(metrics_dir, "silhouette_vs_strength_by_place.png")
    plot_silhouette_vs_strength(
        df_summary=df_summary,
        out_path_png=out_sil,
        group_by=["place"],
        title="Silhouette score vs strength by prompt position",
        legend_outside=True,
    )
    print(f"[MAIN] ✓ Saved plot: {out_sil}")

    if compute_asr and "asr_compliance" in df_summary.columns:
        out_asr = os.path.join(metrics_dir, "asr_vs_strength_by_place.png")
        plot_metric_vs_strength(
            df_summary=df_summary,
            metric_col="asr_compliance",
            out_path_png=out_asr,
            group_by=["place"],
            title="Attack success rate vs strength by prompt position",
            ylabel="Attack success rate",
            legend_outside=True,
        )
        print(f"[MAIN] ✓ Saved plot: {out_asr}")

    return df_summary


def run_experiment(
        *,
        model: str,
        alpaca_sample_size: int,
        harmbench_sample_size: int,
        harmbench_config: str,
        places: List[str],
        strengths: List[int],
        batch_size: int,
        seed: int,
        coords_cache_dir: str,
        compute_asr: bool,
        guard_max_new_tokens: int,
        gen_max_new_tokens: int,
) -> str:
    config = load_config()

    if model not in config.get("models", {}):
        raise ValueError(f"Unknown model '{model}'. Available: {list(config.get('models', {}).keys())}")

    device_map = config["defaults"].get("device_map", "auto")
    dtype = config["defaults"].get("dtype", "float32")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    run_dir = os.path.join(base_results_dir, "politeness_safety", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*90}")
    print("Politeness safety axes experiment")
    print(f"{'='*90}")
    print(f"Model: {model}")
    print(f"Alpaca sample size: {alpaca_sample_size}")
    print(f"HarmBench sample size: {harmbench_sample_size} (config={harmbench_config})")
    print(f"Places: {places}")
    print(f"Strengths: {strengths}")
    print(f"Batch size: {batch_size}")
    print(f"Seed: {seed}")
    print(f"Coords cache dir: {coords_cache_dir}")
    print(f"Compute ASR: {compute_asr}")
    if compute_asr:
        print("ASR judge: meta-llama/Llama-Guard-3-8B (COMPLIED/REFUSED)")
    print(f"Output dir: {run_dir}")
    print(f"{'='*90}\n")

    print("[DATA] Loading Alpaca")
    alpaca_items = load_dataset_by_name("alpaca", sample_size=int(alpaca_sample_size), seed=int(seed))
    print("[DATA] ✓ Alpaca loaded")

    print("[DATA] Loading HarmBench")
    harmbench_items = load_dataset_by_name(
        "harmbench",
        sample_size=int(harmbench_sample_size),
        seed=int(seed),
        config_name=str(harmbench_config),
    )
    print("[DATA] ✓ HarmBench loaded")

    model_path = config["models"][model]

    df_summary = run_for_one_model(
        model_name=model,
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
        compute_asr=bool(compute_asr),
        guard_max_new_tokens=int(guard_max_new_tokens),
        gen_max_new_tokens=int(gen_max_new_tokens),
    )

    out_all = os.path.join(run_dir, "politeness_safety_summary.csv")
    df_summary.to_csv(out_all, index=False)
    print(f"[MAIN] ✓ Saved final summary: {out_all}")

    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Politeness safety axes: clustering (Alpaca vs HarmBench) and ASR (compliance) using Llama-Guard-3."
    )

    parser.add_argument("--model", type=str, default="L3-8B", help="Model key from config.yaml")

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
        help="Default: ../data/embeddings_2d_cache"
    )

    parser.add_argument("--compute_asr", action="store_true", help="Compute compliance-based ASR.")
    parser.add_argument("--guard_max_new_tokens", type=int, default=24)
    parser.add_argument("--gen_max_new_tokens", type=int, default=128)

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
        model=str(args.model),
        alpaca_sample_size=int(args.alpaca_sample_size),
        harmbench_sample_size=int(args.harmbench_sample_size),
        harmbench_config=str(args.harmbench_config),
        places=args.places,
        strengths=strengths,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        coords_cache_dir=str(coords_cache_dir),
        compute_asr=bool(args.compute_asr),
        guard_max_new_tokens=int(args.guard_max_new_tokens),
        gen_max_new_tokens=int(args.gen_max_new_tokens),
    )


if __name__ == "__main__":
    main()