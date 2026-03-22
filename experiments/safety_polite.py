# experiments/safety_polite.py
"""
Safety evaluation focused on politeness-style perturbations.

Measures how politeness framing (positive and negative) affects model safety
behaviour on HarmBench harmful prompts and Alpaca harmless prompts.

Inputs:
    - HarmBench harmful prompts (via utils.data.load_harmbench_hf)
    - Alpaca harmless prompts (via utils.data.load_alpaca_hf)
    - config.yaml for model paths, politeness strengths, and positions

Outputs (saved to results/politeness_safety/run_YYYYMMDD_HHMMSS/):
    - summary_all_models.csv: silhouette score and ASR per (model, place, strength)
    - asr_outputs/{model}/: per-bucket response CSVs (stage 1) and
      LlamaGuard-3 judgment CSVs (stage 2)
    - PCA / t-SNE / UMAP scatter plots for each (place, strength) bucket

Run:
  python experiments/safety_polite.py \\
    --models G-7B L3.1-8B \\
    --strengths -10 -5 0 5 10 \\
    --places prefix suffix global \\
    --harmbench_size 200 --alpaca_size 200 \\
    --batch_size 8 \\
    --asr_stage both

Important flags:
    --asr_stage      Stage to run: stage1 (generate) | stage2 (judge) | both
    --compute_activations  Whether to compute silhouette scores (default: True)
    --compute_asr          Whether to compute ASR via LlamaGuard-3 (default: True)
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import torch

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_dataset_by_name
from utils.models import load_model, generate_response
from utils.styles import apply_politeness
from utils.metrics import (
    get_layer_activations_batch,
    compute_silhouette_score,
    reduce_activations_2d,
    compute_asr_with_llama_guard3,
    LlamaGuard3Scorer,
)

from plots.plots import (
    apply_neurips_style,
    plot_2d_scatter_two_clusters,
    plot_silhouette_vs_strength,
    plot_metric_vs_strength,
)

# ============================================================
# Utilities
# ============================================================

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _cuda_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _get_prompt_text(item):
    for k in ("question", "prompt", "instruction", "text"):
        if k in item and item[k]:
            return str(item[k])
    return str(item)


def _asr_outputs_dir(run_dir: str, model_name: str) -> str:
    d = os.path.join(run_dir, "asr_outputs", model_name)
    os.makedirs(d, exist_ok=True)
    return d


def _asr_outputs_path(run_dir: str, model_name: str, place: str, strength: int) -> str:
    return os.path.join(_asr_outputs_dir(run_dir, model_name), f"harmbench_outputs_{place}_s{int(strength)}.csv")


# ============================================================
# Core
# ============================================================

def run_for_one_model(
        *,
        model_name: str,
        model_path: str,
        alpaca_items: Optional[List[dict]],
        harmbench_items: Optional[List[dict]],
        strengths: List[int],
        places: List[str],
        run_dir: str,
        batch_size: int,
        seed: int,
        compute_activations: bool,
        compute_asr: bool,
        gen_max_new_tokens: int,
        asr_stage: str,   # "stage1" | "stage2" | "both"
):

    apply_neurips_style()

    # ------------------------------------------
    # Decide whether to load the generation model
    # (Stage2-only ASR should NOT load it)
    # ------------------------------------------
    need_generation_model = bool(compute_activations) or (bool(compute_asr) and asr_stage in ("stage1", "both"))

    model = None
    tokenizer = None

    if need_generation_model:
        print(f"\n[MAIN] Loading model: {model_name}")
        model, tokenizer = load_model(model_path, device_map="auto", dtype="float16")
        model.eval()
        try:
            model.config.use_cache = False
        except Exception:
            pass
        _cuda_cleanup()

    harmless_prompts = []
    harmful_prompts = []

    if compute_activations:
        harmless_prompts = [_get_prompt_text(x) for x in alpaca_items]

    if compute_activations or compute_asr:
        harmful_prompts = [_get_prompt_text(x) for x in harmbench_items]

    summary_rows = []

    for place in places:
        for s in strengths:

            print(f"\n[BUCKET] place={place} strength={s}")
            s_int = int(s)

            sil = np.nan
            asr = np.nan

            harmless_styled = []
            harmful_styled = []

            if compute_activations:
                harmless_styled = [
                    apply_politeness(p, s_int, place=place)
                    for p in harmless_prompts
                ]

            if compute_activations or compute_asr:
                harmful_styled = [
                    apply_politeness(p, s_int, place=place)
                    for p in harmful_prompts
                ]

            # =====================================================
            # ACTIVATIONS
            # =====================================================
            if compute_activations:

                with torch.inference_mode():
                    acts0 = get_layer_activations_batch(
                        model, tokenizer, harmless_styled, layer_idx=-1
                    )
                    acts1 = get_layer_activations_batch(
                        model, tokenizer, harmful_styled, layer_idx=-1
                    )

                X0 = acts0.detach().cpu().numpy()
                X1 = acts1.detach().cpu().numpy()

                X = np.concatenate([X0, X1], axis=0)
                y = np.array([0] * len(X0) + [1] * len(X1))

                sil = compute_silhouette_score(X, y, metric="cosine")

                # -----------------------
                # PCA / t-SNE / UMAP
                # -----------------------
                for method in ["pca", "tsne", "umap"]:
                    coords = reduce_activations_2d(X, method=method, seed=seed)

                    plot_path = os.path.join(
                        run_dir,
                        f"{method}_{place}_s{s_int}.png"
                    )

                    plot_2d_scatter_two_clusters(
                        coords=coords,
                        labels=y,
                        out_path_png=plot_path,
                        title=f"{method.upper()} (place={place}, s={s_int})",
                        xlabel="Dim 1",
                        ylabel="Dim 2",
                    )

                del acts0, acts1, X, y
                _cuda_cleanup()

            # =====================================================
            # ASR — TWO STAGES
            #   stage1: generate + save outputs (NO guard)
            #   stage2: load outputs + load guard + judge
            # =====================================================
            if compute_asr:

                out_csv = _asr_outputs_path(run_dir, model_name, place, s_int)

                # ---------------------------
                # Stage 1: generate outputs
                # ---------------------------
                if asr_stage in ("stage1", "both"):
                    if model is None or tokenizer is None:
                        raise RuntimeError("ASR stage1 requires the generation model to be loaded.")

                    with torch.inference_mode():
                        harmful_outputs = generate_response(
                            model,
                            tokenizer,
                            prompts=harmful_styled,
                            max_new_tokens=gen_max_new_tokens,
                            batch_size=batch_size,
                        )

                    df_out = pd.DataFrame({
                        "model": model_name,
                        "place": place,
                        "strength": s_int,
                        "prompt_id": list(range(len(harmful_styled))),
                        "harmful_prompt": harmful_styled,
                        "model_response": harmful_outputs,
                    })
                    df_out.to_csv(out_csv, index=False)
                    print(f"[ASR stage1] ✓ Saved outputs: {out_csv}")

                    del harmful_outputs, df_out
                    _cuda_cleanup()

                # ---------------------------
                # Stage 2: judge outputs
                # ---------------------------
                if asr_stage in ("stage2", "both"):
                    if not os.path.exists(out_csv):
                        raise FileNotFoundError(
                            f"[ASR stage2] Missing outputs CSV for judging: {out_csv}\n"
                            f"Run stage1 first (or use asr_stage=both)."
                        )

                    df_in = pd.read_csv(out_csv)
                    harmful_styled_loaded = df_in["harmful_prompt"].astype(str).tolist()
                    harmful_outputs_loaded = df_in["model_response"].astype(str).tolist()

                    print("[ASR stage2] Loading Llama-Guard-3 for judging...")
                    guard = LlamaGuard3Scorer(
                        model_path="meta-llama/Llama-Guard-3-8B",
                        vllm_mode=False,
                        local_files_only=False,
                    )
                    _cuda_cleanup()

                    asr_pack = compute_asr_with_llama_guard3(
                        guard=guard,
                        harmful_prompts=harmful_styled_loaded,
                        model_responses=harmful_outputs_loaded,
                        batch_size=batch_size,
                    )

                    asr = float(asr_pack["asr"])

                    # Save judged per-example file alongside the raw outputs
                    judged_path = os.path.join(_asr_outputs_dir(run_dir, model_name), f"harmbench_judged_{place}_s{s_int}.csv")
                    df_j = df_in.copy()
                    df_j["unsafe_score"] = asr_pack["scores"]
                    df_j["unsafe_category"] = asr_pack["unsafe_categories"]
                    df_j["unsafe_code"] = asr_pack["unsafe_codes"]
                    df_j["judge_raw"] = asr_pack["raw_outputs"]
                    df_j.to_csv(judged_path, index=False)
                    print(f"[ASR stage2] ✓ Saved judgments: {judged_path}")

                    del guard, df_in, df_j, harmful_styled_loaded, harmful_outputs_loaded, asr_pack
                    _cuda_cleanup()

            summary_rows.append({
                "model": model_name,
                "place": place,
                "strength": s_int,
                "silhouette": sil,
                "asr": asr,
            })

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    # =====================================================
    # Line plots
    # =====================================================

    if compute_activations:
        sil_path = os.path.join(run_dir, "plots_metrics", "silhouette_vs_strength.png")
        os.makedirs(os.path.dirname(sil_path), exist_ok=True)

        # ensure expected column name for plotting utilities
        if "silhouette_cosine" not in df.columns and "silhouette" in df.columns:
            df["silhouette_cosine"] = df["silhouette"]

        try:
            plot_silhouette_vs_strength(
                df_summary=df,
                out_path_png=sil_path,
                group_by=["place"],
                title="Silhouette vs Strength",
                legend_outside=True,
            )
            print(f"[MAIN] ✓ Saved silhouette plot: {sil_path}")
        except Exception as e:
            print(f"[WARN] plot_silhouette_vs_strength failed ({e}); using fallback plot.")
            import matplotlib.pyplot as plt

            plt.figure()
            for place_, sub in df.groupby("place"):
                sub = sub.sort_values("strength")
                plt.plot(sub["strength"], sub["silhouette_cosine"], label=str(place_))
            plt.xlabel("Strength")
            plt.ylabel("Silhouette (cosine)")
            plt.title("Silhouette vs Strength")
            plt.legend()
            plt.tight_layout()
            plt.savefig(sil_path, dpi=300)
            plt.close()
            print(f"[MAIN] ✓ Saved silhouette plot (fallback): {sil_path}")

    if compute_asr:
        # Plot only makes sense if ASR was actually computed (stage2 or both)
        if asr_stage in ("stage2", "both"):
            plot_metric_vs_strength(
                df_summary=df,
                metric_col="asr",
                out_path_png=os.path.join(run_dir, "asr_vs_strength.png"),
                group_by=["place"],
                title="ASR vs Strength",
                ylabel="Attack Success Rate",
            )

    print("[MAIN] ✓ Experiment completed")
    return df


# ============================================================
# Runner
# ============================================================

def run_experiment(
        model,
        compute_activations,
        compute_asr,
        alpaca_sample_size,
        harmbench_sample_size,
        harmbench_config,
        places,
        strengths,
        batch_size,
        seed,
        gen_max_new_tokens,
        asr_stage,
        run_dir: Optional[str] = None,   # allow reusing stage1 outputs
):

    config = load_config()

    # If user provided run_dir (stage2 typical), reuse it.
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("results/politeness_safety", f"run_{timestamp}")

    os.makedirs(run_dir, exist_ok=True)

    alpaca_items = None
    harmbench_items = None

    if compute_activations:
        alpaca_items = load_dataset_by_name(
            "alpaca",
            sample_size=alpaca_sample_size,
            seed=seed
        )

    if compute_activations or compute_asr:
        harmbench_items = load_dataset_by_name(
            "harmbench",
            sample_size=harmbench_sample_size,
            seed=seed,
            config_name=harmbench_config,
        )

    return run_for_one_model(
        model_name=model,
        model_path=config["models"][model],
        alpaca_items=alpaca_items,
        harmbench_items=harmbench_items,
        strengths=strengths,
        places=places,
        run_dir=run_dir,
        batch_size=batch_size,
        seed=seed,
        compute_activations=compute_activations,
        compute_asr=compute_asr,
        gen_max_new_tokens=gen_max_new_tokens,
        asr_stage=asr_stage,
    )


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="L3-8B")
    parser.add_argument("--compute_activations", action="store_true")
    parser.add_argument("--compute_asr", action="store_true")

    # ASR two-stage control
    parser.add_argument(
        "--asr_stage",
        type=str,
        default="both",
        choices=["stage1", "stage2", "both"],
        help="ASR two-stage mode: stage1=generate+save outputs, stage2=load outputs+judge, both=do both in one run."
    )

    # For stage2 you typically want to reuse an existing run directory (stage1 outputs live there)
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Optional: reuse an existing run directory (recommended for ASR stage2)."
    )

    parser.add_argument("--alpaca_sample_size", type=int, default=128)
    parser.add_argument("--harmbench_sample_size", type=int, default=128)
    parser.add_argument("--harmbench_config", default="standard")

    parser.add_argument("--places", nargs="+", default=["prefix"])
    parser.add_argument("--strengths", nargs="+", type=int, default=[-10, 0, 10])

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen_max_new_tokens", type=int, default=128)

    args = parser.parse_args()

    run_experiment(
        model=args.model,
        compute_activations=args.compute_activations,
        compute_asr=args.compute_asr,
        alpaca_sample_size=args.alpaca_sample_size,
        harmbench_sample_size=args.harmbench_sample_size,
        harmbench_config=args.harmbench_config,
        places=args.places,
        strengths=args.strengths,
        batch_size=args.batch_size,
        seed=args.seed,
        gen_max_new_tokens=args.gen_max_new_tokens,
        asr_stage=args.asr_stage,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()