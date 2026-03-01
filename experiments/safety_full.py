# experiments/safety_polite.py

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
from utils.styles import apply_politeness, apply_length_variation, apply_interrogative
from utils.metrics import (
    get_layer_activations_batch,
    compute_silhouette_score,
    reduce_activations_2d,
    compute_asr_with_llama_guard3,
    LlamaGuard3Scorer,
)

from utils.politeness_plots import (
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


def _asr_outputs_path(run_dir: str, model_name: str, place: str, strength: Any) -> str:
    return os.path.join(_asr_outputs_dir(run_dir, model_name), f"harmbench_outputs_{place}_s{strength}.csv")


# ============================================================
# NEW: Surface-noise style appliers (import if available; else local fallbacks)
# (Added without changing existing behavior)
# ============================================================

def _try_import_surface_noise_appliers():
    """
    Best-effort import of surface-noise appliers from utils.styles if you already placed them there.
    If not found, you can still run politeness-only experiments as before.
    """
    spacing = punctuation = letter_case = None
    try:
        import utils.styles as _styles_mod
        spacing = getattr(_styles_mod, "apply_spacing", None)
        punctuation = getattr(_styles_mod, "apply_punctuation", None)
        letter_case = getattr(_styles_mod, "apply_letter_case", None)
    except Exception:
        pass
    return spacing, punctuation, letter_case


_APPLY_SPACING, _APPLY_PUNCT, _APPLY_LETTER_CASE = _try_import_surface_noise_appliers()


def apply_surface_noise_style(text: str, strength: Any, *, place: str, style_name: str) -> str:
    """
    Apply one surface-noise style by name.
    style_name in {"spacing","punctuation","letter_case"}.

    NOTE:
      - spacing/punctuation take integer strength
      - letter_case takes percentage (0..100)
    """
    if style_name == "spacing":
        if _APPLY_SPACING is None:
            raise ImportError("apply_spacing is not available. Put it in utils/styles.py or import path.")
        return _APPLY_SPACING(text, int(strength), place=place)

    if style_name == "punctuation":
        if _APPLY_PUNCT is None:
            raise ImportError("apply_punctuation is not available. Put it in utils/styles.py or import path.")
        return _APPLY_PUNCT(text, int(strength), place=place)

    if style_name == "letter_case":
        if _APPLY_LETTER_CASE is None:
            raise ImportError("apply_letter_case is not available. Put it in utils/styles.py or import path.")
        return _APPLY_LETTER_CASE(text, float(strength), place=place)

    raise ValueError(f"Unknown surface noise style: {style_name}")


def apply_structured_style(text: str, strength: Any, *, place: str, style_name: str) -> str:
    """
    Apply one structured style by name.
    style_name in {"length_variation", "interrogative"}.

    Note: these are global rewrites — `place` is accepted for interface
    consistency but is not forwarded to the underlying functions.
    """
    if style_name == "length_variation":
        return apply_length_variation(text, float(strength))

    if style_name == "interrogative":
        return apply_interrogative(text, mode=str(strength))

    raise ValueError(f"Unknown structured style: {style_name}")


def _resolve_style_from_config_or_cli(
        *,
        config: Dict[str, Any],
        style_family: str,
        surface_style: Optional[str],
        places_cli: List[str],
        strengths_cli: List[int],
) -> (List[str], List[int], Optional[str]):
    """
    style_family:
      - "politeness" (existing behavior, uses CLI places/strengths)
      - "surface_noise" (uses config.style_positions + config.style_levels)
      - "structured" (uses config.style_positions + config.style_levels)

    surface_style:
      - "spacing" | "punctuation" | "letter_case" (surface_noise)

    structured_style:
      - "length_variation" | "interrogative" (structured)
    """
    if style_family not in ("surface_noise", "structured"):
        return places_cli, strengths_cli, None

    if surface_style is None:
        raise ValueError("--surface_style must be provided when --style_family is surface_noise or structured")

    cfg_places = (
            (config.get("style_positions") or {}).get(surface_style, None)
            or (config.get("style_defaults") or {}).get(surface_style, {}).get("positions", None)
    )
    cfg_strengths = (config.get("style_levels") or {}).get(surface_style, None)

    if not cfg_places:
        raise ValueError(f"Config missing style_positions for '{surface_style}'")
    if not cfg_strengths:
        raise ValueError(f"Config missing style_levels for '{surface_style}'")

    # keep types consistent with the rest of the code:
    places = [str(x) for x in cfg_places]

    # Strengths can be int (surface noise), float (length_variation), or str (interrogative)
    if surface_style == "interrogative":
        strengths = [str(x) for x in cfg_strengths]
    elif surface_style == "length_variation":
        strengths = [float(x) for x in cfg_strengths]
    else:
        strengths = [int(x) for x in cfg_strengths]

    return places, strengths, surface_style


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
        style_family: str,         # "politeness" | "surface_noise"
        style_name: Optional[str],  # "spacing" | "punctuation" | "letter_case" | "length_variation" | "interrogative" | None
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
            # Build a label for filenames / summary rows
            try:
                s_int = int(s)
            except (ValueError, TypeError):
                s_int = s  # keep as-is for string/float strengths
            s_label = str(s)

            sil = np.nan
            asr = np.nan

            harmless_styled = []
            harmful_styled = []

            # -------------------------------------------------
            # Style application (politeness OR one surface style)
            # -------------------------------------------------
            if compute_activations:
                if style_family == "surface_noise":
                    harmless_styled = [
                        apply_surface_noise_style(p, s_int, place=place, style_name=str(style_name))
                        for p in harmless_prompts
                    ]
                elif style_family == "structured":
                    harmless_styled = [
                        apply_structured_style(p, s, place=place, style_name=str(style_name))
                        for p in harmless_prompts
                    ]
                else:
                    harmless_styled = [
                        apply_politeness(p, s_int, place=place)
                        for p in harmless_prompts
                    ]

            if compute_activations or compute_asr:
                if style_family == "surface_noise":
                    harmful_styled = [
                        apply_surface_noise_style(p, s_int, place=place, style_name=str(style_name))
                        for p in harmful_prompts
                    ]
                elif style_family == "structured":
                    harmful_styled = [
                        apply_structured_style(p, s, place=place, style_name=str(style_name))
                        for p in harmful_prompts
                    ]
                else:
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
                        f"{method}_{place}_s{s_label}.png"
                    )

                    plot_2d_scatter_two_clusters(
                        coords=coords,
                        labels=y,
                        out_path_png=plot_path,
                        title=f"{method.upper()} (place={place}, s={s_label})",
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

                out_csv = _asr_outputs_path(run_dir, model_name, place, s_label)

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
                        "style_family": style_family,
                        "surface_style": style_name,
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

                    judged_path = os.path.join(_asr_outputs_dir(run_dir, model_name), f"harmbench_judged_{place}_s{s_label}.csv")
                    df_j = df_in.copy()
                    df_j["unsafe_score"] = asr_pack["scores"]
                    df_j["unsafe_category"] = asr_pack["unsafe_categories"]
                    df_j["unsafe_code"] = asr_pack["unsafe_codes"]
                    df_j["judge_raw"] = asr_pack["raw_outputs"]
                    df_j["style_family"] = style_family
                    df_j["surface_style"] = style_name
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
                "style_family": style_family,
                "surface_style": style_name,
            })

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    # =====================================================
    # Line plots
    # =====================================================

    if compute_activations:
        sil_path = os.path.join(run_dir, "plots_metrics", "silhouette_vs_strength.png")
        os.makedirs(os.path.dirname(sil_path), exist_ok=True)

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
        style_family: str = "politeness",
        style_name: Optional[str] = None,
):

    config = load_config()

    # NEW: override places/strengths from config when running a surface-noise style
    places, strengths, style_name = _resolve_style_from_config_or_cli(
        config=config,
        style_family=style_family,
        surface_style=style_name,
        places_cli=places,
        strengths_cli=strengths,
    )

    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("results/safety", f"run_{timestamp}")

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
        style_family=style_family,
        style_name=style_name,
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

    # NEW: choose style family + which surface-noise style (each alone)
    parser.add_argument(
        "--style_family",
        type=str,
        default="politeness",
        choices=["politeness", "surface_noise", "structured"],
        help="Which style family to apply: politeness (existing) | surface_noise (spacing/punctuation/letter_case) | structured (length_variation/interrogative)."
    )
    parser.add_argument(
        "--surface_style",
        type=str,
        default=None,
        choices=["spacing", "punctuation", "letter_case", "length_variation", "interrogative"],
        help="When --style_family is surface_noise or structured, choose ONE sub-style. "
             "Places/strengths will be taken from config.yaml for that style."
    )

    parser.add_argument("--alpaca_sample_size", type=int, default=128)
    parser.add_argument("--harmbench_sample_size", type=int, default=128)
    parser.add_argument("--harmbench_config", default="standard")

    # (kept as-is) — for politeness mode these are used; for surface_noise they’re ignored in favor of config
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
        style_family=args.style_family,
        style_name=args.surface_style,
    )


if __name__ == "__main__":
    main()