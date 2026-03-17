import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE)) if os.path.basename(os.path.dirname(_THIS_FILE)) == "experiments" else os.getcwd()
sys.path.append(_REPO_ROOT)

from utils.data import load_dataset_by_name
from utils.models import load_model
from utils.styles import apply_politeness
from utils.metrics import get_layer_activations_batch, compute_silhouette_score, reduce_activations_2d
from utils.plots import apply_neurips_style


DEFAULT_MODEL = "G-7B"
DEFAULT_STRENGTH = -10
DEFAULT_PLACES = ["prefix", "suffix", "global"]
DEFAULT_ALPACA_SIZE = 128
DEFAULT_HARMBENCH_SIZE = 128
DEFAULT_HARMBENCH_CONFIG = "standard"
DEFAULT_BATCH_SIZE = 16
DEFAULT_SEED = 42


def load_config() -> Dict:
    config_path = os.path.join(_REPO_ROOT, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cuda_cleanup() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _get_prompt_text(item: dict) -> str:
    for k in ("question", "prompt", "instruction", "text"):
        if k in item and item[k]:
            return str(item[k])
    return str(item)


def _centroid(coords: np.ndarray) -> np.ndarray:
    return np.asarray(coords, dtype=np.float32).mean(axis=0)


def _save_coords_csv(
    *,
    coords: np.ndarray,
    groups: List[str],
    prompt_ids: List[int],
    out_csv: str,
    method: str,
    place: str,
    strength: int,
) -> None:
    df = pd.DataFrame(
        {
            "prompt_id": prompt_ids,
            "group": groups,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "method": method,
            "place": place,
            "strength": strength,
        }
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)


def plot_baseline_vs_negative_politeness(
    *,
    coords: np.ndarray,
    counts: Dict[str, int],
    out_path_png: str,
    title: str,
) -> None:
    """
    Plot four groups in one 2D projection:
      - harmless, baseline  (blue)
      - harmless, s=-10     (green)
      - harmful, baseline   (red)
      - harmful, s=-10      (orange)

    Also draws arrows from each baseline centroid to its perturbed centroid.
    """
    apply_neurips_style()

    n_harmless_base = counts["harmless_baseline"]
    n_harmful_base = counts["harmful_baseline"]
    n_harmless_neg = counts["harmless_neg10"]
    n_harmful_neg = counts["harmful_neg10"]

    i0 = 0
    i1 = i0 + n_harmless_base
    i2 = i1 + n_harmful_base
    i3 = i2 + n_harmless_neg
    i4 = i3 + n_harmful_neg

    harmless_base = coords[i0:i1]
    harmful_base = coords[i1:i2]
    harmless_neg = coords[i2:i3]
    harmful_neg = coords[i3:i4]

    c_harm_base = "red"
    c_harm_neg = "orange"
    c_harmless_base = "royalblue"
    c_harmless_neg = "green"

    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    ax.scatter(harmful_base[:, 0], harmful_base[:, 1], s=16, alpha=0.65, color=c_harm_base,
               edgecolors="none", label="harmful, baseline", rasterized=True)
    ax.scatter(harmful_neg[:, 0], harmful_neg[:, 1], s=16, alpha=0.65, color=c_harm_neg,
               edgecolors="none", label="harmful, s=-10", rasterized=True)
    ax.scatter(harmless_base[:, 0], harmless_base[:, 1], s=16, alpha=0.65, color=c_harmless_base,
               edgecolors="none", label="harmless, baseline", rasterized=True)
    ax.scatter(harmless_neg[:, 0], harmless_neg[:, 1], s=16, alpha=0.65, color=c_harmless_neg,
               edgecolors="none", label="harmless, s=-10", rasterized=True)

    harmless_base_ctr = _centroid(harmless_base)
    harmless_neg_ctr = _centroid(harmless_neg)
    harmful_base_ctr = _centroid(harmful_base)
    harmful_neg_ctr = _centroid(harmful_neg)

    ax.annotate(
        "",
        xy=harmful_neg_ctr,
        xytext=harmful_base_ctr,
        arrowprops=dict(arrowstyle="->", lw=1.8, color="black"),
        zorder=5,
    )
    ax.annotate(
        "",
        xy=harmless_neg_ctr,
        xytext=harmless_base_ctr,
        arrowprops=dict(arrowstyle="->", lw=1.8, color="black"),
        zorder=5,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="w", markerfacecolor=c_harm_base,
               markeredgecolor="none", markersize=6, label="harmful, baseline"),
        Line2D([0], [0], marker="o", linestyle="", color="w", markerfacecolor=c_harm_neg,
               markeredgecolor="none", markersize=6, label="harmful, s=-10"),
        Line2D([0], [0], marker="o", linestyle="", color="w", markerfacecolor=c_harmless_base,
               markeredgecolor="none", markersize=6, label="harmless, baseline"),
        Line2D([0], [0], marker="o", linestyle="", color="w", markerfacecolor=c_harmless_neg,
               markeredgecolor="none", markersize=6, label="harmless, s=-10"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="best")

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_axisbelow(True)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path_png), exist_ok=True)
    fig.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_group_activations(
    *,
    model,
    tokenizer,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    place: str,
    strength: int,
) -> Dict[str, np.ndarray]:
    harmless_neg = [apply_politeness(p, strength, place=place) for p in harmless_prompts]
    harmful_neg = [apply_politeness(p, strength, place=place) for p in harmful_prompts]

    with torch.inference_mode():
        acts_harmless_base = get_layer_activations_batch(model, tokenizer, harmless_prompts, layer_idx=-1)
        acts_harmful_base = get_layer_activations_batch(model, tokenizer, harmful_prompts, layer_idx=-1)
        acts_harmless_neg = get_layer_activations_batch(model, tokenizer, harmless_neg, layer_idx=-1)
        acts_harmful_neg = get_layer_activations_batch(model, tokenizer, harmful_neg, layer_idx=-1)

    return {
        "harmless_baseline": acts_harmless_base.detach().cpu().numpy(),
        "harmful_baseline": acts_harmful_base.detach().cpu().numpy(),
        "harmless_neg10": acts_harmless_neg.detach().cpu().numpy(),
        "harmful_neg10": acts_harmful_neg.detach().cpu().numpy(),
    }


def run_experiment(
    *,
    model_name: str,
    strength: int,
    places: List[str],
    alpaca_sample_size: int,
    harmbench_sample_size: int,
    harmbench_config: str,
    batch_size: int,
    seed: int,
    run_dir: str | None,
) -> str:
    if strength != -10:
        raise ValueError("This script is intended for the negative politeness setting only; use --strength -10.")

    config = load_config()
    if model_name not in config["models"]:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(config['models'].keys())}")

    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("results", "politeness_safety_activation_g7b_neg10", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots_shift"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "coords"), exist_ok=True)

    harmless_items = load_dataset_by_name("alpaca", sample_size=alpaca_sample_size, seed=seed)
    harmful_items = load_dataset_by_name(
        "harmbench",
        sample_size=harmbench_sample_size,
        seed=seed,
        config_name=harmbench_config,
    )

    harmless_prompts = [_get_prompt_text(x) for x in harmless_items]
    harmful_prompts = [_get_prompt_text(x) for x in harmful_items]

    print(f"[MAIN] Loading model: {model_name}")
    model, tokenizer = load_model(config["models"][model_name], dtype="float16")
    model.eval()
    try:
        model.config.use_cache = False
    except Exception:
        pass
    _cuda_cleanup()

    summary_rows = []

    for place in places:
        print(f"[BUCKET] place={place} strength={strength}")
        acts = compute_group_activations(
            model=model,
            tokenizer=tokenizer,
            harmful_prompts=harmful_prompts,
            harmless_prompts=harmless_prompts,
            place=place,
            strength=strength,
        )

        X_baseline = np.concatenate([acts["harmless_baseline"], acts["harmful_baseline"]], axis=0)
        y_baseline = np.array([0] * len(acts["harmless_baseline"]) + [1] * len(acts["harmful_baseline"]))
        X_neg = np.concatenate([acts["harmless_neg10"], acts["harmful_neg10"]], axis=0)
        y_neg = np.array([0] * len(acts["harmless_neg10"]) + [1] * len(acts["harmful_neg10"]))

        sil_baseline = float(compute_silhouette_score(X_baseline, y_baseline, metric="cosine"))
        sil_neg = float(compute_silhouette_score(X_neg, y_neg, metric="cosine"))

        X_all = np.concatenate(
            [
                acts["harmless_baseline"],
                acts["harmful_baseline"],
                acts["harmless_neg10"],
                acts["harmful_neg10"],
            ],
            axis=0,
        )

        counts = {
            "harmless_baseline": len(acts["harmless_baseline"]),
            "harmful_baseline": len(acts["harmful_baseline"]),
            "harmless_neg10": len(acts["harmless_neg10"]),
            "harmful_neg10": len(acts["harmful_neg10"]),
        }
        prompt_ids = (
            list(range(counts["harmless_baseline"]))
            + list(range(counts["harmful_baseline"]))
            + list(range(counts["harmless_neg10"]))
            + list(range(counts["harmful_neg10"]))
        )
        groups = (
            ["harmless_baseline"] * counts["harmless_baseline"]
            + ["harmful_baseline"] * counts["harmful_baseline"]
            + ["harmless_neg10"] * counts["harmless_neg10"]
            + ["harmful_neg10"] * counts["harmful_neg10"]
        )

        for method in ("pca", "umap"):
            print(f"[PLOT] Reducing activations with {method.upper()} for place={place}")
            coords = reduce_activations_2d(X_all, method=method, seed=seed)

            coords_csv = os.path.join(run_dir, "coords", f"{method}_{place}_s{strength}.csv")
            _save_coords_csv(
                coords=coords,
                groups=groups,
                prompt_ids=prompt_ids,
                out_csv=coords_csv,
                method=method,
                place=place,
                strength=strength,
            )

            title = f"{method.upper()} representation of baseline vs. negative politeness prompts ({place})"
            out_png = os.path.join(run_dir, "plots_shift", f"{method}_{place}_baseline_vs_s{strength}.png")
            plot_baseline_vs_negative_politeness(
                coords=coords,
                counts=counts,
                out_path_png=out_png,
                title=title,
            )

        summary_rows.append(
            {
                "model": model_name,
                "place": place,
                "strength": strength,
                "harmless_n": counts["harmless_baseline"],
                "harmful_n": counts["harmful_baseline"],
                "silhouette_baseline": sil_baseline,
                "silhouette_neg10": sil_neg,
                "silhouette_delta": sil_neg - sil_baseline,
            }
        )

        del acts, X_baseline, y_baseline, X_neg, y_neg, X_all
        _cuda_cleanup()

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(run_dir, "summary.csv")
    df_summary.to_csv(summary_path, index=False)

    print(f"[MAIN] ✓ Saved summary: {summary_path}")
    print(f"[MAIN] ✓ Run directory: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the politeness safety activation experiment for G-7B at strength -10 and generate baseline-vs-perturbed PCA/UMAP shift plots."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--strength", type=int, default=DEFAULT_STRENGTH)
    parser.add_argument("--places", nargs="+", default=DEFAULT_PLACES)
    parser.add_argument("--alpaca_sample_size", type=int, default=DEFAULT_ALPACA_SIZE)
    parser.add_argument("--harmbench_sample_size", type=int, default=DEFAULT_HARMBENCH_SIZE)
    parser.add_argument("--harmbench_config", default=DEFAULT_HARMBENCH_CONFIG)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--run_dir", default=None)
    args = parser.parse_args()

    if args.model != DEFAULT_MODEL:
        raise ValueError(f"This script is restricted to {DEFAULT_MODEL}. Got: {args.model}")
    if args.strength != DEFAULT_STRENGTH:
        raise ValueError(f"This script is restricted to strength {DEFAULT_STRENGTH}. Got: {args.strength}")

    run_experiment(
        model_name=args.model,
        strength=args.strength,
        places=args.places,
        alpaca_sample_size=args.alpaca_sample_size,
        harmbench_sample_size=args.harmbench_sample_size,
        harmbench_config=args.harmbench_config,
        batch_size=args.batch_size,
        seed=args.seed,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
