"""
2D activation-space visualisation for politeness-induced representation drift.

Projects last-layer activations of harmful prompts (styled and unstyled) into a
2D coordinate space defined by an SVM decision boundary (harmless vs. harmful)
and a 1D PCA axis fitted on harmful baseline activations. Arrows show how
politeness strength shifts the cluster centroids across the boundary.

Also computes silhouette scores for each (strength, place) bucket, matching the
logic in safety_polite.py.

Inputs:
    - HarmBench harmful prompts (via utils.data.load_dataset_by_name)
    - Alpaca harmless prompts (for the SVM classifier)
    - config.yaml for model paths

Outputs (saved to results/politeness_activation_drift/run_YYYYMMDD_HHMMSS/):
    - {method}_harmful_politeness_drift_{place}.png/.pdf
    - summary.csv: silhouette score per (model, place, strength)

Run:
    python plots/plot_2d_activation_safety_politeness.py \\
        --model L3.2-3B \\
        --strengths -10 0 10 \\
        --place global \\
        --method pca

Important flags:
    --model      Model key from config.yaml
    --strengths  List of strength values; must include 0 (the unperturbed baseline)
    --method     Dimensionality-reduction method: pca (default) or umap
    --place      Style injection position: global | prefix | suffix
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = (
    os.path.dirname(os.path.dirname(_THIS_FILE))
    if os.path.basename(os.path.dirname(_THIS_FILE)) == "experiments"
    else os.getcwd()
)
sys.path.append(_REPO_ROOT)

from utils.data import load_dataset_by_name
from utils.models import load_model
from utils.styles import apply_politeness
from utils.metrics import get_layer_activations_batch, compute_silhouette_score, reduce_activations_2d
from plots.plots import apply_neurips_style


DEFAULT_MODEL           = "L3.2-3B"
DEFAULT_PLACE           = "global"
DEFAULT_STRENGTHS       = [0, -10,10]
DEFAULT_ALPACA_SIZE     = 128
DEFAULT_HARMBENCH_SIZE  = 128
DEFAULT_HARMBENCH_CONFIG = "standard"
DEFAULT_BATCH_SIZE      = 20
DEFAULT_SEED            = 42
DEFAULT_METHOD          = "pca"


def load_config() -> Dict:
    """Load the project-level config.yaml from the repository root."""
    config_path = os.path.join(_REPO_ROOT, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cuda_cleanup() -> None:
    """Free GPU memory cache and release inter-process CUDA resources."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _get_prompt_text(item: dict) -> str:
    """Extract the prompt string from a dataset item dict, trying common field names."""
    for k in ("question", "prompt", "instruction", "text"):
        if k in item and item[k]:
            return str(item[k])
    return str(item)


def _centroid(coords: np.ndarray) -> np.ndarray:
    """Return the mean of a set of 2D coordinates."""
    return np.asarray(coords, dtype=np.float64).mean(axis=0)


def _fit_transform_baseline(X_base: np.ndarray, method: str, seed: int):
    """
    Standardise and fit_transform the baseline activations — identical to
    what safety_polite.py does via reduce_activations_2d(X_base, method, seed).
    Returns (reducer, mu, sd, coords_base) so perturbed sets can be projected
    with .transform() into the exact same coordinate space.
    """
    mu    = X_base.mean(axis=0, keepdims=True)
    sd    = X_base.std(axis=0, keepdims=True) + 1e-8
    X_std = (X_base - mu) / sd

    if method == "pca":
        from sklearn.decomposition import PCA
        reducer     = PCA(n_components=2, random_state=seed)
        coords_base = reducer.fit_transform(X_std)          # same as safety_polite.py

    elif method == "umap":
        try:
            import umap as umap_lib
        except ImportError as e:
            raise ImportError("Install umap-learn: pip install umap-learn") from e
        n_neighbors = max(2, min(15, len(X_std) - 1))
        reducer = umap_lib.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=seed,
        )
        coords_base = reducer.fit_transform(X_std)          # same as safety_polite.py

    else:
        raise ValueError(f"Unknown method for fit+transform: {method}")

    return reducer, mu, sd, np.asarray(coords_base, dtype=np.float32)


def _project(acts: np.ndarray, reducer, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Project perturbed activations into the baseline-fitted coordinate space."""
    X_std = (acts - mu) / sd
    return np.asarray(reducer.transform(X_std), dtype=np.float32)


def _get_activations_chunked(
        model, tokenizer, prompts: List[str], batch_size: int
) -> np.ndarray:
    """Run forward passes in mini-batches; return CPU numpy [N, H]."""
    parts = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        with torch.inference_mode():
            acts = get_layer_activations_batch(model, tokenizer, chunk, layer_idx=-1)
        parts.append(acts.detach().cpu().numpy())
        del acts
        _cuda_cleanup()
    return np.concatenate(parts, axis=0)


def plot_harmful_politeness_drift(
        *,
        coords_harmless_base: np.ndarray,
        coords_harmful_base:  np.ndarray,
        coords_by_strength:   Dict[int, np.ndarray],
        acts_harmful_base:    np.ndarray,
        acts_by_strength:     Dict[int, np.ndarray],
        strengths:            List[int],
        out_path_png:         str,
        method:               str,
        place:                str,
        model_name:           str = "",
        seed:                 int = 42,
) -> None:
    """
    Axes:
      x = SVM decision score from a LinearSVC trained on 2D-projected baseline
          activations (harmless vs harmful).
      y = first principal component (1D PCA) of the high-dim activation vectors,
          fitted on harmful baseline activations only.

    Background: smooth gradient from royalblue (x < 0, harmless region) to
    lightcoral (x > 0, harmful region) via tanh(x * 0.5), with a dashed
    vertical boundary at x = 0.

    Foreground: harmful prompts at each strength, s=0 in red, others from a
    palette that avoids blue and red.  Arrows radiate from the s=0 centroid.
    """
    from sklearn.svm import LinearSVC
    from sklearn.decomposition import PCA as PCA1D

    apply_neurips_style()

    # ── Train LinearSVC on 2D projected baseline (harmless vs harmful) ──────
    X_cls = np.vstack([coords_harmless_base, coords_harmful_base])
    y_cls = np.array(
        [0] * len(coords_harmless_base) + [1] * len(coords_harmful_base)
    )
    clf = LinearSVC(max_iter=5000, C=1.0)
    clf.fit(X_cls, y_cls)

    # ── Fit 1D PCA on standardised high-dim harmful baseline activations ────
    mu_hd = acts_harmful_base.mean(axis=0, keepdims=True)
    sd_hd = acts_harmful_base.std(axis=0, keepdims=True) + 1e-8
    pca1d = PCA1D(n_components=1, random_state=seed)
    pca1d.fit((acts_harmful_base - mu_hd) / sd_hd)

    # ── Palette — baseline red; others skip blue/red ─────────────────────
    _OTHER_COLORS = [
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # grey
        "#bcbd22",  # olive
        "#17becf",  # teal
    ]
    _non_baseline = [_s for _s in strengths if _s != 0]
    point_styles = {
        0: ("red", "Harmful", 60, 0.85)
    }
    for _i, _s in enumerate(_non_baseline):
        if _s > 0:
            _label = f"Harmful + Polite"
        else:
            _label = f"Harmful + Rude"
        point_styles[_s] = (
            _OTHER_COLORS[_i % len(_OTHER_COLORS)],
            _label,
            60, 0.85,
        )

    # ── Compute (x, y) per strength ──────────────────────────────────────
    xy_by_strength: Dict[int, tuple] = {}
    for s in strengths:
        x_s = -clf.decision_function(coords_by_strength[s])
        acts_std = (acts_by_strength[s] - mu_hd) / sd_hd
        y_s = pca1d.transform(acts_std)[:, 0]
        xy_by_strength[s] = (x_s, y_s)

    # ── Axis extents ──────────────────────────────────────────────────────
    all_x = np.concatenate([xy_by_strength[s][0] for s in strengths])
    all_y = np.concatenate([xy_by_strength[s][1] for s in strengths])
    x_margin = (all_x.max() - all_x.min()) * 0.35 + 1.2
    y_margin = (all_y.max() - all_y.min()) * 0.35 + 1.2
    x_min, x_max = float(all_x.min() - x_margin), float(all_x.max() + x_margin)
    y_min, y_max = float(all_y.min() - y_margin), float(all_y.max() + y_margin)

    # ── Figure ────────────────────────────────────────────────────────────
    _x_pad = (x_max - x_min) * -0.05
    _y_pad = (y_max - y_min) * -0.05
    x_min, x_max = x_min - _x_pad, x_max + _x_pad
    y_min, y_max = y_min - _y_pad, y_max + _y_pad

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Flat background: salmon/pink left of x=0 (refusal), lavender right of x=0 (compliance)
    ax.axvspan(x_min, 0,     alpha=0.45, color="#f07070",  zorder=0)   # deeper pink / salmon
    ax.axvspan(0,     x_max, alpha=0.45, color="#7b9fd4",  zorder=0)   # deeper steel blue / lavender

    # Dashed boundary at decision score = 0
    ax.axvline(x=0, color="#333333", linewidth=2.0, linestyle="--", zorder=2)

    # ── Scatter points ────────────────────────────────────────────────────
    centroids = {}
    for s in strengths:
        x_s, y_s = xy_by_strength[s]
        color, label, size, alpha = point_styles[s]
        ax.scatter(
            x_s, y_s,
            s=size, c=color, alpha=alpha,
            edgecolors="none", label=label,
            rasterized=True, zorder=3,
        )
        centroids[s] = np.array([float(x_s.mean()), float(y_s.mean())])

    # ── Arrows: s=0 centroid → each other strength's centroid ────────────
    for s_to in [_s for _s in strengths if _s != 0]:
        ax.annotate(
            "",
            xy=centroids[s_to],
            xytext=centroids[0],
            arrowprops=dict(
                arrowstyle="-|>",
                lw=2.0,
                color="black",
                mutation_scale=14,
            ),
            zorder=5,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)
    fig.tight_layout()
    # Lock axes position, then add legend and labels so they don't affect the frame
    ax.set_position(ax.get_position())

    # ── Region labels directly on plot ────────────────────────────────────
    ax.text(
        (x_min + 0) / 2, y_max - (y_max - y_min) * 0.05,
        "Refusal Subspace",
        ha="center", va="top", fontsize=20, fontweight="bold",
        color="#7a1a1a", alpha=0.85, zorder=4,
    )
    ax.text(
        (0 + x_max) / 2, y_max - (y_max - y_min) * 0.05,
        "Compliance Subspace",
        ha="center", va="top", fontsize=20, fontweight="bold",
        color="#1a3a6a", alpha=0.85, zorder=4,
    )

    # ── Legend (scatter points only) ──────────────────────────────────────
    from matplotlib.patches import Patch
    _scatter_handles, _ = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles=_scatter_handles,
        frameon=True,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        bbox_transform=ax.transAxes,
        fontsize=20,
        ncol=3,
        columnspacing=0.8,
        handlelength=1.0,
        markerscale=1.5,
        labelspacing=0.3,
        borderpad=0.5,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.97)
    leg.get_frame().set_edgecolor("none")

    os.makedirs(os.path.dirname(out_path_png), exist_ok=True)
    fig.savefig(out_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_path_png.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  [SAVE] {out_path_png}")


def run_experiment(
        *,
        model_name:           str,
        place:                str,
        strengths:            List[int],
        alpaca_sample_size:   int,
        harmbench_sample_size: int,
        harmbench_config:     str,
        batch_size:           int,
        seed:                 int,
        method:               str,
        run_dir:              str | None,
) -> str:
    """
    Run the full activation-drift experiment for one model.

    Steps:
      1. Load Alpaca (harmless) and HarmBench (harmful) prompts.
      2. Collect last-layer activations for baseline and each styled strength.
      3. Fit a 2D projection (PCA or UMAP) on the harmful activations.
      4. Train a LinearSVC on the 2D-projected baseline (harmless vs. harmful).
      5. Plot how harmful centroids drift across the decision boundary as
         politeness strength increases.
      6. Compute and save silhouette scores to summary.csv.

    Returns:
        Path to the run directory with results.
    """
    config = load_config()
    if model_name not in config["models"]:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(config['models'].keys())}")

    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(
            "results", "politeness_activation_drift", f"run_{timestamp}"
        )
    os.makedirs(run_dir, exist_ok=True)

    # ── Load datasets ──────────────────────────────────────────────────────
    harmless_items = load_dataset_by_name("alpaca", sample_size=alpaca_sample_size, seed=seed)
    harmful_items  = load_dataset_by_name(
        "harmbench",
        sample_size=harmbench_sample_size,
        seed=seed,
        config_name=harmbench_config,
    )
    harmless_prompts = [_get_prompt_text(x) for x in harmless_items]
    harmful_prompts  = [_get_prompt_text(x) for x in harmful_items]

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"[MAIN] Loading model: {model_name}")
    model, tokenizer = load_model(config["models"][model_name], dtype="float16")
    model.eval()
    try:
        model.config.use_cache = False
    except Exception:
        pass
    _cuda_cleanup()

    # ── Get activations for background classifier ──────────────────────────
    print("[ACT] harmless baseline …")
    acts_harmless_base = _get_activations_chunked(model, tokenizer, harmless_prompts, batch_size)
    print("[ACT] harmful baseline …")
    acts_harmful_base  = _get_activations_chunked(model, tokenizer, harmful_prompts,  batch_size)

    # ── Get activations for harmful prompts at each strength ───────────────
    acts_harmful_by_strength: Dict[int, np.ndarray] = {}
    for s in strengths:
        if s == 0:
            acts_harmful_by_strength[s] = acts_harmful_base
        else:
            print(f"[ACT] harmful s={s}, place={place} …")
            styled = [apply_politeness(p, s, place=place) for p in harmful_prompts]
            acts_harmful_by_strength[s] = _get_activations_chunked(
                model, tokenizer, styled, batch_size
            )

    # ── Project activations to 2D ──────────────────────────────────────────
    print(f"[PROJ] Fitting {method.upper()} …")

    if method == "tsne":
        # t-SNE has no .transform(); embed harmful points only (same rationale
        # as the PCA/UMAP branch: avoid the harmless/harmful axis dominating).
        # Harmless baseline is embedded together so the classifier can be trained.
        X_all = np.vstack(
            [acts_harmless_base, acts_harmful_base]
            + [acts_harmful_by_strength[s] for s in strengths if s != 0]
        )
        coords_all = reduce_activations_2d(X_all, method="tsne", seed=seed)
        n_hl = len(acts_harmless_base)
        n_hm = len(acts_harmful_base)
        coords_harmless_base = coords_all[:n_hl]
        coords_harmful_base  = coords_all[n_hl : n_hl + n_hm]
        nonzero_strengths_tsne = [s for s in strengths if s != 0]
        coords_by_strength: Dict[int, np.ndarray] = {
            strengths[0]: coords_harmful_base
        }
        for i, s in enumerate(nonzero_strengths_tsne):
            lo = n_hl + n_hm + i * n_hm
            coords_by_strength[s] = coords_all[lo : lo + n_hm]
    else:
        # Fit reducer on harmful activations only (baseline + all perturbed
        # strengths).  This ensures PC1/PC2 capture the politeness-induced
        # drift rather than the harmless-vs-harmful separation axis, which
        # otherwise dominates the variance and compresses drift to higher dims.
        X_harmful_fit = np.vstack(
            [acts_harmful_base]
            + [acts_harmful_by_strength[s] for s in strengths if s != 0]
        )
        reducer, mu, sd, coords_harmful_fit = _fit_transform_baseline(
            X_harmful_fit, method=method, seed=seed
        )
        # Split out baseline coords (first len(acts_harmful_base) rows)
        coords_harmful_base = coords_harmful_fit[: len(acts_harmful_base)]
        # Project harmless baseline into the same space (for the classifier)
        coords_harmless_base = _project(acts_harmless_base, reducer, mu, sd)
        # Reconstruct per-strength coord slices from the stacked fit result
        n_h = len(acts_harmful_base)
        nonzero_strengths = [s for s in strengths if s != 0]
        coords_by_strength: Dict[int, np.ndarray] = {
            strengths[0]: coords_harmful_base   # unperturbed = baseline slice
        }
        for i, s in enumerate(nonzero_strengths):
            lo = n_h + i * n_h
            coords_by_strength[s] = coords_harmful_fit[lo : lo + n_h]
        print(f"[PROJ] Done.")

    # ── Plot ───────────────────────────────────────────────────────────────
    out_png = os.path.join(run_dir, f"{method}_harmful_politeness_drift_{place}.png")
    plot_harmful_politeness_drift(
        coords_harmless_base=coords_harmless_base,
        coords_harmful_base=coords_harmful_base,
        coords_by_strength=coords_by_strength,
        acts_harmful_base=acts_harmful_base,
        acts_by_strength=acts_harmful_by_strength,
        strengths=strengths,
        out_path_png=out_png,
        method=method,
        place=place,
        model_name=model_name,
        seed=seed,
    )

    # ── Silhouette summary ─────────────────────────────────────────────────
    # Match safety_polite.py exactly: BOTH harmless and harmful prompts are
    # styled at strength s before computing silhouette.  Using unstyled
    # harmless activations for all strengths gives wrong values.
    summary_rows = []
    for s in strengths:
        if s == 0:
            acts_harmless_s = acts_harmless_base
        else:
            print(f"[SIL] harmless s={s}, place={place} …")
            harmless_styled = [apply_politeness(p, s, place=place) for p in harmless_prompts]
            acts_harmless_s = _get_activations_chunked(model, tokenizer, harmless_styled, batch_size)

        X = np.vstack([acts_harmless_s, acts_harmful_by_strength[s]])
        y = np.array([0] * len(acts_harmless_s) + [1] * len(acts_harmful_by_strength[s]))
        sil = float(compute_silhouette_score(X, y, metric="cosine"))
        summary_rows.append({"model": model_name, "place": place, "strength": s, "silhouette": sil})

    df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(run_dir, "summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"[MAIN] Summary saved: {summary_path}")
    print(f"[MAIN] Run directory: {run_dir}")
    return run_dir


def main() -> None:
    """Parse CLI arguments and launch the activation-drift experiment."""
    parser = argparse.ArgumentParser(
        description=(
            "Visualise how harmful-prompt activations drift across the "
            "compliance/refusal decision boundary as politeness increases."
        )
    )
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument("--place",    default=DEFAULT_PLACE,
                        choices=["global", "prefix", "suffix"])
    parser.add_argument("--strengths", nargs="+", type=int, default=DEFAULT_STRENGTHS,
                        help="Strength values; must include 0 (the baseline). "
                             "s=0 points are shown in red. Arrows radiate from the "
                             "s=0 centroid to each other strength's centroid.")
    parser.add_argument("--alpaca_sample_size",    type=int, default=DEFAULT_ALPACA_SIZE)
    parser.add_argument("--harmbench_sample_size", type=int, default=DEFAULT_HARMBENCH_SIZE)
    parser.add_argument("--harmbench_config",      default=DEFAULT_HARMBENCH_CONFIG)
    parser.add_argument("--batch_size", type=int,  default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed",       type=int,  default=DEFAULT_SEED)
    parser.add_argument("--method",     default=DEFAULT_METHOD, choices=["pca", "umap"])
    parser.add_argument("--run_dir",    default=None)
    args = parser.parse_args()

    if len(args.strengths) < 2:
        raise ValueError("--strengths must have at least 2 values.")
    if 0 not in args.strengths:
        raise ValueError("--strengths must include 0 (the unperturbed baseline).")

    run_experiment(
        model_name=args.model,
        place=args.place,
        strengths=args.strengths,
        alpaca_sample_size=args.alpaca_sample_size,
        harmbench_sample_size=args.harmbench_sample_size,
        harmbench_config=args.harmbench_config,
        batch_size=args.batch_size,
        seed=args.seed,
        method=args.method,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()