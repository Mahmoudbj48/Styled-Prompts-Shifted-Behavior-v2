# Do LLMs Treat Input Consistently Across Variations? Generalization via Stability


## Overview

We formalize **stylistic generalization** as the ability of an instruction-tuned LLM to maintain consistent per-instance behavior under semantic-preserving variation in prompt formulation, and introduce the **Stability-Aware Generalization Objective (SAGO)** framework for evaluating it. SAGO has three components: (1) a benchmark of controlled input variations applied to each prompt, (2) the **Stability Generalization Score (SGS)** that quantifies how much a chosen behavioral property changes across variations, computed per axis, and (3) a set of axes — concrete model properties along which generalization may fail — over which the score is evaluated.

We evaluate eight open-source and three closed-source instruction-tuned LLMs across six benchmark datasets and three families of semantic-preserving variations along four behavioral axes: activation geometry, generation quality, confidence and uncertainty, and response mirroring. **No model achieves uniform stylistic generalization** (p < 0.001 for all SGS values). Generation quality is consistently the most sensitive axis; confidence remains largely stable. Content stability and response mirroring are independent failure modes — the model with the lowest generation sensitivity records the highest mirroring instability, a dissociation that a single aggregate robustness score would conflate.

---

## Variation Families

Three families of semantic-preserving prompt variations, each applied with controlled subtype, strength, and position:

| Family | Subtypes | Range |
|--------|----------|-------|
| **Social Register** | Politeness levels (rude → neutral → polite) | strength s ∈ {-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10}, positions: prefix, suffix, global |
| **Surface Noise** | Spacing, punctuation, letter case | spacing s ∈ {0,1,5,20,50,100}; punctuation s ∈ {0,1,3,5,10,20}; casing s ∈ {0,10,25,50,75,100}; positions: prefix, suffix, global |
| **Structural Rewriting** | Length variation, sentence form (LLM-rewritten) | length s ∈ {0.25, 0.5, 1.0, 1.5, 2.0, 3.0}; sentence form s ∈ {interrogative, imperative}; global only |

All variants are validated with BERTScore > 0.85 against the baseline prompt to enforce semantic preservation.

---

## Models

**Open-source (8):**

| Family | Models |
|--------|--------|
| **Llama** | Llama-3.2-3B-Instruct (L-3B), Llama-3.1-8B-Instruct (L-8B) |
| **Gemma** | Gemma-2B-IT (G-2B), Gemma-7B-IT (G-7B), Gemma-4-E4B-IT (G4-E4B) |
| **Qwen** | Qwen2.5-1.5B-Instruct (Q-1.5B), Qwen2.5-7B-Instruct (Q-7B), Qwen3.5-9B (Q3.5-9B) |

**Closed-source (3):** GPT-5.4, Gemini-2.5-Flash, Claude-Sonnet-4.6

Activation geometry is unavailable for all closed-source models (no internal-state access via API). Confidence and uncertainty are computed for GPT-5.4 only, as it is the sole closed-source model that exposes token-level log-probabilities.

---

## Behavioral Axes

| Axis | Δ definition | Description |
|------|--------------|-------------|
| **Activation Geometry** (Δ-Cos) | Mean cosine similarity (variant, baseline) − 1 across all layers | Representational drift in hidden states under variation |
| **Generation Quality** (Δ-BLEU, Δ-BERT) | BLEU(variant, baseline) − 1, BERTScore-F1(variant, baseline) − 1 | Lexical and semantic dissimilarity between baseline and variant outputs |
| **Confidence & Uncertainty** (Δ-Prob, Δ-Ent) | log p(variant) − log p(baseline), entropy analog | Shift in token-level predictive confidence and distributional sharpness |
| **Response Mirroring** (Δ-MR) | Binary detector m(i,v) ∈ {0,1} | Whether the model's response adapts to match the variant's framing |

For each axis X, SGS is the mean per-prompt standard deviation of normalized Δ values across all variants in 𝒱. Lower SGS = stronger generalization; SGS = 0 iff Δ_X(i,v) = 0 for every prompt and variant.

Statistical significance is assessed per (model, axis) and (model, axis, dataset) combination via a one-sample one-sided t-test (H₀: 𝔼[s_X(i)] = 0) at α = 0.05.

---

## Datasets

N = 16 prompts sampled per dataset (justified by the prompt sample size ablation). Open-source models are evaluated on all six datasets; closed-source models on three (TruthfulQA, Alpaca, SimpleQA Verified) due to API cost constraints.

| Dataset | Description |
|---------|-------------|
| **TruthfulQA** | Factual questions probing susceptibility to misconceptions across health, law, finance, politics |
| **Natural Questions** | Real user search queries with factual answers from open-domain Wikipedia |
| **Alpaca** | Diverse general instruction-following prompts (brainstorming, classification, writing, creative generation) |
| **SimpleQA (Verified)** | Short, unambiguous factual questions across science, history, geography, entertainment with verified answers |
| **TriviaQA** | Trivia questions from quiz-league and Wikipedia sources with naturally varied phrasing |
| **HotpotQA** | Multi-hop questions over Wikipedia paragraphs requiring reasoning over multiple supporting facts |

---

## Repository Structure

```
experiments/
├── politeness.py                    # Social register variation
├── spacing.py                       # Surface noise: spacing
├── punctuation.py                   # Surface noise: punctuation
├── letter_case.py                   # Surface noise: letter case
├── length_variation.py              # Structural rewriting: length
├── interrogative_vs_imperative.py   # Structural rewriting: sentence form
├── compute_mirroring.py             # Mirroring detector for surface noise variants
└── run_closed_models.py             # Closed-source model experiments

plots/
├── plots.py                         # Core plotting utilities
├── run_plots.py                     # CLI runner for all plot types
├── plot_individual_figures.py
├── run_bertscore_check_paper.py     # BERTScore semantic-preservation check
└── sensitivity_analysis.py

utils/
├── data.py                          # Dataset loading
├── models.py                        # Model loading and generation
├── styles.py                        # Variation transformation functions
├── metrics.py                       # Metric computation (activations, BERTScore, …)
├── llm_client.py                    # API client for closed-source models and LLM-rewriter
├── llm_style_cache.py               # Cache for LLM-rewritten variants
├── compute_delta_metrics.py         # Per-instance Δ_X(i,v) computation
├── compute_sgs_table.py             # SGS table generation (open-source models)
├── compute_sgs_closed_table.py      # SGS table generation (closed-source models)
├── closed_model_metrics.py          # Metrics specific to closed-source model outputs
├── surface_mirroring_detector.py    # Rule-based mirroring detector for surface noise
└── significance_test.py             # One-sided t-test (H₀: 𝔼[s_X(i)] = 0)

config.yaml                          # Model paths, datasets, variation strengths, positions
requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Several components require API keys, configured in a `.env` file at the project root:

```
# Required for closed-source model experiments (GPT-5.4)
OPENAI_API_KEY=...

# Required for closed-source model experiments (Gemini-2.5-Flash)
GOOGLE_API_KEY=...

# Required for closed-source model experiments (Claude-Sonnet-4.6)
ANTHROPIC_API_KEY=...

# Required for structural rewriting (length variation, interrogative vs. imperative)
# and LLM-as-judge mirroring detection (social register)
# Uses GPT-4o-mini by default — can reuse OPENAI_API_KEY above
```

All scripts are run from the repository root.

---

## Running Experiments

### Open-source models

Each experiment script accepts `--models`, `--dataset`, and `--experiments` flags. `--experiments all` runs all four behavioral axes.

```bash
# Social register (politeness)
python experiments/politeness.py \
    --models L-8B \
    --dataset alpaca \
    --experiments all \
    --places prefix suffix global

# Surface noise (spacing, punctuation, letter_case share the same interface)
python experiments/spacing.py --models L-8B --dataset alpaca --experiments all
python experiments/punctuation.py --models L-8B --dataset alpaca --experiments all
python experiments/letter_case.py --models L-8B --dataset alpaca --experiments all

# Structural rewriting (requires LLM rewriter API)
python experiments/length_variation.py \
    --models L-8B \
    --dataset alpaca \
    --experiments all \
    --rewrite_provider openai \
    --rewrite_model gpt-4o-mini

python experiments/interrogative_vs_imperative.py \
    --models L-8B \
    --dataset alpaca \
    --experiments all \
    --rewrite_provider openai \
    --rewrite_model gpt-4o-mini
```

### Closed-source models

```bash
python experiments/run_closed_models.py \
    --dataset alpaca \
    --experiments all
```

### Mirroring detection (surface noise post-processing)

```bash
python experiments/compute_mirroring.py \
    --input results/spacing/run_<timestamp>/full_results_all_models.csv \
    --style spacing
```

---

## Generating Plots

All plotting is driven by `plots/run_plots.py`.

```bash
# Aggregate plots for a single variation family
python plots/run_plots.py \
    --runs results/politeness/run_<timestamp>/summary.csv \
    --out_dir results/combined_plots/politeness \
    --style_name politeness \
    --dataset_name alpaca \
    --save_pdf

# Multi-family radar plots (SGS visualization)
python plots/run_plots.py \
    --multi_style_radar \
    --out_dir results/combined_plots/radar \
    --style_data \
        politeness:results/politeness/run_a/summary.csv \
        spacing:results/spacing/run_x/summary.csv \
        letter_case:results/letter_case/run_y/summary.csv \
    --save_pdf

# BERTScore semantic-preservation check
python plots/run_bertscore_check_paper.py
```
