# Rephrase and Fail: Stylistic Prompt Variation Exposes Multi-Axis Generalization Failures in LLMs


## Overview

We formalize **stylistic generalization** as the ability of an instruction-tuned LLM to maintain consistent per-instance behavior under semantic-preserving variation in prompt style. We introduce the **Stylistic Generalization Score (SGS)**, which quantifies generalization as the mean per-prompt standard deviation of behavioral deltas across multi-dimensional stylistic variations, evaluated along four behavioral axes: activation geometry, generation quality, confidence and uncertainty, and style mirroring.

We evaluate eight open-source and three closed-source instruction-tuned LLMs across six benchmark datasets and three families of semantic-preserving stylistic transformations. **No model achieves uniform stylistic generalization** (p < 0.001 for all SGS values). Generation quality is consistently the most sensitive axis; confidence remains largely stable. Content stability and style mirroring are independent failure modes — the model with the lowest generation sensitivity records the highest mirroring instability, a dissociation that a single aggregate robustness score would conflate.

---

## Stylistic Transformations

Three families of semantic-preserving prompt styles, each applied with controlled strength levels:

| Family | Styles |
|--------|--------|
| **Politeness / Social Tone** | Greetings, politeness markers, conversational phrasing — applied at prefix, suffix, or global position |
| **Surface Noise** | Spacing irregularities, punctuation variation, letter case randomization |
| **Structured Rewriting** | Length expansion/compression, interrogative vs. imperative reformulation (LLM-rewritten) |

---

## Models

**Open-source (8):**

| Family | Models |
|--------|--------|
| **Llama** | Llama-3.2-3B-Instruct (L3.2-3B), Llama-3.1-8B-Instruct (L3.1-8B) |
| **Gemma** | Gemma-2B-IT (G-2B), Gemma-7B-IT (G-7B), Gemma-4-E4B-IT (G4-E4B) |
| **Qwen** | Qwen2.5-1.5B-Instruct (Q2.5-1.5B), Qwen2.5-7B-Instruct (Q2.5-7B), Qwen3.5-9B (Q3.5-9B) |

**Closed-source (3):** GPT-5.4, Gemini-2.5-Flash, Claude-Sonnet-4.6

---

## Behavioral Axes

| Axis | Metrics | Description |
|------|---------|-------------|
| **Activation Geometry** | Cosine similarity | Representational drift in hidden states under stylistic variation |
| **Generation Quality** | BLEU, BERTScore-F1 | Lexical and semantic similarity between baseline and styled outputs |
| **Confidence & Uncertainty** | Δ log-prob, entropy shift | Shift in token-level predictive confidence and output distribution sharpness |
| **Style Mirroring** | LLM judge, rule-based heuristics, length check | Whether the model's response adapts to match the prompt's style |

SGS aggregates per-instance deltas across all four axes into a single generalization score per model.

---

## Datasets

N = 16 prompts sampled per dataset. Open-source models are evaluated on all six; closed-source models on three (TruthfulQA, Alpaca, SimpleQA Verified) due to API cost constraints.

| Dataset | Description |
|---------|-------------|
| **TruthfulQA** | Factual questions probing susceptibility to misconceptions |
| **Natural Questions** | Real user search queries with factual answers |
| **Alpaca** | Diverse general instruction-following prompts |
| **SimpleQA (Verified)** | Short, unambiguous factual questions with verified answers |
| **TriviaQA** | Trivia questions with naturally varied phrasing |
| **HotpotQA** | Multi-hop reasoning questions over multiple supporting facts |

---

## Repository Structure

```
experiments/
├── politeness.py                    # Politeness / social tone experiments
├── spacing.py                       # Surface noise: spacing
├── punctuation.py                   # Surface noise: punctuation
├── letter_case.py                   # Surface noise: letter case
├── length_variation.py              # Structured rewriting: length
├── interrogative_vs_imperative.py   # Structured rewriting: interrogative vs. imperative
├── compute_mirroring.py             # Style mirroring detection for surface noise styles
└── run_closed_models.py             # Experiments for closed-source models

plots/
├── plots.py                         # Core plotting utilities
├── run_plots.py                     # CLI runner for all plot types
├── plot_individual_figures.py
├── run_bertscore_check_paper.py     # BERTScore semantic-preservation check
└── sensitivity_analysis.py

utils/
├── data.py                          # Dataset loading
├── models.py                        # Model loading and generation
├── styles.py                        # Style transformation functions
├── metrics.py                       # Metric computation (activations, BERTScore, …)
├── llm_client.py                    # API client for closed models and LLM-rewriting
├── llm_style_cache.py               # Cache for LLM-rewritten prompts
├── compute_delta_metrics.py         # Per-instance behavioral delta computation
├── compute_sgs_table.py             # SGS table generation (open-source models)
├── compute_sgs_closed_table.py      # SGS table generation (closed-source models)
├── closed_model_metrics.py          # Metrics specific to closed-source model outputs
├── surface_mirroring_detector.py    # Rule-based style mirroring detection
└── significance_test.py             # Statistical significance tests

config.yaml                          # Model paths, datasets, style levels, positions
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

# Required for structured rewriting (length variation, interrogative vs. imperative)
# and LLM-as-judge style mirroring (politeness)
# Uses GPT-4o-mini by default — can reuse OPENAI_API_KEY above
```

All scripts are run from the repository root.

---

## Running Experiments

### Open-source models

Each experiment script accepts `--models`, `--dataset`, and `--experiments` flags. `--experiments all` runs all four behavioral axes.

```bash
# Politeness
python experiments/politeness.py \
    --models L3.1-8B \
    --dataset alpaca \
    --experiments all \
    --places prefix suffix global

# Surface noise (spacing, punctuation, letter_case share the same interface)
python experiments/spacing.py --models L3.1-8B --dataset alpaca --experiments all
python experiments/punctuation.py --models L3.1-8B --dataset alpaca --experiments all
python experiments/letter_case.py --models L3.1-8B --dataset alpaca --experiments all

# Structured rewriting (requires LLM rewriter API)
python experiments/length_variation.py \
    --models L3.1-8B \
    --dataset alpaca \
    --experiments all \
    --rewrite_provider openai \
    --rewrite_model gpt-4o-mini

python experiments/interrogative_vs_imperative.py \
    --models L3.1-8B \
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

### Style mirroring (surface noise post-processing)

```bash
python experiments/compute_mirroring.py \
    --input results/spacing/run_<timestamp>/full_results_all_models.csv \
    --style spacing
```

---

## Generating Plots

All plotting is driven by `plots/run_plots.py`.

```bash
# Aggregate plots for a single style
python plots/run_plots.py \
    --runs results/politeness/run_<timestamp>/summary.csv \
    --out_dir results/combined_plots/politeness \
    --style_name politeness \
    --dataset_name alpaca \
    --save_pdf

# Multi-style radar plots (SGS visualization)
python plots/run_plots.py \
    --multi_style_radar \
    --out_dir results/combined_plots/radar \
    --style_data \
        politeness:results/politeness/run_a/summary.csv \
        spacing:results/spacing/run_x/summary.csv \
        letter_case:results/letter_case/run_y/summary.csv \
    --save_pdf

# BERTScore semantic-preservation check for prompts
python plots/run_bertscore_check_paper.py
```
