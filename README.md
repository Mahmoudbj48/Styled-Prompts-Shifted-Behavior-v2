# Styled Prompts, Shifted Behavior

This project studies how natural human writing styles in prompts affect the behavior of instruction-tuned large language models (LLMs).

While many robustness studies focus on adversarial inputs or token-level artifacts (e.g., padding tokens), real users rarely write clean benchmark prompts. Instead, they use informal language, politeness markers, typos, or varied formatting. These stylistic variations may change how LLMs process prompts internally and affect their responses.

Our work systematically evaluates **stylistic prompt variation as a structured robustness space**, analyzing how stylistic changes influence LLM behavior across multiple behavioral axes — including internal representations, model confidence, safety behavior, and output characteristics.

---

## Writing Styles

We analyze three families of realistic prompt styles, each applied with a controlled **strength parameter** (ranging from minimal to strong stylistic modification) and inserted at different **prompt positions**:

| Position | Description |
|----------|-------------|
| Global | Full prompt rewrite |
| Prefix | Style applied to the beginning |
| Suffix | Style applied to the end |

### Politeness / Social Tone
Greetings, politeness markers ("please", "thank you"), and conversational phrasing.

### Surface Noise / Informal Variation
Typos, misspellings, spacing irregularities, and informal abbreviations.

### Structured Rewriting
Reformulating prompts while preserving semantic intent.

---

## Models Evaluated

We evaluate three open-source instruction-tuned model families at two parameter scales each, enabling comparison across architectures and model sizes.

| Family | Models |
|--------|--------|
| **Llama** | Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct |
| **Gemma** | Gemma-2B-it, Gemma-7B-it |
| **Qwen2.5** | Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct |

---

## Behavioral Axes

We evaluate stylistic effects across several behavioral dimensions:

| Axis | Metric | Description |
|------|--------|-------------|
| **Activation Geometry** | Cosine similarity | Measures representational drift in hidden states |
| **Safety Separation** | Silhouette score | Separation between harmful and harmless prompt representations |
| **Confidence** | Δ log-prob | Shift in token-level predictive confidence |
| **Uncertainty** | Entropy shift | Change in output distribution sharpness |
| **Style Mirroring** | LLM judge | Whether model responses mirror the prompt's style |
| **Reasoning Structure** | CoT analysis | Effects on chain-of-thought explanation and step count |

---

## Datasets

Each dataset is evaluated on a subset of approximately 128 prompts.

| Dataset | Use |
|---------|-----|
| **TruthfulQA** | Activation similarity, confidence, and uncertainty analysis |
| **Natural Questions** | Validation of activation and confidence patterns on real queries |
| **GSM8K** | Chain-of-thought reasoning structure analysis |
| **HarmBench** | Harmful prompts for safety analysis |
| **Alpaca** | Harmless prompts paired with HarmBench for silhouette separation |

---

## Repository Structure

```
experiments/
├── politeness.py                  # Politeness / social tone — all behavioral axes
├── spacing.py                     # Surface noise: spacing irregularities
├── punctuation.py                 # Surface noise: punctuation variation
├── letter_case.py                 # Surface noise: letter case variation
├── length_variation.py            # Structured rewriting: length expansion/compression
├── interrogative_vs_imperative.py # Structured rewriting: interrogative vs imperative
├── safety_full.py                 # Safety analysis (activations + ASR) — all styles
├── mirroring.py                   # Style mirroring evaluation (LLM judge)
├── cot_reasoning_generate.py      # Generate chain-of-thought responses under style
├── cot_reasoning.py               # Analyze CoT reasoning structure and step count
└── polite_prompt_check.py         # BERTScore prompt preservation check (all datasets)

plots/
├── plots.py                       # All plotting utilities (line, ridge, radar plots)
├── radar_plots.py                 # Multi-style radar plot generator (Types A–D)
├── run_plots.py                   # CLI runner: aggregate plots, radar, prompt check
├── plot_2d_activation_safety.py   # 2D activation scatter (safety vs politeness)
└── plot_individual_figures.py     # Individual publication-ready figure scripts

utils/
├── data.py                        # Dataset loading utilities
├── models.py                      # Model loading and generation
├── styles.py                      # Style transformation functions
├── metrics.py                     # Metric computation (activations, ASR, BERTScore, …)
├── latex_plots.py                 # LaTeX figure string generation for the paper
├── latex_plots_surface.py         # LaTeX figures for surface noise results
└── llm_style_cache.py             # LLM-rewrite cache for structured styles

data/               # Cached styled prompts and model outputs
results/            # Experiment outputs, plots, and summary CSVs
config.yaml         # Model paths, style levels, and dataset configuration
```

---

## Configuration

Model paths, style strength levels, and dataset settings are defined in `config.yaml`. Style positions and levels for surface noise and structured rewriting styles are also read from this file and do not need to be passed via CLI.

---

## Running Experiments

### Install Dependencies

```bash
pip install -r requirements.txt
```

All scripts are run from the repository root. Results are saved under `results/` in timestamped subdirectories.

---

### Politeness / Social Tone

```bash
python experiments/politeness.py \
    --models L3.1-8B \
    --dataset truthful_qa \
    --experiments all \
    --places prefix suffix global \
    --batch_size 21
```

Key flags: `--experiments` accepts any subset of `prompt response activation confidence mirroring` or `all`.

---

### Surface Noise

Each surface noise style has its own script with the same interface.

```bash
# Spacing
python experiments/spacing.py --models L3.1-8B --dataset truthful_qa --experiments all

# Punctuation
python experiments/punctuation.py --models L3.1-8B --dataset truthful_qa --experiments all

# Letter case (strengths are percentages 0–100)
python experiments/letter_case.py --models L3.1-8B --dataset truthful_qa --experiments all
```

---

### Structured Rewriting

Structured styles rewrite prompts via an LLM API. Set your API key in `.env` before running.

```bash
# Length variation
python experiments/length_variation.py \
    --models L3.1-8B \
    --experiments all \
    --rewrite_provider openai \
    --rewrite_model gpt-4o-mini

# Interrogative vs imperative
python experiments/interrogative_vs_imperative.py \
    --models L3.1-8B \
    --experiments activation confidence \
    --rewrite_provider openai \
    --rewrite_model gpt-4o-mini
```

---

### Chain-of-Thought Reasoning

```bash
# Step 1: generate CoT responses under style
python experiments/cot_reasoning_generate.py \
    --models L3.1-8B \
    --dataset gsm8k \
    --style politeness \
    --batch_size 32

# Step 2: analyze CoT structure and step count
python experiments/cot_reasoning.py \
    --models L3.1-8B \
    --dataset gsm8k \
    --style politeness
```

---

### Safety Analysis

`safety_full.py` supports politeness, surface noise, and structured styles in a single script.

```bash
# Stage 1: generate model responses
python experiments/safety_full.py \
    --model L3.1-8B \
    --compute_activations \
    --compute_asr \
    --asr_stage stage1 \
    --style_family politeness \
    --places prefix suffix global

# Stage 2: judge responses with Llama-Guard-3 (reuse stage1 run directory)
python experiments/safety_full.py \
    --model L3.1-8B \
    --compute_asr \
    --asr_stage stage2 \
    --run_dir results/safety/run_<timestamp>

# Surface noise example
python experiments/safety_full.py \
    --model L3.1-8B \
    --compute_activations --compute_asr \
    --asr_stage both \
    --style_family surface_noise \
    --style_name spacing
```

---

### Style Mirroring

```bash
python experiments/mirroring.py \
    --models L3.1-8B \
    --dataset truthful_qa \
    --style politeness \
    --places prefix suffix global
```

---

## Generating Plots

All plotting is driven by `plots/run_plots.py`, which has three modes.

### Aggregate Line / Ridge / Radar Plots

```bash
python plots/run_plots.py \
    --runs results/politeness/run_<timestamp>/summary.csv \
    --out_dir results/combined_plots/politeness \
    --style_name politeness \
    --dataset_name TruthfulQA \
    --save_pdf
```

For surface noise styles (e.g. spacing):

```bash
python plots/run_plots.py \
    --runs results/spacing/run_<timestamp>/combined_means_by_model_place_strength.csv \
    --out_dir results/combined_plots/spacing \
    --style_name spacing \
    --dataset_name TruthfulQA
```

---

### Multi-Style Radar Plots

Generates Type D radar plots comparing all styles and behavioral axes across models.

```bash
python plots/run_plots.py \
    --multi_style_radar \
    --out_dir results/combined_plots/radar \
    --style_data \
        politeness:results/politeness/run_a/summary.csv \
        spacing:results/spacing/run_x/summary.csv \
        letter_case:results/letter_case/run_y/summary.csv \
    --asr_data \
        politeness:results/safety/run_pol/summary.csv \
    --cot_data \
        politeness:results/cot/run_pol/ \
    --save_pdf
```

---

### BERTScore Prompt-Preservation Check

Evaluates how much each style changes the semantic content of prompts.

```bash
python plots/run_plots.py \
    --prompt_check \
    --out_dir results/prompt_check \
    --save_pdf
```
