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
| **Bias** | Demographic bias benchmarks | Sensitivity of bias behavior to stylistic changes |
| **Reasoning Structure** | Structural analysis | Effects on explanation and reasoning chains |

---

## Datasets

Each dataset is evaluated on a subset of approximately 128 prompts.

| Dataset | Use |
|---------|-----|
| **TruthfulQA** | Activation similarity, confidence, and uncertainty analysis |
| **Natural Questions** | Validation of activation and confidence patterns on real queries |
| **HarmBench** | Harmful prompts for safety analysis |
| **Alpaca** | Harmless prompts paired with HarmBench for silhouette separation |
| **BBQ** | Demographic bias sensitivity measurement |

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
├── bbq_bias_full.py               # Demographic bias analysis (BBQ benchmark)
├── cot_reasoning_generate.py      # Generate chain-of-thought responses
├── cot_reasoning.py               # Analyze CoT reasoning structure
└── polite_prompt_check.py         # BERTScore prompt preservation check (all datasets)

utils/
├── plot_utils.py                  # Shared NeurIPS-style plot helpers 
├── aggregate_plots.py             # Aggregate line / radar / ridge plots across runs
├── politeness_plots.py            # Politeness-specific plots + BERTScore prompt plot
├── surface_plots.py               # Surface noise experiment plots
├── structuredness_plots.py        # Structured rewriting experiment plots
├── cot_plots.py                   # Chain-of-thought reasoning plots
├── latex_plots.py                 # LaTeX figure string generation for the paper
├── data.py                        # Dataset loading utilities
├── models.py                      # Model loading and generation
├── styles.py                      # Style transformation functions
├── metrics.py                     # Metric computation (activations, ASR, BERTScore, …)
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

### Politeness / Social Tone

```bash
# Run all behavioral axes on one or more models
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

Each surface noise style has its own script. They share the same interface.

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

Structured styles rewrite prompts via an LLM API. Set your API in .env key before running.

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

### Bias (BBQ)

```bash
python experiments/bbq_bias_full.py --model L3.1-8B --sample_size 128
```


---

### Chain-of-Thought Reasoning

```bash
# Generate CoT responses
python experiments/cot_reasoning_generate.py \
    --models L3.1-8B \
    --dataset gsm8k \
    --style politeness \
    --batch_size 32

# Analyze CoT structure
python experiments/cot_reasoning.py \
    --models L3.1-8B \
    --dataset gsm8k \
    --style politeness
```

---

