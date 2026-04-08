#!/bin/bash
# New Models Experiments

# ============================================================
# POLITENESS
# ============================================================

# --- truthful_qa ---
python experiments/politeness.py --models Q3.5-9B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global
python experiments/politeness.py --models Q3.5-27B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global
python experiments/politeness.py --models G4-26B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global
python experiments/politeness.py --models G4-E4B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global

## --- natural_questions ---
#python experiments/politeness.py --models Q3.5-9B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global
#python experiments/politeness.py --models Q3.5-27B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global
#python experiments/politeness.py --models G4-26B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global
#python experiments/politeness.py --models G4-E4B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global

# ============================================================
# SAFETY - POLITENESS
# ============================================================

# --- activations ---
python experiments/safety_full.py --models Q3.5-9B  --style_family politeness --places prefix suffix global --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/politeness_activations
python experiments/safety_full.py --models Q3.5-27B --style_family politeness --places prefix suffix global --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/politeness_activations
python experiments/safety_full.py --models G4-26B   --style_family politeness --places prefix suffix global --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/politeness_activations
python experiments/safety_full.py --models G4-E4B   --style_family politeness --places prefix suffix global --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/politeness_activations

# --- ASR (stage1 + stage2) ---
python experiments/safety_full.py --models Q3.5-9B  --style_family politeness --places prefix suffix global --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/politeness_asr
python experiments/safety_full.py --models Q3.5-27B --style_family politeness --places prefix suffix global --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/politeness_asr
python experiments/safety_full.py --models G4-26B   --style_family politeness --places prefix suffix global --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/politeness_asr
python experiments/safety_full.py --models G4-E4B   --style_family politeness --places prefix suffix global --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/politeness_asr

# ============================================================
# COT - POLITENESS
# ============================================================

# --- generate CoT responses (gsm8k) ---
python experiments/cot_reasoning_generate.py --models Q3.5-9B Q3.5-27B G4-26B G4-E4B --dataset gsm8k --sample_size 128 --style politeness --places global --strengths -8 -4 0 4 8 --max_new_tokens 300
COT_RUN_DIR=$(ls -td results/cot_responses/run_gsm8k_politeness_*/ | head -1)

# --- CoT analysis ---
python experiments/compute_cot_analysis.py --run_dir "$COT_RUN_DIR" --sample_size 128 --judge_provider openai --judge_model gpt-4o-mini

# --- correctness analysis ---
python experiments/compute_correctness.py --run_dir "$COT_RUN_DIR" --sample_size 128 --judge_provider openai --judge_model gpt-4o-mini

# ============================================================
# LENGTH VARIATION
# ============================================================

# --- truthful_qa ---
python experiments/length_variation.py --models Q3.5-9B --dataset truthful_qa --sample_size 128 --experiments all
python experiments/length_variation.py --models Q3.5-27B --dataset truthful_qa --sample_size 128 --experiments all
python experiments/length_variation.py --models G4-26B --dataset truthful_qa --sample_size 128 --experiments all
python experiments/length_variation.py --models G4-E4B --dataset truthful_qa --sample_size 128 --experiments all

## --- natural_questions ---
#python experiments/length_variation.py --models Q3.5-9B --dataset natural_questions --sample_size 128 --experiments all
#python experiments/length_variation.py --models Q3.5-27B --dataset natural_questions --sample_size 128 --experiments all
#python experiments/length_variation.py --models G4-26B --dataset natural_questions --sample_size 128 --experiments all
#python experiments/length_variation.py --models G4-E4B --dataset natural_questions --sample_size 128 --experiments all

# ============================================================
# SAFETY - LENGTH VARIATION
# ============================================================

# --- activations ---
python experiments/safety_full.py --models Q3.5-9B  --style_family structured --style_name length_variation --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/length_variation_activations
python experiments/safety_full.py --models Q3.5-27B --style_family structured --style_name length_variation --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/length_variation_activations
python experiments/safety_full.py --models G4-26B   --style_family structured --style_name length_variation --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/length_variation_activations
python experiments/safety_full.py --models G4-E4B   --style_family structured --style_name length_variation --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/length_variation_activations

# --- ASR (stage1 + stage2) ---
python experiments/safety_full.py --models Q3.5-9B  --style_family structured --style_name length_variation --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/length_variation_asr
python experiments/safety_full.py --models Q3.5-27B --style_family structured --style_name length_variation --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/length_variation_asr
python experiments/safety_full.py --models G4-26B   --style_family structured --style_name length_variation --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/length_variation_asr
python experiments/safety_full.py --models G4-E4B   --style_family structured --style_name length_variation --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/length_variation_asr

# ============================================================
# COT - LENGTH VARIATION
# ============================================================

# --- generate CoT responses (gsm8k) ---
python experiments/cot_reasoning_generate.py --models Q3.5-9B Q3.5-27B G4-26B G4-E4B --dataset gsm8k --sample_size 128 --style length_variation --max_new_tokens 300
COT_RUN_DIR=$(ls -td results/cot_responses/run_gsm8k_length_variation_*/ | head -1)

# --- CoT analysis ---
python experiments/compute_cot_analysis.py --run_dir "$COT_RUN_DIR" --sample_size 128 --judge_provider openai --judge_model gpt-4o-mini

# --- correctness analysis ---
python experiments/compute_correctness.py --run_dir "$COT_RUN_DIR" --sample_size 128 --judge_provider openai --judge_model gpt-4o-mini

# ============================================================
# LETTER CASE
# ============================================================

# --- truthful_qa ---
python experiments/letter_case.py --models Q3.5-9B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global
python experiments/letter_case.py --models Q3.5-27B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global
python experiments/letter_case.py --models G4-26B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global
python experiments/letter_case.py --models G4-E4B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global

# --- compute mirroring ---
LETTER_CASE_RUN_DIR=$(ls -td results/letter_case/run_multi_truthful_qa_*/ | head -1)
python experiments/compute_mirroring.py --input "${LETTER_CASE_RUN_DIR}full_results_all_models.csv" --style letter_case

## --- natural_questions ---
#python experiments/letter_case.py --models Q3.5-9B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global
#python experiments/letter_case.py --models Q3.5-27B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global
#python experiments/letter_case.py --models G4-26B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global
#python experiments/letter_case.py --models G4-E4B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global

# ============================================================
# SAFETY - LETTER CASE
# ============================================================

# --- activations ---
python experiments/safety_full.py --models Q3.5-9B  --style_family surface_noise --style_name letter_case --places prefix suffix global --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/letter_case_activations
python experiments/safety_full.py --models Q3.5-27B --style_family surface_noise --style_name letter_case --places prefix suffix global --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/letter_case_activations
python experiments/safety_full.py --models G4-26B   --style_family surface_noise --style_name letter_case --places prefix suffix global --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/letter_case_activations
python experiments/safety_full.py --models G4-E4B   --style_family surface_noise --style_name letter_case --places prefix suffix global --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/letter_case_activations

# --- ASR (stage1 + stage2) ---
python experiments/safety_full.py --models Q3.5-9B  --style_family surface_noise --style_name letter_case --places prefix suffix global --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/letter_case_asr
python experiments/safety_full.py --models Q3.5-27B --style_family surface_noise --style_name letter_case --places prefix suffix global --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/letter_case_asr
python experiments/safety_full.py --models G4-26B   --style_family surface_noise --style_name letter_case --places prefix suffix global --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/letter_case_asr
python experiments/safety_full.py --models G4-E4B   --style_family surface_noise --style_name letter_case --places prefix suffix global --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/letter_case_asr

# ============================================================
# COT - LETTER CASE
# ============================================================

# --- generate CoT responses (gsm8k) ---
python experiments/cot_reasoning_generate.py --models Q3.5-9B Q3.5-27B G4-26B G4-E4B --dataset gsm8k --sample_size 128 --style letter_case --places global --strengths 0 25 50 100 --max_new_tokens 300
COT_RUN_DIR=$(ls -td results/cot_responses/run_gsm8k_letter_case_*/ | head -1)

# --- CoT analysis ---
python experiments/compute_cot_analysis.py --run_dir "$COT_RUN_DIR" --sample_size 128 --judge_provider openai --judge_model gpt-4o-mini

# --- correctness analysis ---
python experiments/compute_correctness.py --run_dir "$COT_RUN_DIR" --sample_size 128 --judge_provider openai --judge_model gpt-4o-mini

# ============================================================
# PUNCTUATION
# ============================================================

# --- truthful_qa ---
python experiments/punctuation.py --models Q3.5-9B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global
python experiments/punctuation.py --models Q3.5-27B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global
python experiments/punctuation.py --models G4-26B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global
python experiments/punctuation.py --models G4-E4B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global

# --- compute mirroring ---
PUNCTUATION_RUN_DIR=$(ls -td results/punctuation/run_multi_truthful_qa_*/ | head -1)
python experiments/compute_mirroring.py --input "${PUNCTUATION_RUN_DIR}full_results_all_models.csv" --style punctuation

## --- natural_questions ---
#python experiments/punctuation.py --models Q3.5-9B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global
#python experiments/punctuation.py --models Q3.5-27B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global
#python experiments/punctuation.py --models G4-26B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global
#python experiments/punctuation.py --models G4-E4B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global

# ============================================================
# SAFETY - PUNCTUATION
# ============================================================

# --- activations ---
python experiments/safety_full.py --models Q3.5-9B  --style_family surface_noise --style_name punctuation --places prefix suffix global --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/punctuation_activations
python experiments/safety_full.py --models Q3.5-27B --style_family surface_noise --style_name punctuation --places prefix suffix global --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/punctuation_activations
python experiments/safety_full.py --models G4-26B   --style_family surface_noise --style_name punctuation --places prefix suffix global --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/punctuation_activations
python experiments/safety_full.py --models G4-E4B   --style_family surface_noise --style_name punctuation --places prefix suffix global --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/punctuation_activations

# --- ASR (stage1 + stage2) ---
python experiments/safety_full.py --models Q3.5-9B  --style_family surface_noise --style_name punctuation --places prefix suffix global --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/punctuation_asr
python experiments/safety_full.py --models Q3.5-27B --style_family surface_noise --style_name punctuation --places prefix suffix global --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/punctuation_asr
python experiments/safety_full.py --models G4-26B   --style_family surface_noise --style_name punctuation --places prefix suffix global --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/punctuation_asr
python experiments/safety_full.py --models G4-E4B   --style_family surface_noise --style_name punctuation --places prefix suffix global --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/punctuation_asr

# ============================================================
# COT - PUNCTUATION
# ============================================================

# --- generate CoT responses (gsm8k) ---
python experiments/cot_reasoning_generate.py --models Q3.5-9B Q3.5-27B G4-26B G4-E4B --dataset gsm8k --sample_size 128 --style punctuation --places global --strengths 0 3 10 20 --max_new_tokens 300
COT_RUN_DIR=$(ls -td results/cot_responses/run_gsm8k_punctuation_*/ | head -1)

# --- CoT analysis ---
python experiments/compute_cot_analysis.py --run_dir "$COT_RUN_DIR" --sample_size 128 --judge_provider openai --judge_model gpt-4o-mini

# --- correctness analysis ---
python experiments/compute_correctness.py --run_dir "$COT_RUN_DIR" --sample_size 128 --judge_provider openai --judge_model gpt-4o-mini

# ============================================================
# SPACING
# ============================================================

# --- truthful_qa ---
python experiments/spacing.py --models Q3.5-9B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global
python experiments/spacing.py --models Q3.5-27B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global
python experiments/spacing.py --models G4-26B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global
python experiments/spacing.py --models G4-E4B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global

# --- compute mirroring ---
SPACING_RUN_DIR=$(ls -td results/spacing/run_multi_truthful_qa_*/ | head -1)
python experiments/compute_mirroring.py --input "${SPACING_RUN_DIR}full_results_all_models.csv" --style spacing

## --- natural_questions ---
#python experiments/spacing.py --models Q3.5-9B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global
#python experiments/spacing.py --models Q3.5-27B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global
#python experiments/spacing.py --models G4-26B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global
#python experiments/spacing.py --models G4-E4B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global

# ============================================================
# SAFETY - SPACING
# ============================================================

# --- activations ---
python experiments/safety_full.py --models Q3.5-9B  --style_family surface_noise --style_name spacing --places prefix suffix global --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/spacing_activations
python experiments/safety_full.py --models Q3.5-27B --style_family surface_noise --style_name spacing --places prefix suffix global --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/spacing_activations
python experiments/safety_full.py --models G4-26B   --style_family surface_noise --style_name spacing --places prefix suffix global --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/spacing_activations
python experiments/safety_full.py --models G4-E4B   --style_family surface_noise --style_name spacing --places prefix suffix global --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/spacing_activations

# --- ASR (stage1 + stage2) ---
python experiments/safety_full.py --models Q3.5-9B  --style_family surface_noise --style_name spacing --places prefix suffix global --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/spacing_asr
python experiments/safety_full.py --models Q3.5-27B --style_family surface_noise --style_name spacing --places prefix suffix global --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/spacing_asr
python experiments/safety_full.py --models G4-26B   --style_family surface_noise --style_name spacing --places prefix suffix global --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/spacing_asr
python experiments/safety_full.py --models G4-E4B   --style_family surface_noise --style_name spacing --places prefix suffix global --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/spacing_asr

# ============================================================
# COT - SPACING
# ============================================================

# --- generate CoT responses (gsm8k) ---
python experiments/cot_reasoning_generate.py --models Q3.5-9B Q3.5-27B G4-26B G4-E4B --dataset gsm8k --sample_size 128 --style spacing --places global --strengths 0 20 50 100 --max_new_tokens 300
COT_RUN_DIR=$(ls -td results/cot_responses/run_gsm8k_spacing_*/ | head -1)

# --- CoT analysis ---
python experiments/compute_cot_analysis.py --run_dir "$COT_RUN_DIR" --sample_size 128 --judge_provider openai --judge_model gpt-4o-mini

# --- correctness analysis ---
python experiments/compute_correctness.py --run_dir "$COT_RUN_DIR" --sample_size 128 --judge_provider openai --judge_model gpt-4o-mini

# ============================================================
# INTERROGATIVE VS IMPERATIVE
# ============================================================

# --- truthful_qa ---
python experiments/interrogative_vs_imperative.py --models Q3.5-9B --dataset truthful_qa --sample_size 128 --experiments all
python experiments/interrogative_vs_imperative.py --models Q3.5-27B --dataset truthful_qa --sample_size 128 --experiments all
python experiments/interrogative_vs_imperative.py --models G4-26B --dataset truthful_qa --sample_size 128 --experiments all
python experiments/interrogative_vs_imperative.py --models G4-E4B --dataset truthful_qa --sample_size 128 --experiments all

## --- natural_questions ---
#python experiments/interrogative_vs_imperative.py --models Q3.5-9B --dataset natural_questions --sample_size 128 --experiments all
#python experiments/interrogative_vs_imperative.py --models Q3.5-27B --dataset natural_questions --sample_size 128 --experiments all
#python experiments/interrogative_vs_imperative.py --models G4-26B --dataset natural_questions --sample_size 128 --experiments all
#python experiments/interrogative_vs_imperative.py --models G4-E4B --dataset natural_questions --sample_size 128 --experiments all

# ============================================================
# SAFETY - INTERROGATIVE VS IMPERATIVE
# ============================================================

# --- activations ---
python experiments/safety_full.py --models Q3.5-9B  --style_family structured --style_name inter_vs_imper --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/inter_vs_imper_activations
python experiments/safety_full.py --models Q3.5-27B --style_family structured --style_name inter_vs_imper --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/inter_vs_imper_activations
python experiments/safety_full.py --models G4-26B   --style_family structured --style_name inter_vs_imper --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/inter_vs_imper_activations
python experiments/safety_full.py --models G4-E4B   --style_family structured --style_name inter_vs_imper --harmbench_size 128 --alpaca_size 128 --compute_asr False --run_dir results/safety/inter_vs_imper_activations

# --- ASR (stage1 + stage2) ---
python experiments/safety_full.py --models Q3.5-9B  --style_family structured --style_name inter_vs_imper --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/inter_vs_imper_asr
python experiments/safety_full.py --models Q3.5-27B --style_family structured --style_name inter_vs_imper --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/inter_vs_imper_asr
python experiments/safety_full.py --models G4-26B   --style_family structured --style_name inter_vs_imper --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/inter_vs_imper_asr
python experiments/safety_full.py --models G4-E4B   --style_family structured --style_name inter_vs_imper --harmbench_size 128 --asr_stage both --compute_activations False --run_dir results/safety/inter_vs_imper_asr

# ============================================================
# COT - INTERROGATIVE VS IMPERATIVE
# ============================================================

# --- generate CoT responses (gsm8k) ---
python experiments/cot_reasoning_generate.py --models Q3.5-9B Q3.5-27B G4-26B G4-E4B --dataset gsm8k --sample_size 128 --style inter_vs_imper --max_new_tokens 300
COT_RUN_DIR=$(ls -td results/cot_responses/run_gsm8k_inter_vs_imper_*/ | head -1)

# --- CoT analysis ---
python experiments/compute_cot_analysis.py --run_dir "$COT_RUN_DIR" --sample_size 128 --judge_provider openai --judge_model gpt-4o-mini

# --- correctness analysis ---
python experiments/compute_correctness.py --run_dir "$COT_RUN_DIR" --sample_size 128 --judge_provider openai --judge_model gpt-4o-mini
