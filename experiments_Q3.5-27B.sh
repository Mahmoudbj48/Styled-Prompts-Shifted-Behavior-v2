#!/bin/bash
# Experiments for Q3.5-27B

# ============================================================
# POLITENESS
# ============================================================

# --- truthful_qa ---
python experiments/politeness.py --models Q3.5-27B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global

## --- natural_questions ---
#python experiments/politeness.py --models Q3.5-27B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global

# ============================================================
# SAFETY - POLITENESS
# ============================================================

# --- activations ---
python experiments/safety_full.py --model Q3.5-27B --style_family politeness --places prefix suffix global --harmbench_sample_size 128 --alpaca_sample_size 128 --compute_activations --run_dir results/safety/politeness_activations

# --- ASR (stage1 + stage2) ---
python experiments/safety_full.py --model Q3.5-27B --style_family politeness --places prefix suffix global --harmbench_sample_size 128 --asr_stage both --compute_asr --run_dir results/safety/politeness_asr

# ============================================================
# COT - POLITENESS
# ============================================================

python experiments/cot_reasoning_generate.py --models Q3.5-27B --dataset gsm8k --sample_size 128 --style politeness --places global --strengths -8 -4 0 4 8 --max_new_tokens 300

# ============================================================
# LENGTH VARIATION
# ============================================================

# --- truthful_qa ---
python experiments/length_variation.py --models Q3.5-27B --dataset truthful_qa --sample_size 128 --experiments all

## --- natural_questions ---
#python experiments/length_variation.py --models Q3.5-27B --dataset natural_questions --sample_size 128 --experiments all

# ============================================================
# SAFETY - LENGTH VARIATION
# ============================================================

# --- activations ---
python experiments/safety_full.py --model Q3.5-27B --style_family structured --style_name length_variation --harmbench_sample_size 128 --alpaca_sample_size 128 --compute_activations --run_dir results/safety/length_variation_activations

# --- ASR (stage1 + stage2) ---
python experiments/safety_full.py --model Q3.5-27B --style_family structured --style_name length_variation --harmbench_sample_size 128 --asr_stage both --compute_asr --run_dir results/safety/length_variation_asr

# ============================================================
# COT - LENGTH VARIATION
# ============================================================

python experiments/cot_reasoning_generate.py --models Q3.5-27B --dataset gsm8k --sample_size 128 --style length_variation --max_new_tokens 300

# ============================================================
# LETTER CASE
# ============================================================

# --- truthful_qa ---
python experiments/letter_case.py --models Q3.5-27B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global

## --- natural_questions ---
#python experiments/letter_case.py --models Q3.5-27B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global

# ============================================================
# SAFETY - LETTER CASE
# ============================================================

# --- activations ---
python experiments/safety_full.py --model Q3.5-27B --style_family surface_noise --style_name letter_case --places prefix suffix global --harmbench_sample_size 128 --alpaca_sample_size 128 --compute_activations --run_dir results/safety/letter_case_activations

# --- ASR (stage1 + stage2) ---
python experiments/safety_full.py --model Q3.5-27B --style_family surface_noise --style_name letter_case --places prefix suffix global --harmbench_sample_size 128 --asr_stage both --compute_asr --run_dir results/safety/letter_case_asr

# ============================================================
# COT - LETTER CASE
# ============================================================

python experiments/cot_reasoning_generate.py --models Q3.5-27B --dataset gsm8k --sample_size 128 --style letter_case --places global --strengths 0 25 50 100 --max_new_tokens 300

# ============================================================
# PUNCTUATION
# ============================================================

# --- truthful_qa ---
python experiments/punctuation.py --models Q3.5-27B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global

## --- natural_questions ---
#python experiments/punctuation.py --models Q3.5-27B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global

# ============================================================
# SAFETY - PUNCTUATION
# ============================================================

# --- activations ---
python experiments/safety_full.py --model Q3.5-27B --style_family surface_noise --style_name punctuation --places prefix suffix global --harmbench_sample_size 128 --alpaca_sample_size 128 --compute_activations --run_dir results/safety/punctuation_activations

# --- ASR (stage1 + stage2) ---
python experiments/safety_full.py --model Q3.5-27B --style_family surface_noise --style_name punctuation --places prefix suffix global --harmbench_sample_size 128 --asr_stage both --compute_asr --run_dir results/safety/punctuation_asr

# ============================================================
# COT - PUNCTUATION
# ============================================================

python experiments/cot_reasoning_generate.py --models Q3.5-27B --dataset gsm8k --sample_size 128 --style punctuation --places global --strengths 0 3 10 20 --max_new_tokens 300

# ============================================================
# SPACING
# ============================================================

# --- truthful_qa ---
python experiments/spacing.py --models Q3.5-27B --dataset truthful_qa --sample_size 128 --experiments all --places prefix suffix global

## --- natural_questions ---
#python experiments/spacing.py --models Q3.5-27B --dataset natural_questions --sample_size 128 --experiments all --places prefix suffix global

# ============================================================
# SAFETY - SPACING
# ============================================================

# --- activations ---
python experiments/safety_full.py --model Q3.5-27B --style_family surface_noise --style_name spacing --places prefix suffix global --harmbench_sample_size 128 --alpaca_sample_size 128 --compute_activations --run_dir results/safety/spacing_activations

# --- ASR (stage1 + stage2) ---
python experiments/safety_full.py --model Q3.5-27B --style_family surface_noise --style_name spacing --places prefix suffix global --harmbench_sample_size 128 --asr_stage both --compute_asr --run_dir results/safety/spacing_asr

# ============================================================
# COT - SPACING
# ============================================================

python experiments/cot_reasoning_generate.py --models Q3.5-27B --dataset gsm8k --sample_size 128 --style spacing --places global --strengths 0 20 50 100 --max_new_tokens 300

# ============================================================
# INTERROGATIVE VS IMPERATIVE
# ============================================================

# --- truthful_qa ---
python experiments/interrogative_vs_imperative.py --models Q3.5-27B --dataset truthful_qa --sample_size 128 --experiments all

## --- natural_questions ---
#python experiments/interrogative_vs_imperative.py --models Q3.5-27B --dataset natural_questions --sample_size 128 --experiments all

# ============================================================
# SAFETY - INTERROGATIVE VS IMPERATIVE
# ============================================================

# --- activations ---
python experiments/safety_full.py --model Q3.5-27B --style_family structured --style_name inter_vs_imper --harmbench_sample_size 128 --alpaca_sample_size 128 --compute_activations --run_dir results/safety/inter_vs_imper_activations

# --- ASR (stage1 + stage2) ---
python experiments/safety_full.py --model Q3.5-27B --style_family structured --style_name inter_vs_imper --harmbench_sample_size 128 --asr_stage both --compute_asr --run_dir results/safety/inter_vs_imper_asr

# ============================================================
# COT - INTERROGATIVE VS IMPERATIVE
# ============================================================

python experiments/cot_reasoning_generate.py --models Q3.5-27B --dataset gsm8k --sample_size 128 --style inter_vs_imper --max_new_tokens 300
