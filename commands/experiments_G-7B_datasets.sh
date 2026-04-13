#!/bin/bash
# All style experiments — G-7B
# Datasets: alpaca, simpleqa_verified, trivia_qa, hotpot_qa
# Note: natural_questions excluded for this model
# Sample size: 16
#
# Usage: bash commands/experiments_G-7B_datasets.sh [BATCH_SIZE]

BATCH_SIZE=${1:-40}

# ============================================================
# ALPACA
# ============================================================
python experiments/politeness.py --models G-7B --dataset alpaca --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/spacing.py --models G-7B --dataset alpaca --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/punctuation.py --models G-7B --dataset alpaca --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/letter_case.py --models G-7B --dataset alpaca --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/length_variation.py --models G-7B --dataset alpaca --sample_size 16 --experiments all --batch_size "$BATCH_SIZE"
python experiments/interrogative_vs_imperative.py --models G-7B --dataset alpaca --sample_size 16 --experiments all --batch_size "$BATCH_SIZE"

# ============================================================
# SIMPLEQA VERIFIED
# ============================================================
python experiments/politeness.py --models G-7B --dataset simpleqa_verified --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/spacing.py --models G-7B --dataset simpleqa_verified --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/punctuation.py --models G-7B --dataset simpleqa_verified --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/letter_case.py --models G-7B --dataset simpleqa_verified --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/length_variation.py --models G-7B --dataset simpleqa_verified --sample_size 16 --experiments all --batch_size "$BATCH_SIZE"
python experiments/interrogative_vs_imperative.py --models G-7B --dataset simpleqa_verified --sample_size 16 --experiments all --batch_size "$BATCH_SIZE"

# ============================================================
# TRIVIA QA
# ============================================================
python experiments/politeness.py --models G-7B --dataset trivia_qa --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/spacing.py --models G-7B --dataset trivia_qa --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/punctuation.py --models G-7B --dataset trivia_qa --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/letter_case.py --models G-7B --dataset trivia_qa --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/length_variation.py --models G-7B --dataset trivia_qa --sample_size 16 --experiments all --batch_size "$BATCH_SIZE"
python experiments/interrogative_vs_imperative.py --models G-7B --dataset trivia_qa --sample_size 16 --experiments all --batch_size "$BATCH_SIZE"

# ============================================================
# HOTPOT QA
# ============================================================
python experiments/politeness.py --models G-7B --dataset hotpot_qa --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/spacing.py --models G-7B --dataset hotpot_qa --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/punctuation.py --models G-7B --dataset hotpot_qa --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/letter_case.py --models G-7B --dataset hotpot_qa --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/length_variation.py --models G-7B --dataset hotpot_qa --sample_size 16 --experiments all --batch_size "$BATCH_SIZE"
python experiments/interrogative_vs_imperative.py --models G-7B --dataset hotpot_qa --sample_size 16 --experiments all --batch_size "$BATCH_SIZE"
