#!/bin/bash
# All style experiments — G4-E4B
# Datasets: natural_questions, alpaca, simpleqa_verified, trivia_qa, hotpot_qa
# Sample size: 16
#
# Usage: bash commands/experiments_G4-E4B_datasets.sh [BATCH_SIZE]

BATCH_SIZE=${1:-40}

# ============================================================
# NATURAL QUESTIONS
# ============================================================
python experiments/politeness.py --models G4-E4B --dataset natural_questions --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/spacing.py --models G4-E4B --dataset natural_questions --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/punctuation.py --models G4-E4B --dataset natural_questions --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/letter_case.py --models G4-E4B --dataset natural_questions --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/length_variation.py --models G4-E4B --dataset natural_questions --sample_size 16 --experiments all --batch_size "$BATCH_SIZE"
python experiments/interrogative_vs_imperative.py --models G4-E4B --dataset natural_questions --sample_size 16 --experiments all --batch_size "$BATCH_SIZE"

# ============================================================
# ALPACA
# ============================================================
python experiments/politeness.py --models G4-E4B --dataset alpaca --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/spacing.py --models G4-E4B --dataset alpaca --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/punctuation.py --models G4-E4B --dataset alpaca --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/letter_case.py --models G4-E4B --dataset alpaca --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/length_variation.py --models G4-E4B --dataset alpaca --sample_size 16 --experiments all --batch_size "$BATCH_SIZE"
python experiments/interrogative_vs_imperative.py --models G4-E4B --dataset alpaca --sample_size 16 --experiments all --batch_size "$BATCH_SIZE"

# ============================================================
# SIMPLEQA VERIFIED
# ============================================================
python experiments/politeness.py --models G4-E4B --dataset simpleqa_verified --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/spacing.py --models G4-E4B --dataset simpleqa_verified --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/punctuation.py --models G4-E4B --dataset simpleqa_verified --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/letter_case.py --models G4-E4B --dataset simpleqa_verified --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/length_variation.py --models G4-E4B --dataset simpleqa_verified --sample_size 16 --experiments all --batch_size "$BATCH_SIZE"
python experiments/interrogative_vs_imperative.py --models G4-E4B --dataset simpleqa_verified --sample_size 16 --experiments all --batch_size "$BATCH_SIZE"

# ============================================================
# TRIVIA QA
# ============================================================
python experiments/politeness.py --models G4-E4B --dataset trivia_qa --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/spacing.py --models G4-E4B --dataset trivia_qa --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/punctuation.py --models G4-E4B --dataset trivia_qa --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/letter_case.py --models G4-E4B --dataset trivia_qa --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/length_variation.py --models G4-E4B --dataset trivia_qa --sample_size 16 --experiments all --batch_size "$BATCH_SIZE"
python experiments/interrogative_vs_imperative.py --models G4-E4B --dataset trivia_qa --sample_size 16 --experiments all --batch_size "$BATCH_SIZE"

# ============================================================
# HOTPOT QA
# ============================================================
python experiments/politeness.py --models G4-E4B --dataset hotpot_qa --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/spacing.py --models G4-E4B --dataset hotpot_qa --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/punctuation.py --models G4-E4B --dataset hotpot_qa --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/letter_case.py --models G4-E4B --dataset hotpot_qa --sample_size 16 --experiments all --places prefix suffix global --batch_size "$BATCH_SIZE"
python experiments/length_variation.py --models G4-E4B --dataset hotpot_qa --sample_size 16 --experiments all --batch_size "$BATCH_SIZE"
python experiments/interrogative_vs_imperative.py --models G4-E4B --dataset hotpot_qa --sample_size 16 --experiments all --batch_size "$BATCH_SIZE"
