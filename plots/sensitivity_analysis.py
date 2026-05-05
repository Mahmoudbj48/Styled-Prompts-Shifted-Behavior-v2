#!/usr/bin/env python3
"""
Sensitivity analysis: how many prompts do we need for stable results?

Method
------
All variations are loaded together. For each model and sample size N ∈ {4, 8, 16,
32, 64, 128} we bootstrap 500 random subsets of N prompt-IDs and compute each
metric's mean over the sampled varied rows.

Y-axis: Metric value. One line per model with P10–P90 bootstrap band.
A vertical dashed line marks the recommended N (smallest N within 15% of the
full N=128 estimate for all models).

Usage
-----
  python plots/sensitivity_analysis.py
  python plots/sensitivity_analysis.py --out_dir results/sensitivity_analysis --n_bootstrap 500
  python plots/sensitivity_analysis.py --variations spacing punctuation letter_case
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── TruthfulQA CSV paths ───────────────────────────────────────────────────────
VARIATION_CSV_PATHS: dict[str, list[str]] = {
    "spacing": [
        "results/spacing/run_multi_truthful_qa_20260224_154406/full_results_all_models.csv",  # L3.2-3B/TruthfulQA
        "results/spacing/run_multi_natural_questions_20260416_120651/full_results_all_models.csv",  # L3.2-3B/Natural Questions
        "results/spacing/run_multi_alpaca_20260416_135049/full_results_all_models.csv",  # L3.2-3B/Alpaca
        "results/spacing/run_multi_simpleqa_verified_20260416_154600/full_results_all_models.csv",  # L3.2-3B/SimpleQA Verified
        "results/spacing/run_multi_trivia_qa_20260416_173400/full_results_all_models.csv",  # L3.2-3B/TriviaQA
        "results/spacing/run_multi_hotpot_qa_20260416_192143/full_results_all_models.csv",  # L3.2-3B/HotpotQA
        "results/spacing/run_multi_truthful_qa_20260224_164304/full_results_all_models.csv",  # L3.1-8B/TruthfulQA
        "results/spacing/run_multi_natural_questions_20260303_125434/full_results_all_models.csv",  # L3.1-8B/Natural Questions
        "results/spacing/run_multi_alpaca_20260419_221401/full_results_all_models.csv",  # L3.1-8B/Alpaca
        "results/spacing/run_multi_simpleqa_verified_20260419_233712/full_results_all_models.csv",  # L3.1-8B/SimpleQA Verified
        "results/spacing/run_multi_trivia_qa_20260420_005114/full_results_all_models.csv",  # L3.1-8B/TriviaQA
        "results/spacing/run_multi_hotpot_qa_20260420_020656/full_results_all_models.csv",  # L3.1-8B/HotpotQA
        "results/spacing/run_multi_truthful_qa_20260223_221345/full_results_all_models.csv",  # G-2B/TruthfulQA
        "results/spacing/run_multi_natural_questions_20260416_115300/full_results_all_models.csv",  # G-2B/Natural Questions
        "results/spacing/run_multi_alpaca_20260416_125945/full_results_all_models.csv",  # G-2B/Alpaca
        "results/spacing/run_multi_simpleqa_verified_20260416_140431/full_results_all_models.csv",  # G-2B/SimpleQA Verified
        "results/spacing/run_multi_trivia_qa_20260416_144318/full_results_all_models.csv",  # G-2B/TriviaQA
        "results/spacing/run_multi_hotpot_qa_20260416_152216/full_results_all_models.csv",  # G-2B/HotpotQA
        "results/spacing/run_multi_truthful_qa_20260409_163159/full_results_all_models.csv",  # G4-E4B/TruthfulQA
        "results/spacing/run_multi_natural_questions_20260419_163119/full_results_all_models.csv",  # G4-E4B/Natural Questions
        "results/spacing/run_multi_alpaca_20260419_183244/full_results_all_models.csv",  # G4-E4B/Alpaca
        "results/spacing/run_multi_simpleqa_verified_20260419_204433/full_results_all_models.csv",  # G4-E4B/SimpleQA Verified
        "results/spacing/run_multi_trivia_qa_20260419_224944/full_results_all_models.csv",  # G4-E4B/TriviaQA
        "results/spacing/run_multi_hotpot_qa_20260420_004659/full_results_all_models.csv",  # G4-E4B/HotpotQA
        "results/spacing/run_multi_truthful_qa_20260224_173202/full_results_all_models.csv",  # G-7B/TruthfulQA
        "results/spacing/run_multi_natural_questions_20260303_130138/full_results_all_models.csv",  # G-7B/Natural Questions
        "results/spacing/run_multi_alpaca_20260420_121817/full_results_all_models.csv",  # G-7B/Alpaca
        "results/spacing/run_multi_simpleqa_verified_20260420_125300/full_results_all_models.csv",  # G-7B/SimpleQA Verified
        "results/spacing/run_multi_trivia_qa_20260420_133758/full_results_all_models.csv",  # G-7B/TriviaQA
        "results/spacing/run_multi_hotpot_qa_20260420_141011/full_results_all_models.csv",  # G-7B/HotpotQA
        "results/spacing/run_multi_truthful_qa_20260224_160542/full_results_all_models.csv",  # Q2.5-1.5B/TruthfulQA
        "results/spacing/run_multi_natural_questions_20260416_115143/full_results_all_models.csv",  # Q2.5-1.5B/Natural Questions
        "results/spacing/run_multi_alpaca_20260416_123332/full_results_all_models.csv",  # Q2.5-1.5B/Alpaca
        "results/spacing/run_multi_simpleqa_verified_20260416_134342/full_results_all_models.csv",  # Q2.5-1.5B/SimpleQA Verified
        "results/spacing/run_multi_trivia_qa_20260416_144844/full_results_all_models.csv",  # Q2.5-1.5B/TriviaQA
        "results/spacing/run_multi_hotpot_qa_20260416_152741/full_results_all_models.csv",  # Q2.5-1.5B/HotpotQA
        "results/spacing/run_multi_truthful_qa_20260224_181508/full_results_all_models.csv",  # Q2.5-7B/TruthfulQA
        "results/spacing/run_multi_natural_questions_20260303_130609/full_results_all_models.csv",  # Q2.5-7B/Natural Questions
        "results/spacing/run_multi_alpaca_20260419_165120/full_results_all_models.csv",  # Q2.5-7B/Alpaca
        "results/spacing/run_multi_simpleqa_verified_20260419_180305/full_results_all_models.csv",  # Q2.5-7B/SimpleQA Verified
        "results/spacing/run_multi_trivia_qa_20260419_201113/full_results_all_models.csv",  # Q2.5-7B/TriviaQA
        "results/spacing/run_multi_hotpot_qa_20260419_212231/full_results_all_models.csv",  # Q2.5-7B/HotpotQA
        "results/spacing/run_multi_truthful_qa_20260410_094504/full_results_all_models.csv",  # Q3.5-9B/TruthfulQA
        "results/spacing/run_multi_natural_questions_20260419_170320/full_results_all_models.csv",  # Q3.5-9B/Natural Questions
        "results/spacing/run_multi_alpaca_20260419_191509/full_results_all_models.csv",  # Q3.5-9B/Alpaca
        "results/spacing/run_multi_simpleqa_verified_20260419_211628/full_results_all_models.csv",  # Q3.5-9B/SimpleQA Verified
        "results/spacing/run_multi_trivia_qa_20260420_013355/full_results_all_models.csv",  # Q3.5-9B/TriviaQA
        "results/spacing/run_multi_hotpot_qa_20260420_032953/full_results_all_models.csv",  # Q3.5-9B/HotpotQA
    ],
    "punctuation": [
        "results/punctuation/run_multi_truthful_qa_20260222_215846/full_results_all_models.csv",  # L3.2-3B/TruthfulQA
        "results/punctuation/run_multi_natural_questions_20260416_122203/full_results_all_models.csv",  # L3.2-3B/Natural Questions
        "results/punctuation/run_multi_alpaca_20260416_141001/full_results_all_models.csv",  # L3.2-3B/Alpaca
        "results/punctuation/run_multi_simpleqa_verified_20260416_160338/full_results_all_models.csv",  # L3.2-3B/SimpleQA Verified
        "results/punctuation/run_multi_trivia_qa_20260416_175102/full_results_all_models.csv",  # L3.2-3B/TriviaQA
        "results/punctuation/run_multi_hotpot_qa_20260416_193901/full_results_all_models.csv",  # L3.2-3B/HotpotQA
        "results/punctuation/run_multi_truthful_qa_20260224_185252/full_results_all_models.csv",  # L3.1-8B/TruthfulQA
        "results/punctuation/run_multi_natural_questions_20260303_131249/full_results_all_models.csv",  # L3.1-8B/Natural Questions
        "results/punctuation/run_multi_alpaca_20260419_222538/full_results_all_models.csv",  # L3.1-8B/Alpaca
        "results/punctuation/run_multi_simpleqa_verified_20260419_234509/full_results_all_models.csv",  # L3.1-8B/SimpleQA Verified
        "results/punctuation/run_multi_trivia_qa_20260420_010134/full_results_all_models.csv",  # L3.1-8B/TriviaQA
        "results/punctuation/run_multi_hotpot_qa_20260420_021310/full_results_all_models.csv",  # L3.1-8B/HotpotQA
        "results/punctuation/run_multi_truthful_qa_20260223_215814/full_results_all_models.csv",  # G-2B/TruthfulQA
        "results/punctuation/run_multi_natural_questions_20260416_120118/full_results_all_models.csv",  # G-2B/Natural Questions
        "results/punctuation/run_multi_alpaca_20260416_131215/full_results_all_models.csv",  # G-2B/Alpaca
        "results/punctuation/run_multi_simpleqa_verified_20260416_140907/full_results_all_models.csv",  # G-2B/SimpleQA Verified
        "results/punctuation/run_multi_trivia_qa_20260416_144756/full_results_all_models.csv",  # G-2B/TriviaQA
        "results/punctuation/run_multi_hotpot_qa_20260416_152608/full_results_all_models.csv",  # G-2B/HotpotQA
        "results/punctuation/run_multi_truthful_qa_20260409_143341/full_results_all_models.csv",  # G4-E4B/TruthfulQA
        "results/punctuation/run_multi_natural_questions_20260419_164334/full_results_all_models.csv",  # G4-E4B/Natural Questions
        "results/punctuation/run_multi_alpaca_20260419_185046/full_results_all_models.csv",  # G4-E4B/Alpaca
        "results/punctuation/run_multi_simpleqa_verified_20260419_210017/full_results_all_models.csv",  # G4-E4B/SimpleQA Verified
        "results/punctuation/run_multi_trivia_qa_20260419_230437/full_results_all_models.csv",  # G4-E4B/TriviaQA
        "results/punctuation/run_multi_hotpot_qa_20260420_010121/full_results_all_models.csv",  # G4-E4B/HotpotQA
        "results/punctuation/run_multi_truthful_qa_20260224_193108/full_results_all_models.csv",  # G-7B/TruthfulQA
        "results/punctuation/run_multi_natural_questions_20260303_131952/full_results_all_models.csv",  # G-7B/Natural Questions
        "results/punctuation/run_multi_alpaca_20260420_121919/full_results_all_models.csv",  # G-7B/Alpaca
        "results/punctuation/run_multi_simpleqa_verified_20260420_125733/full_results_all_models.csv",  # G-7B/SimpleQA Verified
        "results/punctuation/run_multi_trivia_qa_20260420_133852/full_results_all_models.csv",  # G-7B/TriviaQA
        "results/punctuation/run_multi_hotpot_qa_20260420_141118/full_results_all_models.csv",  # G-7B/HotpotQA
        "results/punctuation/run_multi_truthful_qa_20260224_162341/full_results_all_models.csv",  # Q2.5-1.5B/TruthfulQA
        "results/punctuation/run_multi_natural_questions_20260416_115228/full_results_all_models.csv",  # Q2.5-1.5B/Natural Questions
        "results/punctuation/run_multi_alpaca_20260416_124414/full_results_all_models.csv",  # Q2.5-1.5B/Alpaca
        "results/punctuation/run_multi_simpleqa_verified_20260416_135309/full_results_all_models.csv",  # Q2.5-1.5B/SimpleQA Verified
        "results/punctuation/run_multi_trivia_qa_20260416_144929/full_results_all_models.csv",  # Q2.5-1.5B/TriviaQA
        "results/punctuation/run_multi_hotpot_qa_20260416_152829/full_results_all_models.csv",  # Q2.5-1.5B/HotpotQA
        "results/punctuation/run_multi_truthful_qa_20260224_195351/full_results_all_models.csv",  # Q2.5-7B/TruthfulQA
        "results/punctuation/run_multi_natural_questions_20260303_132435/full_results_all_models.csv",  # Q2.5-7B/Natural Questions
        "results/punctuation/run_multi_alpaca_20260419_165229/full_results_all_models.csv",  # Q2.5-7B/Alpaca
        "results/punctuation/run_multi_simpleqa_verified_20260419_182107/full_results_all_models.csv",  # Q2.5-7B/SimpleQA Verified
        "results/punctuation/run_multi_trivia_qa_20260419_201220/full_results_all_models.csv",  # Q2.5-7B/TriviaQA
        "results/punctuation/run_multi_hotpot_qa_20260419_212335/full_results_all_models.csv",  # Q2.5-7B/HotpotQA
        "results/punctuation/run_multi_truthful_qa_20260410_042041/full_results_all_models.csv",  # Q3.5-9B/TruthfulQA
        "results/punctuation/run_multi_natural_questions_20260419_170558/full_results_all_models.csv",  # Q3.5-9B/Natural Questions
        "results/punctuation/run_multi_alpaca_20260419_191827/full_results_all_models.csv",  # Q3.5-9B/Alpaca
        "results/punctuation/run_multi_simpleqa_verified_20260419_220035/full_results_all_models.csv",  # Q3.5-9B/SimpleQA Verified
        "results/punctuation/run_multi_trivia_qa_20260420_013848/full_results_all_models.csv",  # Q3.5-9B/TriviaQA
        "results/punctuation/run_multi_hotpot_qa_20260420_033255/full_results_all_models.csv",  # Q3.5-9B/HotpotQA
    ],
    "letter_case": [
        "results/letter_case/run_multi_truthful_qa_20260223_143708/full_results_all_models.csv",  # L3.2-3B/TruthfulQA
        "results/letter_case/run_multi_natural_questions_20260416_123726/full_results_all_models.csv",  # L3.2-3B/Natural Questions
        "results/letter_case/run_multi_alpaca_20260416_142919/full_results_all_models.csv",  # L3.2-3B/Alpaca
        "results/letter_case/run_multi_simpleqa_verified_20260416_162122/full_results_all_models.csv",  # L3.2-3B/SimpleQA Verified
        "results/letter_case/run_multi_trivia_qa_20260416_180823/full_results_all_models.csv",  # L3.2-3B/TriviaQA
        "results/letter_case/run_multi_hotpot_qa_20260416_195607/full_results_all_models.csv",  # L3.2-3B/HotpotQA
        "results/letter_case/run_multi_truthful_qa_20260224_213628/full_results_all_models.csv",  # L3.1-8B/TruthfulQA
        "results/letter_case/run_multi_natural_questions_20260228_231937/full_results_all_models.csv",  # L3.1-8B/Natural Questions
        "results/letter_case/run_multi_alpaca_20260419_223724/full_results_all_models.csv",  # L3.1-8B/Alpaca
        "results/letter_case/run_multi_simpleqa_verified_20260419_235321/full_results_all_models.csv",  # L3.1-8B/SimpleQA Verified
        "results/letter_case/run_multi_trivia_qa_20260420_011135/full_results_all_models.csv",  # L3.1-8B/TriviaQA
        "results/letter_case/run_multi_hotpot_qa_20260420_022050/full_results_all_models.csv",  # L3.1-8B/HotpotQA
        "results/letter_case/run_multi_truthful_qa_20260223_152205/full_results_all_models.csv",  # G-2B/TruthfulQA
        "results/letter_case/run_multi_natural_questions_20260416_120945/full_results_all_models.csv",  # G-2B/Natural Questions
        "results/letter_case/run_multi_alpaca_20260416_132452/full_results_all_models.csv",  # G-2B/Alpaca
        "results/letter_case/run_multi_simpleqa_verified_20260416_141359/full_results_all_models.csv",  # G-2B/SimpleQA Verified
        "results/letter_case/run_multi_trivia_qa_20260416_145328/full_results_all_models.csv",  # G-2B/TriviaQA
        "results/letter_case/run_multi_hotpot_qa_20260416_153006/full_results_all_models.csv",  # G-2B/HotpotQA
        "results/letter_case/run_multi_truthful_qa_20260409_121907/full_results_all_models.csv",  # G4-E4B/TruthfulQA
        "results/letter_case/run_multi_natural_questions_20260419_165551/full_results_all_models.csv",  # G4-E4B/Natural Questions
        "results/letter_case/run_multi_alpaca_20260419_190856/full_results_all_models.csv",  # G4-E4B/Alpaca
        "results/letter_case/run_multi_simpleqa_verified_20260419_211753/full_results_all_models.csv",  # G4-E4B/SimpleQA Verified
        "results/letter_case/run_multi_trivia_qa_20260419_231934/full_results_all_models.csv",  # G4-E4B/TriviaQA
        "results/letter_case/run_multi_hotpot_qa_20260420_011536/full_results_all_models.csv",  # G4-E4B/HotpotQA
        "results/letter_case/run_multi_truthful_qa_20260225_005805/full_results_all_models.csv",  # G-7B/TruthfulQA
        "results/letter_case/run_multi_natural_questions_20260228_231743/full_results_all_models.csv",  # G-7B/Natural Questions
        "results/letter_case/run_multi_alpaca_20260420_122022/full_results_all_models.csv",  # G-7B/Alpaca
        "results/letter_case/run_multi_simpleqa_verified_20260420_130214/full_results_all_models.csv",  # G-7B/SimpleQA Verified
        "results/letter_case/run_multi_trivia_qa_20260420_133949/full_results_all_models.csv",  # G-7B/TriviaQA
        "results/letter_case/run_multi_hotpot_qa_20260420_141215/full_results_all_models.csv",  # G-7B/HotpotQA
        "results/letter_case/run_multi_truthful_qa_20260223_214923/full_results_all_models.csv",  # Q2.5-1.5B/TruthfulQA
        "results/letter_case/run_multi_natural_questions_20260416_115315/full_results_all_models.csv",  # Q2.5-1.5B/Natural Questions
        "results/letter_case/run_multi_alpaca_20260416_125459/full_results_all_models.csv",  # Q2.5-1.5B/Alpaca
        "results/letter_case/run_multi_simpleqa_verified_20260416_140234/full_results_all_models.csv",  # Q2.5-1.5B/SimpleQA Verified
        "results/letter_case/run_multi_trivia_qa_20260416_145014/full_results_all_models.csv",  # Q2.5-1.5B/TriviaQA
        "results/letter_case/run_multi_hotpot_qa_20260416_152913/full_results_all_models.csv",  # Q2.5-1.5B/HotpotQA
        "results/letter_case/run_multi_truthful_qa_20260225_032721/full_results_all_models.csv",  # Q2.5-7B/TruthfulQA
        "results/letter_case/run_multi_natural_questions_20260302_171030/full_results_all_models.csv",  # Q2.5-7B/Natural Questions
        "results/letter_case/run_multi_alpaca_20260419_165336/full_results_all_models.csv",  # Q2.5-7B/Alpaca
        "results/letter_case/run_multi_simpleqa_verified_20260419_183912/full_results_all_models.csv",  # Q2.5-7B/SimpleQA Verified
        "results/letter_case/run_multi_trivia_qa_20260419_201326/full_results_all_models.csv",  # Q2.5-7B/TriviaQA
        "results/letter_case/run_multi_hotpot_qa_20260419_212438/full_results_all_models.csv",  # Q2.5-7B/HotpotQA
        "results/letter_case/run_multi_truthful_qa_20260409_222404/full_results_all_models.csv",  # Q3.5-9B/TruthfulQA
        "results/letter_case/run_multi_natural_questions_20260419_170848/full_results_all_models.csv",  # Q3.5-9B/Natural Questions
        "results/letter_case/run_multi_alpaca_20260419_192146/full_results_all_models.csv",  # Q3.5-9B/Alpaca
        "results/letter_case/run_multi_simpleqa_verified_20260419_224319/full_results_all_models.csv",  # Q3.5-9B/SimpleQA Verified
        "results/letter_case/run_multi_trivia_qa_20260420_014154/full_results_all_models.csv",  # Q3.5-9B/TriviaQA
        "results/letter_case/run_multi_hotpot_qa_20260420_033559/full_results_all_models.csv",  # Q3.5-9B/HotpotQA
    ],
    "politeness": [
        "results/politeness/run_multi_truthful_qa_20260222_113341/full_results_all_models.csv",  # L3.2-3B/TruthfulQA
        "results/politeness/run_multi_natural_questions_20260416_112513/full_results_all_models.csv",  # L3.2-3B/Natural Questions
        "results/politeness/run_multi_alpaca_20260416_130117/full_results_all_models.csv",  # L3.2-3B/Alpaca
        "results/politeness/run_multi_simpleqa_verified_20260416_145922/full_results_all_models.csv",  # L3.2-3B/SimpleQA Verified
        "results/politeness/run_multi_trivia_qa_20260416_164813/full_results_all_models.csv",  # L3.2-3B/TriviaQA
        "results/politeness/run_multi_hotpot_qa_20260416_183545/full_results_all_models.csv",  # L3.2-3B/HotpotQA
        "results/politeness/run_multi_truthful_qa_20260222_132615/full_results_all_models.csv",  # L3.1-8B/TruthfulQA
        "results/politeness/run_multi_natural_questions_20260223_230428/full_results_all_models.csv",  # L3.1-8B/Natural Questions
        "results/politeness/run_multi_alpaca_20260419_212741/full_results_all_models.csv",  # L3.1-8B/Alpaca
        "results/politeness/run_multi_simpleqa_verified_20260419_225606/full_results_all_models.csv",  # L3.1-8B/SimpleQA Verified
        "results/politeness/run_multi_trivia_qa_20260420_000652/full_results_all_models.csv",  # L3.1-8B/TriviaQA
        "results/politeness/run_multi_hotpot_qa_20260420_012947/full_results_all_models.csv",  # L3.1-8B/HotpotQA
        "results/politeness/run_multi_truthful_qa_20260221_112112/full_results_all_models.csv",  # G-2B/TruthfulQA
        "results/politeness/run_multi_natural_questions_20260419_151622/full_results_all_models.csv",  # G-2B/Natural Questions
        "results/politeness/run_multi_alpaca_20260416_122253/full_results_all_models.csv",  # G-2B/Alpaca
        "results/politeness/run_multi_simpleqa_verified_20260416_134405/full_results_all_models.csv",  # G-2B/SimpleQA Verified
        "results/politeness/run_multi_trivia_qa_20260416_142157/full_results_all_models.csv",  # G-2B/TriviaQA
        "results/politeness/run_multi_hotpot_qa_20260416_150300/full_results_all_models.csv",  # G-2B/HotpotQA
        "results/politeness/run_multi_truthful_qa_20260409_021222/full_results_all_models.csv",  # G4-E4B/TruthfulQA
        "results/politeness/run_multi_natural_questions_20260419_152533/full_results_all_models.csv",  # G4-E4B/Natural Questions
        "results/politeness/run_multi_alpaca_20260419_171545/full_results_all_models.csv",  # G4-E4B/Alpaca
        "results/politeness/run_multi_simpleqa_verified_20260419_193704/full_results_all_models.csv",  # G4-E4B/SimpleQA Verified
        "results/politeness/run_multi_trivia_qa_20260419_214443/full_results_all_models.csv",  # G4-E4B/TriviaQA
        "results/politeness/run_multi_hotpot_qa_20260419_234339/full_results_all_models.csv",  # G4-E4B/HotpotQA
        "results/politeness/run_multi_truthful_qa_20260222_150512/full_results_all_models.csv",  # G-7B/TruthfulQA
        "results/politeness/run_multi_natural_questions_20260422_000653/full_results_all_models.csv",  # G-7B / Natural Questions  — run pending
        "results/politeness/run_multi_alpaca_20260420_112748/full_results_all_models.csv",  # G-7B/Alpaca
        "results/politeness/run_multi_simpleqa_verified_20260420_122244/full_results_all_models.csv",  # G-7B/SimpleQA Verified
        "results/politeness/run_multi_trivia_qa_20260420_131019/full_results_all_models.csv",  # G-7B/TriviaQA
        "results/politeness/run_multi_hotpot_qa_20260420_134207/full_results_all_models.csv",  # G-7B/HotpotQA
        "results/politeness/run_multi_truthful_qa_20260221_145354/full_results_all_models.csv",  # Q2.5-1.5B/TruthfulQA
        "results/politeness/run_multi_natural_questions_20260415_192823/full_results_all_models.csv",  # Q2.5-1.5B/Natural Questions
        "results/politeness/run_multi_alpaca_20260416_120003/full_results_all_models.csv",  # Q2.5-1.5B/Alpaca
        "results/politeness/run_multi_simpleqa_verified_20260416_131328/full_results_all_models.csv",  # Q2.5-1.5B/SimpleQA Verified
        "results/politeness/run_multi_trivia_qa_20260416_141914/full_results_all_models.csv",  # Q2.5-1.5B/TriviaQA
        "results/politeness/run_multi_hotpot_qa_20260416_145752/full_results_all_models.csv",  # Q2.5-1.5B/HotpotQA
        "results/politeness/run_multi_truthful_qa_20260223_120916/full_results_all_models.csv",  # Q2.5-7B/TruthfulQA
        "results/politeness/run_multi_natural_questions_20260422_000930/full_results_all_models.csv",# Q2.5-7B / Natural Questions  — run pending
        "results/politeness/run_multi_alpaca_20260419_153633/full_results_all_models.csv",  # Q2.5-7B/Alpaca
        "results/politeness/run_multi_simpleqa_verified_20260419_165609/full_results_all_models.csv",  # Q2.5-7B/SimpleQA Verified
        "results/politeness/run_multi_trivia_qa_20260419_190720/full_results_all_models.csv",  # Q2.5-7B/TriviaQA
        "results/politeness/run_multi_hotpot_qa_20260419_201554/full_results_all_models.csv",  # Q2.5-7B/HotpotQA
        "results/politeness/run_multi_truthful_qa_20260409_021310/full_results_all_models.csv",  # Q3.5-9B/TruthfulQA
        "results/politeness/run_multi_natural_questions_20260419_152502/full_results_all_models.csv",  # Q3.5-9B/Natural Questions
        "results/politeness/run_multi_alpaca_20260419_171348/full_results_all_models.csv",  # Q3.5-9B/Alpaca
        "results/politeness/run_multi_simpleqa_verified_20260419_192716/full_results_all_models.csv",  # Q3.5-9B/SimpleQA Verified
        "results/politeness/run_multi_trivia_qa_20260419_234941/full_results_all_models.csv",  # Q3.5-9B/TriviaQA
        "results/politeness/run_multi_hotpot_qa_20260420_014716/full_results_all_models.csv",  # Q3.5-9B/HotpotQA
    ],
    "length_variation": [
        "results/length_variation/run_multi_truthful_qa_20260304_055338/full_results_all_models.csv",  # L3.2-3B/TruthfulQA, L3.1-8B/TruthfulQA, G-2B/TruthfulQA, G-7B/TruthfulQA, Q2.5-1.5B/TruthfulQA, Q2.5-7B/TruthfulQA
        "results/length_variation/run_multi_natural_questions_20260416_125304/full_results_all_models.csv",  # L3.2-3B/Natural Questions
        "results/length_variation/run_multi_alpaca_20260416_144937/full_results_all_models.csv",  # L3.2-3B/Alpaca
        "results/length_variation/run_multi_simpleqa_verified_20260416_163938/full_results_all_models.csv",  # L3.2-3B/SimpleQA Verified
        "results/length_variation/run_multi_trivia_qa_20260416_182633/full_results_all_models.csv",  # L3.2-3B/TriviaQA
        "results/length_variation/run_multi_hotpot_qa_20260416_201351/full_results_all_models.csv",  # L3.2-3B/HotpotQA
        "results/length_variation/run_multi_natural_questions_20260304_024838/full_results_all_models.csv",  # L3.1-8B/Natural Questions, G-7B/Natural Questions, Q2.5-7B/Natural Questions
        "results/length_variation/run_multi_alpaca_20260419_224942/full_results_all_models.csv",  # L3.1-8B/Alpaca
        "results/length_variation/run_multi_simpleqa_verified_20260420_000154/full_results_all_models.csv",  # L3.1-8B/SimpleQA Verified
        "results/length_variation/run_multi_trivia_qa_20260420_012223/full_results_all_models.csv",  # L3.1-8B/TriviaQA
        "results/length_variation/run_multi_hotpot_qa_20260420_022858/full_results_all_models.csv",  # L3.1-8B/HotpotQA
        "results/length_variation/run_multi_natural_questions_20260416_121815/full_results_all_models.csv",  # G-2B/Natural Questions
        "results/length_variation/run_multi_alpaca_20260416_133747/full_results_all_models.csv",  # G-2B/Alpaca
        "results/length_variation/run_multi_simpleqa_verified_20260416_141915/full_results_all_models.csv",  # G-2B/SimpleQA Verified
        "results/length_variation/run_multi_trivia_qa_20260416_145958/full_results_all_models.csv",  # G-2B/TriviaQA
        "results/length_variation/run_multi_hotpot_qa_20260416_153436/full_results_all_models.csv",  # G-2B/HotpotQA
        "results/length_variation/run_multi_truthful_qa_20260409_100412/full_results_all_models.csv",  # G4-E4B/TruthfulQA
        "results/length_variation/run_multi_natural_questions_20260419_170824/full_results_all_models.csv",  # G4-E4B/Natural Questions
        "results/length_variation/run_multi_alpaca_20260419_192733/full_results_all_models.csv",  # G4-E4B/Alpaca
        "results/length_variation/run_multi_simpleqa_verified_20260419_213456/full_results_all_models.csv",  # G4-E4B/SimpleQA Verified
        "results/length_variation/run_multi_trivia_qa_20260419_233509/full_results_all_models.csv",  # G4-E4B/TriviaQA
        "results/length_variation/run_multi_hotpot_qa_20260420_013019/full_results_all_models.csv",  # G4-E4B/HotpotQA
        "results/length_variation/run_multi_alpaca_20260420_122123/full_results_all_models.csv",  # G-7B/Alpaca
        "results/length_variation/run_multi_simpleqa_verified_20260420_130706/full_results_all_models.csv",  # G-7B/SimpleQA Verified
        "results/length_variation/run_multi_trivia_qa_20260420_134050/full_results_all_models.csv",  # G-7B/TriviaQA
        "results/length_variation/run_multi_hotpot_qa_20260420_141314/full_results_all_models.csv",  # G-7B/HotpotQA
        "results/length_variation/run_multi_natural_questions_20260416_115359/full_results_all_models.csv",  # Q2.5-1.5B/Natural Questions
        "results/length_variation/run_multi_alpaca_20260416_130559/full_results_all_models.csv",  # Q2.5-1.5B/Alpaca
        "results/length_variation/run_multi_simpleqa_verified_20260416_141211/full_results_all_models.csv",  # Q2.5-1.5B/SimpleQA Verified
        "results/length_variation/run_multi_trivia_qa_20260416_145058/full_results_all_models.csv",  # Q2.5-1.5B/TriviaQA
        "results/length_variation/run_multi_hotpot_qa_20260416_153003/full_results_all_models.csv",  # Q2.5-1.5B/HotpotQA
        "results/length_variation/run_multi_alpaca_20260419_165444/full_results_all_models.csv",  # Q2.5-7B/Alpaca
        "results/length_variation/run_multi_simpleqa_verified_20260419_185735/full_results_all_models.csv",  # Q2.5-7B/SimpleQA Verified
        "results/length_variation/run_multi_trivia_qa_20260419_201433/full_results_all_models.csv",  # Q2.5-7B/TriviaQA
        "results/length_variation/run_multi_hotpot_qa_20260419_212544/full_results_all_models.csv",  # Q2.5-7B/HotpotQA
        "results/length_variation/run_multi_truthful_qa_20260409_164416/full_results_all_models.csv",  # Q3.5-9B/TruthfulQA
        "results/length_variation/run_multi_natural_questions_20260419_171147/full_results_all_models.csv",  # Q3.5-9B/Natural Questions
        "results/length_variation/run_multi_alpaca_20260419_192455/full_results_all_models.csv",  # Q3.5-9B/Alpaca
        "results/length_variation/run_multi_simpleqa_verified_20260419_232717/full_results_all_models.csv",  # Q3.5-9B/SimpleQA Verified
        "results/length_variation/run_multi_trivia_qa_20260420_014456/full_results_all_models.csv",  # Q3.5-9B/TriviaQA
        "results/length_variation/run_multi_hotpot_qa_20260420_033905/full_results_all_models.csv",  # Q3.5-9B/HotpotQA
    ],
    "inter_vs_imper": [
        "results/inter_vs_imper/run_multi_truthful_qa_20260304_165515/full_results_all_models.csv",  # L3.2-3B/TruthfulQA, L3.1-8B/TruthfulQA, G-2B/TruthfulQA, G-7B/TruthfulQA, Q2.5-1.5B/TruthfulQA, Q2.5-7B/TruthfulQA
        "results/inter_vs_imper/run_multi_natural_questions_20260416_125828/full_results_all_models.csv",  # L3.2-3B/Natural Questions
        "results/inter_vs_imper/run_multi_alpaca_20260416_145603/full_results_all_models.csv",  # L3.2-3B/Alpaca
        "results/inter_vs_imper/run_multi_simpleqa_verified_20260416_164503/full_results_all_models.csv",  # L3.2-3B/SimpleQA Verified
        "results/inter_vs_imper/run_multi_trivia_qa_20260416_183238/full_results_all_models.csv",  # L3.2-3B/TriviaQA
        "results/inter_vs_imper/run_multi_hotpot_qa_20260416_201952/full_results_all_models.csv",  # L3.2-3B/HotpotQA
        "results/inter_vs_imper/run_multi_natural_questions_20260304_175953/full_results_all_models.csv",  # L3.1-8B/Natural Questions, G-7B/Natural Questions, Q2.5-7B/Natural Questions
        "results/inter_vs_imper/run_multi_alpaca_20260419_225350/full_results_all_models.csv",  # L3.1-8B/Alpaca
        "results/inter_vs_imper/run_multi_simpleqa_verified_20260420_000500/full_results_all_models.csv",  # L3.1-8B/SimpleQA Verified
        "results/inter_vs_imper/run_multi_trivia_qa_20260420_012625/full_results_all_models.csv",  # L3.1-8B/TriviaQA
        "results/inter_vs_imper/run_multi_hotpot_qa_20260420_023148/full_results_all_models.csv",  # L3.1-8B/HotpotQA
        "results/inter_vs_imper/run_multi_natural_questions_20260416_122117/full_results_all_models.csv",  # G-2B/Natural Questions
        "results/inter_vs_imper/run_multi_alpaca_20260416_134154/full_results_all_models.csv",  # G-2B/Alpaca
        "results/inter_vs_imper/run_multi_simpleqa_verified_20260416_142100/full_results_all_models.csv",  # G-2B/SimpleQA Verified
        "results/inter_vs_imper/run_multi_trivia_qa_20260416_150150/full_results_all_models.csv",  # G-2B/TriviaQA
        "results/inter_vs_imper/run_multi_hotpot_qa_20260416_153606/full_results_all_models.csv",  # G-2B/HotpotQA
        "results/inter_vs_imper/run_multi_truthful_qa_20260409_183536/full_results_all_models.csv",  # G4-E4B/TruthfulQA
        "results/inter_vs_imper/run_multi_natural_questions_20260419_171304/full_results_all_models.csv",  # G4-E4B/Natural Questions
        "results/inter_vs_imper/run_multi_alpaca_20260419_193341/full_results_all_models.csv",  # G4-E4B/Alpaca
        "results/inter_vs_imper/run_multi_simpleqa_verified_20260419_214140/full_results_all_models.csv",  # G4-E4B/SimpleQA Verified
        "results/inter_vs_imper/run_multi_trivia_qa_20260419_234037/full_results_all_models.csv",  # G4-E4B/TriviaQA
        "results/inter_vs_imper/run_multi_hotpot_qa_20260420_013545/full_results_all_models.csv",  # G4-E4B/HotpotQA
        "results/inter_vs_imper/run_multi_alpaca_20260420_122202/full_results_all_models.csv",  # G-7B/Alpaca
        "results/inter_vs_imper/run_multi_simpleqa_verified_20260420_130901/full_results_all_models.csv",  # G-7B/SimpleQA Verified
        "results/inter_vs_imper/run_multi_trivia_qa_20260420_134128/full_results_all_models.csv",  # G-7B/TriviaQA
        "results/inter_vs_imper/run_multi_hotpot_qa_20260420_141403/full_results_all_models.csv",  # G-7B/HotpotQA
        "results/inter_vs_imper/run_multi_natural_questions_20260416_115804/full_results_all_models.csv",  # Q2.5-1.5B/Natural Questions
        "results/inter_vs_imper/run_multi_alpaca_20260416_131107/full_results_all_models.csv",  # Q2.5-1.5B/Alpaca
        "results/inter_vs_imper/run_multi_simpleqa_verified_20260416_141652/full_results_all_models.csv",  # Q2.5-1.5B/SimpleQA Verified
        "results/inter_vs_imper/run_multi_trivia_qa_20260416_145539/full_results_all_models.csv",  # Q2.5-1.5B/TriviaQA
        "results/inter_vs_imper/run_multi_hotpot_qa_20260416_153454/full_results_all_models.csv",  # Q2.5-1.5B/HotpotQA
        "results/inter_vs_imper/run_multi_alpaca_20260419_165528/full_results_all_models.csv",  # Q2.5-7B/Alpaca
        "results/inter_vs_imper/run_multi_simpleqa_verified_20260419_190354/full_results_all_models.csv",  # Q2.5-7B/SimpleQA Verified
        "results/inter_vs_imper/run_multi_trivia_qa_20260419_201515/full_results_all_models.csv",  # Q2.5-7B/TriviaQA
        "results/inter_vs_imper/run_multi_hotpot_qa_20260419_212628/full_results_all_models.csv",  # Q2.5-7B/HotpotQA
        "results/inter_vs_imper/run_multi_truthful_qa_20260410_151032/full_results_all_models.csv",  # Q3.5-9B/TruthfulQA
        "results/inter_vs_imper/run_multi_natural_questions_20260419_171302/full_results_all_models.csv",  # Q3.5-9B/Natural Questions
        "results/inter_vs_imper/run_multi_alpaca_20260419_192620/full_results_all_models.csv",  # Q3.5-9B/Alpaca
        "results/inter_vs_imper/run_multi_simpleqa_verified_20260419_234155/full_results_all_models.csv",  # Q3.5-9B/SimpleQA Verified
        "results/inter_vs_imper/run_multi_trivia_qa_20260420_014625/full_results_all_models.csv",  # Q3.5-9B/TriviaQA
        "results/inter_vs_imper/run_multi_hotpot_qa_20260420_034027/full_results_all_models.csv",  # Q3.5-9B/HotpotQA
    ],
}

# ── CoT run directories (results_with_cot_analysis.csv inside each) ───────────
COT_RUN_DIRS: dict[str, list[str]] = {
    "spacing": [
        "results/cot_responses/run_gsm8k_spacing_20260302_190013",
        "results/cot_responses/run_gsm8k_spacing_20260302_195511",
        "results/cot_responses/run_gsm8k_spacing_20260303_133116",
        "results/cot_responses/run_gsm8k_spacing_20260303_192826",
        "results/cot_responses/run_gsm8k_spacing_20260303_210133",
        "results/cot_responses/run_gsm8k_spacing_20260304_015204",
        # G4-E4B, Q3.5-9B
        "results/cot_responses/run_gsm8k_spacing_20260409_172544",
        "results/cot_responses/run_gsm8k_spacing_20260410_123434",
    ],
    "punctuation": [
        "results/cot_responses/run_gsm8k_punctuation_20260302_211146",
        "results/cot_responses/run_gsm8k_punctuation_20260302_214900",
        "results/cot_responses/run_gsm8k_punctuation_20260303_150618",
        "results/cot_responses/run_gsm8k_punctuation_20260303_195401",
        "results/cot_responses/run_gsm8k_punctuation_20260303_221211",
        "results/cot_responses/run_gsm8k_punctuation_20260304_020806",
        # G4-E4B, Q3.5-9B
        "results/cot_responses/run_gsm8k_punctuation_20260409_152737",
        "results/cot_responses/run_gsm8k_punctuation_20260410_071913",
    ],
    "letter_case": [
        "results/cot_responses/run_gsm8k_letter_case_20260302_230112",
        "results/cot_responses/run_gsm8k_letter_case_20260302_235054",
        "results/cot_responses/run_gsm8k_letter_case_20260303_162219",
        "results/cot_responses/run_gsm8k_letter_case_20260303_201341",
        "results/cot_responses/run_gsm8k_letter_case_20260303_231807",
        "results/cot_responses/run_gsm8k_letter_case_20260304_021959",
        # G4-E4B, Q3.5-9B
        "results/cot_responses/run_gsm8k_letter_case_20260409_131705",
        "results/cot_responses/run_gsm8k_letter_case_20260410_012919",
    ],
    "politeness": [
        "results/cot_responses/run_gsm8k_politeness_20260303_011403",
        "results/cot_responses/run_gsm8k_politeness_20260303_020840",
        "results/cot_responses/run_gsm8k_politeness_20260303_175041",
        "results/cot_responses/run_gsm8k_politeness_20260303_203615",
        "results/cot_responses/run_gsm8k_politeness_20260304_003300",
        "results/cot_responses/run_gsm8k_politeness_20260304_023434",
        # G4-E4B, Q3.5-9B
        "results/cot_responses/run_gsm8k_politeness_20260409_082038",
        "results/cot_responses/run_gsm8k_politeness_20260409_125547",
    ],
    "length_variation": [
        "results/cot_responses/run_gsm8k_length_variation_20260304_075450",
        "results/cot_responses/run_gsm8k_length_variation_20260304_103343",
        "results/cot_responses/run_gsm8k_length_variation_20260304_113317",
        "results/cot_responses/run_gsm8k_length_variation_20260304_132144",
        "results/cot_responses/run_gsm8k_length_variation_20260304_134359",
        "results/cot_responses/run_gsm8k_length_variation_20260304_154646",
        # G4-E4B, Q3.5-9B
        "results/cot_responses/run_gsm8k_length_variation_20260409_103642",
        "results/cot_responses/run_gsm8k_length_variation_20260409_183337",
    ],
    "inter_vs_imper": [
        "results/cot_responses/run_gsm8k_inter_vs_imper_20260304_193938",
        # G4-E4B, Q3.5-9B
        "results/cot_responses/run_gsm8k_inter_vs_imper_20260409_184944",
        "results/cot_responses/run_gsm8k_inter_vs_imper_20260410_160103",
    ],
}

# ── Silhouette CSV paths (pre-aggregated per model/place/strength) ─────────────
# Old models: results/safety/{variation}/{model}/combined_means_by_model_place_strength.csv
# New models: results/safety/{variation}_activations/summary.csv
SILHOUETTE_CSV_PATHS: dict[str, list[str]] = {
    "spacing": [
        "results/safety/spacing/G-2B/combined_means_by_model_place_strength.csv",
        "results/safety/spacing/G-7B/combined_means_by_model_place_strength.csv",
        "results/safety/spacing/L3.1-8B/combined_means_by_model_place_strength.csv",
        "results/safety/spacing/L3.2-3B/combined_means_by_model_place_strength.csv",
        "results/safety/spacing/Q2.5-1.5B/combined_means_by_model_place_strength.csv",
        "results/safety/spacing/Q2.5-7B/combined_means_by_model_place_strength.csv",
        # G4-E4B (Q3.5-9B not yet computed)
        "results/safety/spacing_activations/summary.csv",
    ],
    "punctuation": [
        "results/safety/punctuation/G-2B/combined_means_by_model_place_strength.csv",
        "results/safety/punctuation/G-7B/combined_means_by_model_place_strength.csv",
        "results/safety/punctuation/L3.1-8B/combined_means_by_model_place_strength.csv",
        "results/safety/punctuation/L3.2-3B/combined_means_by_model_place_strength.csv",
        "results/safety/punctuation/Q2.5-1.5B/combined_means_by_model_place_strength.csv",
        "results/safety/punctuation/Q2.5-7B/combined_means_by_model_place_strength.csv",
        "results/safety/punctuation_activations/summary.csv",
    ],
    "letter_case": [
        "results/safety/letter_case/G-2B/combined_means_by_model_place_strength.csv",
        "results/safety/letter_case/G-7B/combined_means_by_model_place_strength.csv",
        "results/safety/letter_case/L3.1-8B/combined_means_by_model_place_strength.csv",
        "results/safety/letter_case/L3.2-3B/combined_means_by_model_place_strength.csv",
        "results/safety/letter_case/Q2.5-1.5B/combined_means_by_model_place_strength.csv",
        "results/safety/letter_case/Q2.5-7B/combined_means_by_model_place_strength.csv",
        "results/safety/letter_case_activations/summary.csv",
    ],
    "politeness": [
        "results/politeness_safety/run_20260226_130811/summary.csv",
        "results/politeness_safety/run_20260226_153724/summary.csv",
        "results/politeness_safety/run_20260226_175026/summary.csv",
        "results/politeness_safety/run_20260226_153336/summary.csv",
        "results/politeness_safety/run_20260226_174657/summary.csv",
        "results/politeness_safety/run_20260226_193617/summary.csv",
        "results/safety/politeness_activations/summary.csv",
    ],
    "length_variation": [
        "results/safety/run_20260304_222347/summary.csv",
        "results/safety/run_20260304_221732/summary.csv",
        "results/safety/run_20260304_215712/summary.csv",
        "results/safety/run_20260304_200957/summary.csv",
        "results/safety/run_20260304_192633/summary.csv",
        "results/safety/run_20260304_185144/summary.csv",
        "results/safety/run_20260304_184908/summary.csv",
        "results/safety/run_20260304_184813/summary.csv",
        "results/safety/run_20260304_184735/summary.csv",
        "results/safety/run_20260304_184644/summary.csv",
        "results/safety/run_20260304_184601/summary.csv",
        "results/safety/run_20260304_184504/summary.csv",
        "results/safety/length_variation_activations/summary.csv",
    ],
    "inter_vs_imper": [
        "results/safety/run_20260304_184455/summary.csv",
        "results/safety/run_20260304_185337/summary.csv",
        "results/safety/run_20260304_185933/summary.csv",
        "results/safety/run_20260304_190008/summary.csv",
        "results/safety/run_20260304_191158/summary.csv",
        "results/safety/run_20260304_191226/summary.csv",
        "results/safety/run_20260304_191522/summary.csv",
        "results/safety/run_20260304_191556/summary.csv",
        "results/safety/run_20260304_192547/summary.csv",
        "results/safety/run_20260304_192616/summary.csv",
        "results/safety/run_20260304_192902/summary.csv",
        "results/safety/run_20260304_192936/summary.csv",
        "results/safety/inter_vs_imper_activations/summary.csv",
    ],
}

# ── New-layout safety ASR dirs (results/safety/{variation}_asr/asr_outputs/{model}/) ─
# harmbench_judged_{place}_s{strength}.csv files inside each model subdir
SAFETY_ASR_STYLE_DIRS: dict[str, str] = {
    "spacing":         "results/safety/spacing_asr/asr_outputs",
    "punctuation":     "results/safety/punctuation_asr/asr_outputs",
    "letter_case":     "results/safety/letter_case_asr/asr_outputs",
    "politeness":      "results/safety/politeness_asr/asr_outputs",
    "length_variation":"results/safety/length_variation_asr/asr_outputs",
    "inter_vs_imper":  "results/safety/inter_vs_imper_asr/asr_outputs",
}

# ── Safety run directories (asr_outputs/*/harmbench_judged_*.csv inside each) ─
# Only variations that have per-prompt data (not pre-aggregated)
SAFETY_RUN_DIRS: dict[str, list[str]] = {
    "politeness": [
        "results/politeness_safety/run_20260226_130811",
        "results/politeness_safety/run_20260226_153724",
        "results/politeness_safety/run_20260226_175026",
        "results/politeness_safety/run_20260226_153336",
        "results/politeness_safety/run_20260226_174657",
        "results/politeness_safety/run_20260226_193617",
    ],
    "length_variation": [
        "results/safety/run_20260304_222347",
        "results/safety/run_20260304_221732",
        "results/safety/run_20260304_215712",
        "results/safety/run_20260304_200957",
        "results/safety/run_20260304_192633",
        "results/safety/run_20260304_185144",
        "results/safety/run_20260304_184908",
        "results/safety/run_20260304_184813",
        "results/safety/run_20260304_184735",
        "results/safety/run_20260304_184644",
        "results/safety/run_20260304_184601",
        "results/safety/run_20260304_184504",
    ],
    "inter_vs_imper": [
        "results/safety/run_20260304_184455",
        "results/safety/run_20260304_185337",
        "results/safety/run_20260304_185933",
        "results/safety/run_20260304_190008",
        "results/safety/run_20260304_191158",
        "results/safety/run_20260304_191226",
        "results/safety/run_20260304_191522",
        "results/safety/run_20260304_191556",
        "results/safety/run_20260304_192547",
        "results/safety/run_20260304_192616",
        "results/safety/run_20260304_192902",
        "results/safety/run_20260304_192936",
    ],
}

# ── Metric label maps ──────────────────────────────────────────────────────────
CONTINUOUS_METRICS: dict[str, str] = {
    "delta_bertscore_response":    "Δ BERTScore",
    "delta_bleu":                  "Δ BLEU",
    "delta_activation_similarity": "Δ Cos. Sim (Activation)",
    "delta_log_prob":              "Δ Log-Prob",
    "delta_entropy":               "Δ Entropy",
}

COT_METRICS: dict[str, str] = {
    "cot_correct_num": "CoT Accuracy",
    "cot_steps":       "CoT Steps",
}

SAFETY_METRICS: dict[str, str] = {
    "unsafe_score": "ASR",
}

ALL_METRIC_LABELS: dict[str, str] = {
    **CONTINUOUS_METRICS,
    **COT_METRICS,
    **SAFETY_METRICS,
    "delta_mirroring_rate": "Δ Mirroring Rate",
    "silhouette":     "Silhouette Score",
}

_METRIC_ORDER = list(ALL_METRIC_LABELS)


def _metric_sort_key(m: str) -> int:
    """Return a sort key for a metric name based on the canonical display order."""
    return _METRIC_ORDER.index(m) if m in _METRIC_ORDER else len(_METRIC_ORDER)


MODEL_COLORS: dict[str, str] = {
    "G-2B":      "#1f77b4",
    "G-7B":      "#ff7f0e",
    "L3.2-3B":   "#2ca02c",
    "L3.1-8B":   "#d62728",
    "Q2.5-1.5B": "#9467bd",
    "Q2.5-7B":   "#8c564b",
    "G4-E4B":    "#17becf",
    "Q3.5-9B":   "#bcbd22",
}

SUBSET_SIZES = [4, 8, 16, 32, 64, 128]
N_BOOTSTRAP  = 500
RANDOM_SEED  = 42


# ── Data loading ───────────────────────────────────────────────────────────────

def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coerce the given columns to numeric, replacing non-numeric values with NaN in-place."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_all_styles(
    style_paths: dict[str, list[str]],
    variations: list[str],
) -> pd.DataFrame:
    """
    Load every requested variation, tag each row with its variation name, and return
    a single deduplicated DataFrame with columns:
        model, variation, prompt_id, place, strength, <metrics>...
    """
    dfs = []
    for variation in variations:
        for rel in style_paths.get(variation, []):
            p = PROJECT_ROOT / rel
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
                df["variation"] = variation
                dfs.append(df)
            except Exception as exc:
                print(f"  [warn] cannot read {p}: {exc}", file=sys.stderr)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = _to_numeric(combined, list(CONTINUOUS_METRICS))

    # Add numeric mirroring rate
    if "mirroring_verdict" in combined.columns and "mirroring_rate" not in combined.columns:
        combined["mirroring_rate"] = (
            combined["mirroring_verdict"].astype(str).str.strip().str.upper() == "YES"
        ).astype(float)

    # Compute delta_mirroring_rate per (model, prompt_id, place) relative to baseline strength
    if "mirroring_rate" in combined.columns and "delta_mirroring_rate" not in combined.columns:
        from utils.compute_delta_metrics import BASELINE_STRENGTH
        id_col = "prompt_id" if "prompt_id" in combined.columns else "problem_id"
        group_keys = [k for k in ["model", id_col, "place"] if k in combined.columns]
        parts = []
        for variation_name, sdf in combined.groupby("variation"):
            sdf = sdf.copy()
            baseline_s = BASELINE_STRENGTH.get(variation_name, 0)
            base = sdf[sdf["strength"] == baseline_s].set_index(group_keys)
            if not base.empty:
                idx = pd.MultiIndex.from_arrays([sdf[k] for k in group_keys])
                sdf["delta_mirroring_rate"] = (
                    sdf["mirroring_rate"].values - base["mirroring_rate"].reindex(idx).values
                )
            parts.append(sdf)
        combined = pd.concat(parts, ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["model", "variation", "prompt_id", "place", "strength"]
    )
    print(
        f"\nLoaded {len(combined)} rows ({before - len(combined)} dupes dropped)\n"
        f"  variations  : {sorted(combined['variation'].unique().tolist())}\n"
        f"  models  : {sorted(combined['model'].unique().tolist())}\n"
        f"  prompts : {combined['prompt_id'].nunique()} unique prompt_ids"
    )
    return combined


def load_cot_df(variations: list[str]) -> pd.DataFrame:
    """
    Load per-problem CoT analysis data from results_with_cot_analysis.csv files.
    Returns a DataFrame with columns:
        model, variation, prompt_id, place, strength, cot_steps, cot_correct_num
    """
    dfs = []
    for variation in variations:
        for run_dir in COT_RUN_DIRS.get(variation, []):
            p = PROJECT_ROOT / run_dir / "results_with_cot_analysis.csv"
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
                df["variation"] = variation
                dfs.append(df)
            except Exception as exc:
                print(f"  [warn] {p}: {exc}", file=sys.stderr)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Rename problem_id → prompt_id for consistency
    if "problem_id" in combined.columns and "prompt_id" not in combined.columns:
        combined = combined.rename(columns={"problem_id": "prompt_id"})

    # cot_correct: "1" → 1.0, "0" → 0.0, anything else → NaN
    combined["cot_correct_num"] = combined["cot_correct"].apply(
        lambda x: 1.0 if str(x).strip() == "1" else (0.0 if str(x).strip() == "0" else np.nan)
    )
    combined["cot_steps"] = pd.to_numeric(combined["cot_steps"], errors="coerce")
    # Remove clearly invalid step counts
    combined.loc[combined["cot_steps"] < 0, "cot_steps"] = np.nan

    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["model", "variation", "prompt_id", "place", "strength"]
    )
    print(
        f"\n[CoT] {len(combined)} rows ({before - len(combined)} dupes dropped)\n"
        f"  variations  : {sorted(combined['variation'].unique().tolist())}\n"
        f"  models  : {sorted(combined['model'].unique().tolist())}\n"
        f"  prompts : {combined['prompt_id'].nunique()} unique prompt_ids"
    )
    return combined


def load_safety_asr_df(variations: list[str]) -> pd.DataFrame:
    """
    Load per-prompt safety data from asr_outputs/*/harmbench_judged_*.csv files.
    Only available for variations in SAFETY_RUN_DIRS.
    Returns a DataFrame with columns:
        model, variation, prompt_id, place, strength, unsafe_score
    """
    dfs = []
    for variation in variations:
        for run_dir in SAFETY_RUN_DIRS.get(variation, []):
            asr_dir = PROJECT_ROOT / run_dir / "asr_outputs"
            if not asr_dir.exists():
                continue
            for judged_file in sorted(asr_dir.glob("*/harmbench_judged_*.csv")):
                try:
                    df = pd.read_csv(judged_file)
                    df["variation"] = variation
                    dfs.append(df)
                except Exception as exc:
                    print(f"  [warn] {judged_file}: {exc}", file=sys.stderr)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Ensure unsafe_score is numeric 0/1
    if "unsafe_score" in combined.columns:
        combined["unsafe_score"] = pd.to_numeric(combined["unsafe_score"], errors="coerce")
    elif "judge_raw" in combined.columns:
        combined["unsafe_score"] = (
            combined["judge_raw"].astype(str).str.strip().str.lower() != "safe"
        ).astype(float)

    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["model", "variation", "prompt_id", "place", "strength"]
    )
    print(
        f"\n[Safety] {len(combined)} rows ({before - len(combined)} dupes dropped)\n"
        f"  variations  : {sorted(combined['variation'].unique().tolist())}\n"
        f"  models  : {sorted(combined['model'].unique().tolist())}\n"
        f"  prompts : {combined['prompt_id'].nunique()} unique prompt_ids"
    )
    return combined


def load_safety_asr_new_df(variations: list[str]) -> pd.DataFrame:
    """
    Load per-prompt safety ASR data from the new-layout flat directories:
        results/safety/{variation}_asr/asr_outputs/{model}/harmbench_judged_*.csv
    Returns a DataFrame with columns:
        model, variation, prompt_id, place, strength, unsafe_score
    """
    dfs = []
    for variation in variations:
        asr_dir = PROJECT_ROOT / SAFETY_ASR_STYLE_DIRS.get(variation, "")
        if not asr_dir.exists():
            continue
        for judged_file in sorted(asr_dir.glob("*/harmbench_judged_*.csv")):
            try:
                df = pd.read_csv(judged_file)
                df["variation"] = variation
                dfs.append(df)
            except Exception as exc:
                print(f"  [warn] {judged_file}: {exc}", file=sys.stderr)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    if "unsafe_score" in combined.columns:
        combined["unsafe_score"] = pd.to_numeric(combined["unsafe_score"], errors="coerce")

    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["model", "variation", "prompt_id", "place", "strength"]
    )
    print(
        f"\n[Safety-new] {len(combined)} rows ({before - len(combined)} dupes dropped)\n"
        f"  variations  : {sorted(combined['variation'].unique().tolist())}\n"
        f"  models  : {sorted(combined['model'].unique().tolist())}\n"
        f"  prompts : {combined['prompt_id'].nunique()} unique prompt_ids"
    )
    return combined


def load_silhouette_df(variations: list[str]) -> pd.DataFrame:
    """
    Load pre-aggregated silhouette scores from results/safety/{variation}_activations/summary.csv.

    Returns a DataFrame with columns:
        model, variation, place, strength, silhouette

    NOTE: silhouette is already one value per (model, variation, place, strength) group —
    there is no per-prompt silhouette. Bootstrap sensitivity will operate over conditions
    (variation × place × strength) rather than prompt_ids.
    """
    dfs = []
    for variation in variations:
        for rel in SILHOUETTE_CSV_PATHS.get(variation, []):
            p = PROJECT_ROOT / rel
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
                df["variation"] = variation
                dfs.append(df)
            except Exception as exc:
                print(f"  [warn] {p}: {exc}", file=sys.stderr)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined["silhouette"] = pd.to_numeric(combined["silhouette"], errors="coerce")
    combined = combined.dropna(subset=["silhouette"])

    before = len(combined)
    combined = combined.drop_duplicates(subset=["model", "variation", "place", "strength"])
    print(
        f"\n[Silhouette] {len(combined)} rows ({before - len(combined)} dupes dropped)\n"
        f"  variations  : {sorted(combined['variation'].unique().tolist())}\n"
        f"  models  : {sorted(combined['model'].unique().tolist())}"
    )
    return combined


# ── Bootstrap sensitivity ──────────────────────────────────────────────────────

def bootstrap_variance_per_model(
    df: pd.DataFrame,
    metric: str,
    subset_sizes: list[int],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, dict[int, float]]:
    """
    Condition-spread check: how much do results vary across (variation × place × strength)?

    For each model and subset size N:
      1. Draw n_bootstrap random subsets of N prompt-IDs (with replacement).
      2. For each subset, compute mean(metric) per (variation × place × strength) group.
      3. Compute the std of those group means within this bootstrap → condition spread.
      4. Average that std across all bootstraps → Y.

    Interpretation:
      High Y → metric varies strongly across variation/place/strength conditions
               → the variation manipulation has a large measurable effect.
      Low Y  → conditions all produce similar metric values
               → little variation detected at this N (may be noise or true flat effect).
      Stable curve → N is sufficient to detect the real condition spread.

    Returns {model: {N: mean_across_bootstrap_condition_std}}
    """
    df_clean = df.dropna(subset=[metric])
    model_results: dict[str, dict[int, float]] = {}

    for model, df_m in df_clean.groupby("model"):
        all_ids = df_m["prompt_id"].unique()
        df_variationd = df_m[df_m["strength"] != 0]
        size_results: dict[int, float] = {}
        for n in subset_sizes:
            per_bootstrap_stds: list[float] = []
            for _ in range(n_bootstrap):
                sampled = rng.choice(all_ids, size=n, replace=True)
                sub_agg = (
                    df_variationd[df_variationd["prompt_id"].isin(sampled)]
                    .groupby(["variation", "place", "strength"])[metric]
                    .mean()
                )
                # std across conditions for this bootstrap sample
                s = sub_agg.std()
                if not np.isnan(s):
                    per_bootstrap_stds.append(s)

            size_results[n] = float(np.nanmean(per_bootstrap_stds)) if per_bootstrap_stds else np.nan

        model_results[str(model)] = size_results

    return model_results


def bootstrap_metric_per_model(
    df: pd.DataFrame,
    metric: str,
    subset_sizes: list[int],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, dict[int, list[float]]]:
    """
    For each model and N, bootstrap N prompt-IDs and compute the mean metric
    across all variationd rows (strength != 0) in the sampled subset.

    Returns {model: {N: [mean_1, mean_2, ...]}}
    """
    df_clean = df.dropna(subset=[metric])
    model_results: dict[str, dict[int, list[float]]] = {}

    for model, df_m in df_clean.groupby("model"):
        all_ids = df_m["prompt_id"].unique()
        df_variationd = df_m[df_m["strength"] != 0]
        size_results: dict[int, list[float]] = {}

        for n in subset_sizes:
            n_sample = min(n, len(all_ids))
            means: list[float] = []
            for _ in range(n_bootstrap):
                sampled = rng.choice(all_ids, size=n_sample, replace=False)
                val = df_variationd[df_variationd["prompt_id"].isin(sampled)][metric].mean()
                if not np.isnan(val):
                    means.append(float(val))
            size_results[n] = means if means else [np.nan]

        model_results[str(model)] = size_results

    return model_results


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _band(corr_dict: dict, sizes: list[int]):
    """Return (means, p10, p90) arrays across bootstrap samples for the given subset sizes."""
    means, lo, hi = [], [], []
    for n in sizes:
        v = corr_dict.get(n, [np.nan])
        if isinstance(v, tuple):          # pre-computed (mean, p10, p90) from CSV
            means.append(v[0]); lo.append(v[1]); hi.append(v[2])
        else:
            v = np.array(v, dtype=float)
            means.append(np.nanmean(v))
            lo.append(np.nanpercentile(v, 10))
            hi.append(np.nanpercentile(v, 90))
    return np.array(means), np.array(lo), np.array(hi)


def _add_recommendation_text(fig, lines: list[str]) -> None:
    """Add a small text block below the legend summarising chosen values."""
    text = "\n".join(lines)
    fig.text(1.01, 0.02, text, transform=fig.transFigure,
             fontsize=7, va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))


def _label_points(ax, xs, ys, color="black", fontsize=5.5, dy=3):
    """Annotate each non-NaN (x, y) with its value as small text above the point."""
    for x, y in zip(xs, ys):
        if not np.isnan(float(y)):
            ax.annotate(f"{y:.3g}", xy=(x, y),
                        xytext=(0, dy), textcoords="offset points",
                        fontsize=fontsize, ha="center", va="bottom",
                        color=color)


# ── Plot helpers for variance ─────────────────────────────────────────────────

def _axis_style_var(ax, subset_sizes: list[int], ylabel: str = "Std of Group Means (Condition Spread)"):
    """Apply standard log2 x-axis styling and labels to a variance/spread panel."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(subset_sizes)
    ax.set_xticklabels([str(n) for n in subset_sizes])
    ax.set_xlabel("Number of Prompts (N)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2, linestyle="--")


# ── Plot 2: variance per-metric — one panel per metric, one line per model ─────

def plot_variance_per_metric(
    all_var_results: dict[str, dict[str, dict[int, float]]],
    subset_sizes: list[int],
    out_path: Path,
):
    """Plot per-metric condition spread (std of group means) vs. number of prompts N."""
    metrics = sorted(all_var_results.keys(), key=_metric_sort_key)

    ncols = 3
    nrows = max(1, (len(metrics) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 3.2 * nrows),
                              squeeze=False)

    chosen_ns: list[int] = []
    for i, metric in enumerate(metrics):
        ax = axes[i // ncols][i % ncols]
        for model in sorted(all_var_results[metric]):
            vals = np.array(
                [all_var_results[metric][model].get(n, np.nan) for n in subset_sizes]
            )
            color = MODEL_COLORS.get(model, "gray")
            ax.plot(subset_sizes, vals, label=model, color=color,
                    marker="o", linewidth=2.2, markersize=5.5)
            _label_points(ax, subset_sizes, vals, color=color)

        # Best N per panel: max across models
        best_ns = [
            _find_best_n(
                {n: [all_var_results[metric][m].get(n, np.nan)] for n in subset_sizes},
                subset_sizes,
            )
            for m in all_var_results[metric]
        ]
        best_n = max(best_ns) if best_ns else max(subset_sizes)
        chosen_ns.append(best_n)
        ax.axvline(best_n, color="black", lw=1.6, ls="--", alpha=0.75, zorder=5)
        ylims = ax.get_ylim()
        ax.text(best_n * 1.06, ylims[0] + 0.04 * (ylims[1] - ylims[0]),
                f"N={best_n}", fontsize=6.5, color="black", va="bottom")

        _axis_style_var(ax, subset_sizes, ylabel="Condition Spread (Std)")
        ax.set_title(ALL_METRIC_LABELS.get(metric, metric), fontsize=11)

    for j in range(len(metrics), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5),
               fontsize=9, framealpha=0.85)
    rec_n = max(chosen_ns) if chosen_ns else max(subset_sizes)
    _add_recommendation_text(fig, [f"Recommended N: {rec_n}"])
    fig.suptitle(
        "Per-Metric Condition Spread: Std of (variation × place × strength) Group Means\n"
        "(High = strong condition effect;  Stable curve = N sufficient to detect it)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")

    # CSV
    rows = [{"metric": m, "best_n": n, "recommended_n": rec_n}
            for m, n in zip(metrics, chosen_ns)]
    rows.append({"metric": "OVERALL", "best_n": rec_n, "recommended_n": rec_n})
    csv_path = out_path.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path}")


_PUB_FONT_AXIS   = 34
_PUB_FONT_TICK   = 30
_PUB_FONT_LEGEND = 28
_PUB_FIG_W       = 12
_PUB_FIG_H       = 7


def plot_variance_per_metric_pub(
    all_var_results: dict[str, dict[str, dict[int, float]]],
    subset_sizes: list[int],
    out_path: Path,
):
    """
    Publication-ready version of plot_variance_per_metric.
    Y = STD (condition spread).  One panel per metric, one line per model.
    No main title.  No recommendation slot.  Large fonts matching plot_individual_figures.py.
    """
    metrics = sorted(all_var_results.keys(), key=_metric_sort_key)

    ncols = 3
    nrows = max(1, (len(metrics) + ncols - 1) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(_PUB_FIG_W * ncols, _PUB_FIG_H * nrows),
        squeeze=False,
    )

    for i, metric in enumerate(metrics):
        ax = axes[i // ncols][i % ncols]
        for model in sorted(all_var_results[metric]):
            vals = np.array(
                [all_var_results[metric][model].get(n, np.nan) for n in subset_sizes]
            )
            color = MODEL_COLORS.get(model, "gray")
            ax.plot(
                subset_sizes, vals,
                label=model, color=color,
                marker="o", linewidth=4.0, markersize=10,
            )

        best_ns = [
            _find_best_n(
                {n: [all_var_results[metric][m].get(n, np.nan)] for n in subset_sizes},
                subset_sizes,
            )
            for m in all_var_results[metric]
        ]
        best_n = max(best_ns) if best_ns else max(subset_sizes)
        ax.axvline(best_n, color="black", lw=3.5, ls="--", alpha=0.85, zorder=5)

        ax.set_xscale("log", base=2)
        ax.set_xticks(subset_sizes)
        ax.set_xticklabels([str(n) for n in subset_sizes], fontsize=_PUB_FONT_TICK)
        ax.set_xlabel("Number of Prompts (N)", fontsize=_PUB_FONT_AXIS)
        ax.set_ylabel("STD", fontsize=_PUB_FONT_AXIS)
        ax.tick_params(axis="y", labelsize=_PUB_FONT_TICK, width=2.5, length=6)
        ax.tick_params(axis="x", width=2.5, length=6)
        ax.set_ylim(bottom=0)
        ax.set_title(ALL_METRIC_LABELS.get(metric, metric), fontsize=_PUB_FONT_AXIS)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(2.5)
        ax.spines["bottom"].set_linewidth(2.5)

    for j in range(len(metrics), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=_PUB_FONT_LEGEND,
        framealpha=0.9,
        handlelength=2.0,
    )
    for lh in leg.legend_handles:
        lh.set_linewidth(4.0)
    plt.tight_layout(h_pad=3.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def _find_best_n(
    metric_per_n: dict,
    subset_sizes: list,
    tolerance: float = 0.15,
):
    """
    Return the smallest N/value where the mean is within `tolerance` of the
    max-key reference.  When values are equal (including all-zero), always
    returns the smallest candidate — never falls back to max early.
    """
    full_n = max(subset_sizes)
    full_mean = np.nanmean(metric_per_n.get(full_n, [np.nan]))
    if np.isnan(full_mean):
        return full_n
    # Use max(|full_mean|, tiny) so zero-valued metrics still get a finite threshold
    ref = max(abs(full_mean), 1e-9)
    for n in sorted(subset_sizes):          # explicit sort guarantees smallest-first
        mean = np.nanmean(metric_per_n.get(n, [np.nan]))
        if not np.isnan(mean) and abs(mean - full_mean) <= tolerance * ref:
            return n
    return full_n


# ── Plot: per-metric metric value — one panel per metric, one line per model ───

def plot_metric_per_model(
    all_metric_results: dict[str, dict[str, dict[int, list[float]]]],
    # metric → model → {n: [values]}
    subset_sizes: list[int],
    out_path: Path,
    tolerance: float = 0.15,
):
    """
    One panel per metric. Y-axis = actual metric value. X-axis = N (log2).
    One line per model with P10–P90 bootstrap band.
    A vertical dashed line marks the 'best N' — the smallest N where every
    model's mean estimate is within `tolerance` of the N=128 reference.
    """
    metrics = sorted(all_metric_results.keys(), key=_metric_sort_key)
    ncols = 3
    nrows = max(1, (len(metrics) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 3.2 * nrows),
                              squeeze=False)

    chosen_ns: list[int] = []

    for i, metric in enumerate(metrics):
        ax = axes[i // ncols][i % ncols]
        model_data = all_metric_results[metric]

        # Best N = smallest N sufficient for all models
        best_ns = [
            _find_best_n(size_dict, subset_sizes, tolerance=tolerance)
            for size_dict in model_data.values()
        ]
        best_n = max(best_ns) if best_ns else max(subset_sizes)
        chosen_ns.append(best_n)

        for model in sorted(model_data):
            means, lo, hi = _band(model_data[model], subset_sizes)
            color = MODEL_COLORS.get(model, "gray")
            ax.plot(subset_sizes, means, label=model, color=color,
                    marker="o", linewidth=2.2, markersize=5.5)
            ax.fill_between(subset_sizes, lo, hi, alpha=0.09, color=color)
            _label_points(ax, subset_sizes, means, color=color)

        # Vertical line at best N
        ax.axvline(best_n, color="black", lw=1.6, ls="--", alpha=0.75,
                   zorder=5, label=f"Best N={best_n}")
        ylims = ax.get_ylim()
        ax.text(best_n * 1.06, ylims[0] + 0.04 * (ylims[1] - ylims[0]),
                f"N={best_n}", fontsize=7, color="black", va="bottom")

        ax.set_xscale("log", base=2)
        ax.set_xticks(subset_sizes)
        ax.set_xticklabels([str(n) for n in subset_sizes])
        ax.set_xlabel("Number of Prompts (N)", fontsize=11)
        ax.set_ylabel(ALL_METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_title(ALL_METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.grid(True, alpha=0.2, linestyle="--")

    for j in range(len(metrics), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5),
               fontsize=9, framealpha=0.85)
    rec_n = max(chosen_ns) if chosen_ns else max(subset_sizes)
    _add_recommendation_text(fig, [f"Recommended N: {rec_n}"])
    fig.suptitle(
        "Sample Size Ablation: Metric Value vs. Number of Prompts\n"
        "(Bands = P10–P90 bootstrap;  dashed line = recommended N per metric)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")

    # CSV
    rows = [{"metric": m, "best_n": n, "recommended_n": rec_n}
            for m, n in zip(metrics, chosen_ns)]
    rows.append({"metric": "OVERALL", "best_n": rec_n, "recommended_n": rec_n})
    csv_path = out_path.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path}")


def plot_metric_per_model_pub(
    all_metric_results: dict[str, dict[str, dict[int, list[float]]]],
    subset_sizes: list[int],
    out_path: Path,
    tolerance: float = 0.15,
):
    """
    Publication-ready version of plot_metric_per_model.
    No main title.  No recommendation slot.  Large fonts matching plot_individual_figures.py.
    """
    metrics = sorted(all_metric_results.keys(), key=_metric_sort_key)
    ncols = 3
    nrows = max(1, (len(metrics) + ncols - 1) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(_PUB_FIG_W * ncols, _PUB_FIG_H * nrows),
        squeeze=False,
    )

    for i, metric in enumerate(metrics):
        ax = axes[i // ncols][i % ncols]
        model_data = all_metric_results[metric]

        best_ns = [
            _find_best_n(size_dict, subset_sizes, tolerance=tolerance)
            for size_dict in model_data.values()
        ]
        best_n = max(best_ns) if best_ns else max(subset_sizes)

        for model in sorted(model_data):
            means, lo, hi = _band(model_data[model], subset_sizes)
            color = MODEL_COLORS.get(model, "gray")
            ax.plot(subset_sizes, means, label=model, color=color,
                    marker="o", linewidth=4.0, markersize=10)
            ax.fill_between(subset_sizes, lo, hi, alpha=0.09, color=color)

        ax.axvline(best_n, color="black", lw=3.5, ls="--", alpha=0.85, zorder=5)

        ax.set_xscale("log", base=2)
        ax.set_xticks(subset_sizes)
        ax.set_xticklabels([str(n) for n in subset_sizes], fontsize=_PUB_FONT_TICK)
        ax.set_xlabel("Number of Prompts (N)", fontsize=_PUB_FONT_AXIS)
        ax.set_ylabel(ALL_METRIC_LABELS.get(metric, metric), fontsize=_PUB_FONT_AXIS)
        ax.tick_params(axis="y", labelsize=_PUB_FONT_TICK, width=2.5, length=6)
        ax.tick_params(axis="x", width=2.5, length=6)
        ax.set_title(ALL_METRIC_LABELS.get(metric, metric), fontsize=_PUB_FONT_AXIS)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(2.5)
        ax.spines["bottom"].set_linewidth(2.5)

    for j in range(len(metrics), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=_PUB_FONT_LEGEND,
        framealpha=0.9,
        handlelength=2.0,
    )
    for lh in leg.legend_handles:
        lh.set_linewidth(4.0)
    plt.tight_layout(h_pad=3.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ── Place ablation ────────────────────────────────────────────────────────────

def compute_place_stats(
    df: pd.DataFrame,
    metric: str,
) -> tuple[dict, dict]:
    """
    Returns (means, variances):
      means     = {model: {place: mean_metric}}   (varied rows only, strength != 0)
      variances = {model: {place: std_of_per_strength_group_means}}
    """
    df_c = df[df["strength"] != 0].dropna(subset=[metric])
    means: dict = {}
    variances: dict = {}
    for model, grp_m in df_c.groupby("model"):
        means[str(model)] = {}
        variances[str(model)] = {}
        for place, grp_p in grp_m.groupby("place"):
            means[str(model)][str(place)] = float(grp_p[metric].mean())
            per_s = grp_p.groupby("strength")[metric].mean()
            variances[str(model)][str(place)] = float(per_s.std()) if len(per_s) > 1 else np.nan
    return means, variances


def _select_representative(
    model_means: dict,
    candidates: list,
    tolerance: float = 0.15,
    always_include: list | None = None,
) -> list:
    """
    Return the minimal subset of candidates that covers the full effect range,
    always including any items listed in `always_include`.

    Greedy interval cover (sorted by mean effect across models): a candidate
    is added only if its mean effect is more than `tolerance * span` away from
    every already-selected candidate.  This guarantees every unselected point
    is within one threshold-width of a selected one.

    When all values are equal the smallest candidate is returned alone.
    """
    forced = [k for k in (always_include or []) if k in candidates]
    avg = {k: np.nanmean([model_means[m].get(k, np.nan) for m in model_means])
           for k in candidates}
    valid = [k for k in candidates if not np.isnan(avg[k])]
    if not valid:
        return forced or list(candidates)

    lo, hi = min(avg[k] for k in valid), max(avg[k] for k in valid)
    span = hi - lo
    if span < 1e-9:                          # all equal → forced + smallest
        base = [valid[0]]
        return sorted(set(forced + base), key=lambda k: candidates.index(k))

    thresh = tolerance * span
    selected: list = list(forced)            # seed with forced items
    for k in sorted(valid, key=lambda k: avg[k]):
        if k in selected:
            continue
        if not selected or all(abs(avg[k] - avg[s]) > thresh for s in selected):
            selected.append(k)
    # return in original candidate order
    return sorted(selected, key=lambda k: candidates.index(k))


def _make_place_panels(
    all_results: dict,
    ylabel: str,
    suptitle: str,
    out_path: Path,
    pub: bool = False,
) -> None:
    """Render one panel per metric showing a metric or variance value vs. prompt placement."""
    PLACE_ORDER = ["global", "prefix", "suffix"]
    metrics = sorted(all_results.keys(), key=_metric_sort_key)
    ncols = 3
    nrows = max(1, (len(metrics) + ncols - 1) // ncols)

    _fw = _PUB_FIG_W if pub else 4.5
    _fh = _PUB_FIG_H if pub else 3.2
    _fa = _PUB_FONT_AXIS if pub else 11
    _ft = _PUB_FONT_TICK if pub else 10
    _fl = _PUB_FONT_LEGEND if pub else 9
    _lw = 4.0 if pub else 2.2
    _ms = 10 if pub else 5.5

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(_fw * ncols, _fh * nrows),
                              squeeze=False)

    all_selected_places: set = set()
    per_metric_places: dict = {}

    for i, metric in enumerate(metrics):
        ax = axes[i // ncols][i % ncols]
        model_data = all_results[metric]

        places = sorted(
            {p for md in model_data.values() for p in md},
            key=lambda p: PLACE_ORDER.index(p) if p in PLACE_ORDER else 99,
        )
        x_pos = list(range(len(places)))
        selected = _select_representative(model_data, places, always_include=["global"])
        all_selected_places.update(selected)
        per_metric_places[metric] = selected

        for model in sorted(model_data):
            vals = [model_data[model].get(p, np.nan) for p in places]
            color = MODEL_COLORS.get(model, "gray")
            ax.plot(x_pos, vals, label=model, color=color,
                    marker="o", linewidth=_lw, markersize=_ms)
            if not pub:
                _label_points(ax, x_pos, vals, color=color)

        ylims = ax.get_ylim()
        y_range = ylims[1] - ylims[0]
        for idx, sel in enumerate(selected):
            sel_x = places.index(sel)
            ax.axvline(sel_x, color="black", lw=3.5 if pub else 1.0, ls="--", alpha=0.85 if pub else 0.75, zorder=5)
            if not pub:
                y_label = ylims[0] + (0.04 + 0.10 * (idx % 3)) * y_range
                ax.text(sel_x + 0.08, y_label, sel,
                        fontsize=7, color="black", va="bottom")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(places, fontsize=_ft)
        ax.set_xlabel("Placement", fontsize=_fa)
        ax.set_ylabel(
            ALL_METRIC_LABELS.get(metric, metric) if ylabel == "metric" else ylabel,
            fontsize=_fa,
        )
        if pub:
            ax.tick_params(axis="y", labelsize=_ft, width=2.5, length=6)
            ax.tick_params(axis="x", width=2.5, length=6)
        else:
            ax.tick_params(axis="y", labelsize=_ft)
        ax.set_title(ALL_METRIC_LABELS.get(metric, metric), fontsize=_fa)
        ax.grid(True, alpha=0.2, linestyle="--", axis="y")
        if pub:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(2.5)
            ax.spines["bottom"].set_linewidth(2.5)

    for j in range(len(metrics), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    rec_places = sorted(all_selected_places,
                        key=lambda p: PLACE_ORDER.index(p) if p in PLACE_ORDER else 99)
    if pub:
        leg = fig.legend(
            handles, labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=_fl,
            framealpha=0.9,
            handlelength=2.0,
        )
        for lh in leg.legend_handles:
            lh.set_linewidth(4.0)
    else:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5),
                   fontsize=_fl, framealpha=0.85)
        _add_recommendation_text(fig, [f"Recommended places: {', '.join(rec_places)}"])
        fig.suptitle(suptitle, fontsize=11, y=1.01)
    plt.tight_layout(h_pad=3.0 if pub else 1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")

    # CSV
    rec_places_str = ", ".join(rec_places)
    rows = [{"metric": m, "selected_places": ", ".join(per_metric_places.get(m, [])),
             "recommended_places": rec_places_str}
            for m in metrics]
    rows.append({"metric": "OVERALL", "selected_places": rec_places_str,
                 "recommended_places": rec_places_str})
    csv_path = out_path.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path}")


def plot_place_metric(all_means: dict, out_path: Path) -> None:
    """Plot mean metric value by prompt placement for the place ablation study."""
    _make_place_panels(
        all_means, ylabel="metric",
        suptitle=(
            "Place Ablation: Mean Metric Value by Prompt Placement\n"
            "(dashed = placement with strongest effect across models)"
        ),
        out_path=out_path,
    )


def plot_place_metric_pub(all_means: dict, out_path: Path) -> None:
    """Publication-ready version of plot_place_metric."""
    _make_place_panels(all_means, ylabel="metric", suptitle="", out_path=out_path, pub=True)


def plot_place_variance(all_variances: dict, out_path: Path) -> None:
    """Plot condition spread (std) by prompt placement for the place ablation study."""
    _make_place_panels(
        all_variances, ylabel="Condition Spread (Std)",
        suptitle=(
            "Place Ablation: Condition Spread by Prompt Placement\n"
            "(dashed = placement with highest variance across models)"
        ),
        out_path=out_path,
    )


def plot_place_variance_pub(all_variances: dict, out_path: Path) -> None:
    """Publication-ready version of plot_place_variance."""
    _make_place_panels(all_variances, ylabel="STD", suptitle="", out_path=out_path, pub=True)


# ── Strength ablation ─────────────────────────────────────────────────────────

def compute_strength_stats(
    df: pd.DataFrame,
    metric: str,
) -> tuple[dict, dict, list]:
    """
    Returns (means, variances, sorted_strengths):
      means     = {model: {strength: mean_metric}}   (numeric non-zero strengths only)
      variances = {model: {strength: std_of_per_place_group_means}}
    Non-numeric strength values (e.g. 'imperative') are skipped.
    """
    df_c = df.dropna(subset=[metric]).copy()
    df_c["_s"] = pd.to_numeric(df_c["strength"], errors="coerce")
    df_c = df_c.dropna(subset=["_s"])

    means: dict = {}
    variances: dict = {}
    for model, grp_m in df_c.groupby("model"):
        means[str(model)] = {}
        variances[str(model)] = {}
        for s, grp_s in grp_m.groupby("_s"):
            means[str(model)][float(s)] = float(grp_s[metric].mean())
            per_p = grp_s.groupby("place")[metric].mean()
            if len(per_p) > 1:
                variances[str(model)][float(s)] = float(per_p.std())
            else:
                # Single place (e.g. length_variation) — fall back to prompt-level std
                variances[str(model)][float(s)] = float(grp_s[metric].std()) if len(grp_s) > 1 else np.nan

    all_s = sorted({s for md in means.values() for s in md})
    return means, variances, all_s


def _make_strength_panels(
    all_results: dict,
    sorted_strengths: list,
    ylabel: str,
    suptitle: str,
    out_path: Path,
    variation: str = "",
    pub: bool = False,
) -> None:
    """Render one panel per metric showing a metric or variance value vs. variation strength."""
    metrics = sorted(all_results.keys(), key=_metric_sort_key)
    ncols = 3
    nrows = max(1, (len(metrics) + ncols - 1) // ncols)

    _fw = _PUB_FIG_W if pub else 4.5
    _fh = _PUB_FIG_H if pub else 3.2
    _fa = _PUB_FONT_AXIS if pub else 11
    _ft = _PUB_FONT_TICK if pub else 10
    _fl = _PUB_FONT_LEGEND if pub else 9
    _lw = 4.0 if pub else 2.2
    _ms = 10 if pub else 5.5

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(_fw * ncols, _fh * nrows),
                              squeeze=False)

    forced_s = [0.0]
    if variation == "length_variation" and 1.0 in sorted_strengths:
        forced_s.append(1.0)

    all_selected_strengths: set = set()
    per_metric_strengths: dict = {}

    for i, metric in enumerate(metrics):
        ax = axes[i // ncols][i % ncols]
        model_data = all_results[metric]
        selected = _select_representative(model_data, sorted_strengths,
                                          always_include=forced_s)
        all_selected_strengths.update(selected)
        per_metric_strengths[metric] = selected

        for model in sorted(model_data):
            vals = [model_data[model].get(s, np.nan) for s in sorted_strengths]
            color = MODEL_COLORS.get(model, "gray")
            ax.plot(sorted_strengths, vals, label=model, color=color,
                    marker="o", linewidth=_lw, markersize=_ms)
            if not pub:
                _label_points(ax, sorted_strengths, vals, color=color)

        ylims = ax.get_ylim()
        y_range = ylims[1] - ylims[0]
        span = sorted_strengths[-1] - sorted_strengths[0] if len(sorted_strengths) > 1 else 1
        for idx, sel in enumerate(selected):
            ax.axvline(sel, color="black", lw=3.5 if pub else 1.0, ls="--", alpha=0.85 if pub else 0.75, zorder=5)
            if not pub:
                y_label = ylims[0] + (0.04 + 0.10 * (idx % 3)) * y_range
                ax.text(sel + 0.02 * span, y_label,
                        f"s={sel:g}", fontsize=7, color="black", va="bottom")

        ax.set_xlabel("Strength", fontsize=_fa)
        ax.set_ylabel(
            ALL_METRIC_LABELS.get(metric, metric) if ylabel == "metric" else ylabel,
            fontsize=_fa,
        )
        if pub:
            ax.tick_params(axis="both", labelsize=_ft, width=2.5, length=6)
        else:
            ax.tick_params(axis="both", labelsize=_ft)
        ax.set_title(ALL_METRIC_LABELS.get(metric, metric), fontsize=_fa)
        ax.grid(True, alpha=0.2, linestyle="--")
        if pub:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(2.5)
            ax.spines["bottom"].set_linewidth(2.5)

    for j in range(len(metrics), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    rec_strengths = sorted(all_selected_strengths)
    if pub:
        leg = fig.legend(
            handles, labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=_fl,
            framealpha=0.9,
            handlelength=2.0,
        )
        for lh in leg.legend_handles:
            lh.set_linewidth(4.0)
    else:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5),
                   fontsize=_fl, framealpha=0.85)
        _add_recommendation_text(fig, [f"Recommended strengths: {', '.join(f'{s:g}' for s in rec_strengths)}"])
        fig.suptitle(suptitle, fontsize=11, y=1.01)
    plt.tight_layout(h_pad=3.0 if pub else 1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")

    # CSV
    rec_str = ", ".join(f"{s:g}" for s in rec_strengths)
    rows = [{"metric": m,
             "selected_strengths": ", ".join(f"{s:g}" for s in per_metric_strengths.get(m, [])),
             "recommended_strengths": rec_str}
            for m in metrics]
    rows.append({"metric": "OVERALL", "selected_strengths": rec_str,
                 "recommended_strengths": rec_str})
    csv_path = out_path.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path}")


def plot_strength_metric(all_means: dict, sorted_strengths: list, out_path: Path,
                         variation: str = "") -> None:
    """Plot mean metric value by variation strength for the strength ablation study."""
    _make_strength_panels(
        all_means, sorted_strengths, ylabel="metric",
        suptitle=(
            "Strength Ablation: Mean Metric Value by Style Strength\n"
            "(dashed = representative strengths)"
        ),
        out_path=out_path, variation=variation,
    )


def plot_strength_metric_pub(all_means: dict, sorted_strengths: list, out_path: Path,
                              variation: str = "") -> None:
    """Publication-ready version of plot_strength_metric."""
    _make_strength_panels(all_means, sorted_strengths, ylabel="metric", suptitle="",
                          out_path=out_path, variation=variation, pub=True)


def plot_strength_variance(all_variances: dict, sorted_strengths: list, out_path: Path,
                           variation: str = "") -> None:
    """Plot condition spread (std) by variation strength for the strength ablation study."""
    _make_strength_panels(
        all_variances, sorted_strengths, ylabel="Condition Spread (Std)",
        suptitle=(
            "Strength Ablation: Condition Spread by Style Strength\n"
            "(dashed = representative strengths)"
        ),
        out_path=out_path, variation=variation,
    )


def plot_strength_variance_pub(all_variances: dict, sorted_strengths: list, out_path: Path,
                                variation: str = "") -> None:
    """Publication-ready version of plot_strength_variance."""
    _make_strength_panels(all_variances, sorted_strengths, ylabel="STD",
                          suptitle="", out_path=out_path, variation=variation, pub=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    """Parse CLI arguments and run the sensitivity analysis bootstrap experiments."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir",     default="results/sensitivity_analysis")
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed",        type=int, default=RANDOM_SEED)
    parser.add_argument("--variations",      nargs="*", default=None,
                        help="Subset of variations to analyse (default: all)")
    parser.add_argument("--metrics",     nargs="*", default=None,
                        help="Only compute these metrics, e.g. --metrics silhouette unsafe_score")
    parser.add_argument("--models",      nargs="*", default=None,
                        help="Only compute for these models, e.g. --models G4-E4B Q3.5-9B")
    parser.add_argument("--skip_bootstrap", action="store_true",
                        help="Skip data loading and bootstrap; regenerate all plots from "
                             "the cached sensitivity_summary.csv and sensitivity_ablation_cache.pkl.")
    args = parser.parse_args()

    out_dir   = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out   = out_dir / "sensitivity_summary.csv"
    cache_pkl = out_dir / "sensitivity_ablation_cache.pkl"

    # ── Fast path: regenerate plots from existing cache files ─────────────────
    if args.skip_bootstrap:
        if not csv_out.exists():
            print(f"[ERROR] No cached summary at {csv_out}. Run without --skip_bootstrap first.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"  Loading N-ablation results from {csv_out}")
        df_summary = pd.read_csv(csv_out, dtype={"n": str})
        df_plot_num = df_summary[df_summary["n"] != "all"].copy()
        df_plot_num["n"] = pd.to_numeric(df_plot_num["n"])
        plot_metric_results: dict = {}
        plot_var_results:    dict = {}
        for metric, grp_m in df_plot_num.groupby("metric"):
            plot_metric_results[metric] = {}
            plot_var_results[metric]    = {}
            for model, grp_mod in grp_m.groupby("model"):
                plot_metric_results[metric][model] = {
                    int(row["n"]): (row["mean_rho"], row["p10_rho"], row["p90_rho"])
                    for _, row in grp_mod.iterrows()
                }
                plot_var_results[metric][model] = {
                    int(row["n"]): row["bootstrap_std"]
                    for _, row in grp_mod.iterrows()
                }
        all_place_means: dict = {}
        all_place_vars:  dict = {}
        strength_results: dict = {}
        if cache_pkl.exists():
            print(f"  Loading place/strength cache from {cache_pkl}")
            with open(cache_pkl, "rb") as _f:
                _cache = pickle.load(_f)
            all_place_means  = _cache.get("all_place_means", {})
            all_place_vars   = _cache.get("all_place_vars", {})
            strength_results = _cache.get("strength_results", {})
        else:
            print(f"  [WARN] No place/strength cache at {cache_pkl} — those plots skipped.")
        if plot_metric_results:
            print("\nGenerating N-ablation plots …")
            plot_metric_per_model(plot_metric_results, SUBSET_SIZES, out_dir / "metric_ablation.png")
            plot_metric_per_model_pub(plot_metric_results, SUBSET_SIZES, out_dir / "metric_ablation_pub.png")
            plot_variance_per_metric(plot_var_results, SUBSET_SIZES, out_dir / "variance_per_metric.png")
            plot_variance_per_metric_pub(plot_var_results, SUBSET_SIZES, out_dir / "variance_per_metric_pub.png")
        if all_place_means:
            print("\nGenerating place-ablation plots …")
            plot_place_metric(all_place_means,    out_dir / "place_ablation_metric.png")
            plot_place_metric_pub(all_place_means, out_dir / "place_ablation_metric_pub.png")
            plot_place_variance(all_place_vars,   out_dir / "place_ablation_variance.png")
            plot_place_variance_pub(all_place_vars, out_dir / "place_ablation_variance_pub.png")
        if strength_results:
            print("\nGenerating strength-ablation plots …")
            for variation, (sm, sv, all_s) in strength_results.items():
                plot_strength_metric(sm,  all_s, out_dir / f"strength_ablation_metric_{variation}.png",  variation=variation)
                plot_strength_metric_pub(sm,  all_s, out_dir / f"strength_ablation_metric_{variation}_pub.png",  variation=variation)
                plot_strength_variance(sv, all_s, out_dir / f"strength_ablation_variance_{variation}.png", variation=variation)
                plot_strength_variance_pub(sv, all_s, out_dir / f"strength_ablation_variance_{variation}_pub.png", variation=variation)
        df_numeric = df_summary[df_summary["n"] != "all"]
        if not df_numeric.empty:
            df_numeric = df_numeric.copy()
            df_numeric["n"] = pd.to_numeric(df_numeric["n"])
            print("\n── Best N per metric (smallest N within 15% of N=128 for all models) ──")
            for metric in sorted(df_numeric["metric"].unique()):
                sub_m = df_numeric[df_numeric["metric"] == metric]
                best_ns_per_model = {}
                for model, sub_mod in sub_m.groupby("model"):
                    size_dict = {int(row["n"]): [row["mean_rho"]] for _, row in sub_mod.iterrows()}
                    best_ns_per_model[model] = _find_best_n(size_dict, SUBSET_SIZES)
                best_n = max(best_ns_per_model.values())
                models_str = ", ".join(f"{m}={n}" for m, n in sorted(best_ns_per_model.items()))
                print(f"  {metric:30s}: recommended N={best_n}  ({models_str})")
        print("\nDone.")
        return

    rng    = np.random.default_rng(args.seed)
    variations = args.variations or list(VARIATION_CSV_PATHS.keys())

    metrics_filter = set(args.metrics) if args.metrics else None
    models_filter  = set(args.models)  if args.models  else None

    def _want_metric(m: str) -> bool:
        """Return True if metric m passes the CLI metrics filter."""
        return metrics_filter is None or m in metrics_filter

    def _filter_models(df: pd.DataFrame) -> pd.DataFrame:
        """Return df filtered to only the models in the CLI models filter, or unchanged if no filter."""
        if models_filter and "model" in df.columns:
            return df[df["model"].isin(models_filter)]
        return df

    # ── Load all data domains ─────────────────────────────────────────────────
    domain_specs = []

    if metrics_filter is None or metrics_filter & (set(CONTINUOUS_METRICS) | {"delta_mirroring_rate"}):
        print("\n=== Loading TruthfulQA data ===")
        df_tqa = _filter_models(load_all_styles(VARIATION_CSV_PATHS, variations))
        if not df_tqa.empty:
            tqa_metrics = [
                m for m in list(CONTINUOUS_METRICS) + ["delta_mirroring_rate"]
                if _want_metric(m) and m in df_tqa.columns and not df_tqa[m].isna().all()
            ]
            if tqa_metrics:
                domain_specs.append((df_tqa, tqa_metrics, "TruthfulQA"))

    # if metrics_filter is None or metrics_filter & set(COT_METRICS):
    #     print("\n=== Loading CoT (GSM8K) data ===")
    #     df_cot = _filter_models(load_cot_df(variations))
    #     if not df_cot.empty:
    #         cot_metrics = [
    #             m for m in COT_METRICS
    #             if _want_metric(m) and m in df_cot.columns and not df_cot[m].isna().all()
    #         ]
    #         if cot_metrics:
    #             domain_specs.append((df_cot, cot_metrics, "CoT/GSM8K"))

    # if metrics_filter is None or "unsafe_score" in (metrics_filter or set()):
    #     print("\n=== Loading Safety ASR (old runs) ===")
    #     df_asr_old = _filter_models(load_safety_asr_df(variations))
    #     print("\n=== Loading Safety ASR (new runs) ===")
    #     df_asr_new = _filter_models(load_safety_asr_new_df(variations))
    #     df_asr = pd.concat([df_asr_old, df_asr_new], ignore_index=True) if not df_asr_old.empty or not df_asr_new.empty else pd.DataFrame()
    #     if not df_asr.empty:
    #         df_asr = df_asr.drop_duplicates(subset=["model", "variation", "prompt_id", "place", "strength"])
    #         if _want_metric("unsafe_score") and "unsafe_score" in df_asr.columns:
    #             domain_specs.append((df_asr, ["unsafe_score"], "Safety/ASR"))

    # ── Silhouette: pre-aggregated, condition-level bootstrap ─────────────────
    sil_results: dict[str, dict[str, float]] = {}   # model → {place_strength_style: value}
    if _want_metric("silhouette"):
        print("\n=== Loading Silhouette data ===")
        df_sil = _filter_models(load_silhouette_df(variations))
        if not df_sil.empty:
            # Compute condition spread (std across groups) per model — no N dimension
            for model, grp in df_sil.groupby("model"):
                vals = grp["silhouette"].dropna()
                sil_results[str(model)] = float(vals.std()) if len(vals) > 1 else np.nan

            print("\n  Silhouette condition spread (std across style×place×strength):")
            for m, v in sorted(sil_results.items()):
                print(f"    {m}: {v:.4f}")

    if not domain_specs and not sil_results:
        print("No data loaded in any domain — nothing to plot.", file=sys.stderr)
        sys.exit(1)

    # all_metric_results : metric → model → {n: [corrs]}
    # all_var_results    : metric → model → {n: scalar spread}
    all_metric_results: dict = {}
    all_var_results:    dict = {}

    for df_domain, metrics, domain_name in domain_specs:
        print(f"\n{'='*50}\n  Domain: {domain_name}")
        for metric in metrics:
            print(f"\n  ── metric: {metric}")
            model_vals = bootstrap_metric_per_model(
                df_domain, metric, SUBSET_SIZES,
                n_bootstrap=args.n_bootstrap, rng=rng,
            )
            all_metric_results[metric] = model_vals

            model_vars = bootstrap_variance_per_model(
                df_domain, metric, SUBSET_SIZES,
                n_bootstrap=args.n_bootstrap, rng=rng,
            )
            all_var_results[metric] = model_vars

            for model in sorted(model_vals):
                val_row = "  ".join(
                    f"N={n}: "
                    f"mean={np.nanmean(model_vals[model].get(n, [float('nan')])):.4f} "
                    f"bstd={model_vars[model].get(n, float('nan')):.4f}"
                    for n in SUBSET_SIZES
                )
                print(f"    [{model}] {val_row}")

    # ── Place ablation ────────────────────────────────────────────────────────
    print("\n=== Computing place ablation ===")
    all_place_means: dict = {}
    all_place_vars: dict = {}
    for df_domain, metrics, _ in domain_specs:
        for metric in metrics:
            pm, pv = compute_place_stats(df_domain, metric)
            if pm:
                all_place_means[metric] = pm
                all_place_vars[metric] = pv

    # ── Strength ablation (per variation, since strength scales differ) ────────────
    print("\n=== Computing strength ablation ===")
    # strength_results[variation] = (means_by_metric, vars_by_metric, sorted_strengths)
    strength_results: dict = {}
    for variation in variations:
        sm_by_metric: dict = {}
        sv_by_metric: dict = {}
        all_s: list = []
        for df_domain, metrics, _ in domain_specs:
            if "variation" not in df_domain.columns:
                continue
            df_variation = df_domain[df_domain["variation"] == variation]
            if df_variation.empty:
                continue
            for metric in metrics:
                sm, sv, s_vals = compute_strength_stats(df_variation, metric)
                if sm:
                    sm_by_metric[metric] = sm
                    sv_by_metric[metric] = sv
                    all_s = sorted(set(all_s) | set(s_vals))
        if sm_by_metric and all_s:
            strength_results[variation] = (sm_by_metric, sv_by_metric, all_s)
            print(f"  {variation}: strengths = {all_s}")

    # ── Save place/strength ablation cache ────────────────────────────────────
    with open(cache_pkl, "wb") as _f:
        pickle.dump({
            "all_place_means":  all_place_means,
            "all_place_vars":   all_place_vars,
            "strength_results": strength_results,
        }, _f)
    print(f"\nSaved place/strength cache → {cache_pkl}")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    rows = []
    for metric, model_vals in all_metric_results.items():
        for model, size_dict in model_vals.items():
            for n in SUBSET_SIZES:
                v = np.array(size_dict.get(n, [float("nan")]), dtype=float)
                rows.append(dict(
                    metric=metric, model=model, n=n,
                    mean_rho=np.nanmean(v),
                    p10_rho=np.nanpercentile(v, 10),
                    p90_rho=np.nanpercentile(v, 90),
                    std_rho=np.nanstd(v),
                    bootstrap_std=all_var_results.get(metric, {}).get(model, {}).get(n, np.nan),
                ))
    # Silhouette: add as N=all (no per-prompt bootstrap possible)
    for model, spread in sil_results.items():
        rows.append(dict(
            metric="silhouette", model=model, n="all",
            mean_rho=np.nan, p10_rho=np.nan, p90_rho=np.nan, std_rho=np.nan,
            bootstrap_std=spread,
        ))

    df_summary = pd.DataFrame(rows)
    csv_out = out_dir / "sensitivity_summary.csv"
    # Append to existing if filtering (don't overwrite other metrics)
    if (metrics_filter or models_filter) and csv_out.exists():
        df_existing = pd.read_csv(csv_out, dtype={"n": str})
        df_summary["n"] = df_summary["n"].astype(str)
        # Drop rows that will be replaced
        mask = df_existing["metric"].isin(df_summary["metric"].unique())
        if models_filter:
            mask &= df_existing["model"].isin(models_filter)
        df_existing = df_existing[~mask]
        df_summary = pd.concat([df_existing, df_summary], ignore_index=True)
    df_summary.to_csv(csv_out, index=False)
    print(f"\nSaved summary table → {csv_out}")

    # ── Plots — always rebuild from full CSV so all models appear ─────────────
    df_plot = pd.read_csv(csv_out, dtype={"n": str})
    df_plot_num = df_plot[df_plot["n"] != "all"].copy()
    df_plot_num["n"] = pd.to_numeric(df_plot_num["n"])

    # Reconstruct plot_metric_results and plot_var_results from the full CSV.
    # Store pre-computed (mean, lo, hi) tuples so _band can use them directly.
    plot_metric_results: dict = {}
    plot_var_results:    dict = {}
    for metric, grp_m in df_plot_num.groupby("metric"):
        plot_metric_results[metric] = {}
        plot_var_results[metric]    = {}
        for model, grp_mod in grp_m.groupby("model"):
            # Value is a 3-tuple (mean, p10, p90) — _band handles both list and tuple
            plot_metric_results[metric][model] = {
                int(row["n"]): (row["mean_rho"], row["p10_rho"], row["p90_rho"])
                for _, row in grp_mod.iterrows()
            }
            plot_var_results[metric][model] = {
                int(row["n"]): row["bootstrap_std"]
                for _, row in grp_mod.iterrows()
            }

    if plot_metric_results:
        print("\nGenerating N-ablation plots …")
        plot_metric_per_model(plot_metric_results, SUBSET_SIZES, out_dir / "metric_ablation.png")
        plot_metric_per_model_pub(plot_metric_results, SUBSET_SIZES, out_dir / "metric_ablation_pub.png")
        plot_variance_per_metric(plot_var_results, SUBSET_SIZES, out_dir / "variance_per_metric.png")
        plot_variance_per_metric_pub(plot_var_results, SUBSET_SIZES, out_dir / "variance_per_metric_pub.png")

    if all_place_means:
        print("\nGenerating place-ablation plots …")
        plot_place_metric(all_place_means,  out_dir / "place_ablation_metric.png")
        plot_place_metric_pub(all_place_means, out_dir / "place_ablation_metric_pub.png")
        plot_place_variance(all_place_vars, out_dir / "place_ablation_variance.png")
        plot_place_variance_pub(all_place_vars, out_dir / "place_ablation_variance_pub.png")

    if strength_results:
        print("\nGenerating strength-ablation plots …")
        for variation, (sm, sv, all_s) in strength_results.items():
            plot_strength_metric(sm,  all_s, out_dir / f"strength_ablation_metric_{variation}.png",  variation=variation)
            plot_strength_metric_pub(sm,  all_s, out_dir / f"strength_ablation_metric_{variation}_pub.png",  variation=variation)
            plot_strength_variance(sv, all_s, out_dir / f"strength_ablation_variance_{variation}.png", variation=variation)
            plot_strength_variance_pub(sv, all_s, out_dir / f"strength_ablation_variance_{variation}_pub.png", variation=variation)

    # ── Decision table: best N per metric ────────────────────────────────────
    df_numeric = df_summary[df_summary["n"] != "all"]
    if not df_numeric.empty:
        df_numeric = df_numeric.copy()
        df_numeric["n"] = pd.to_numeric(df_numeric["n"])
        print("\n── Best N per metric (smallest N within 15% of N=128 for all models) ──")
        for metric in sorted(df_numeric["metric"].unique()):
            sub_m = df_numeric[df_numeric["metric"] == metric]
            best_ns_per_model = {}
            for model, sub_mod in sub_m.groupby("model"):
                size_dict = {int(row["n"]): [row["mean_rho"]] for _, row in sub_mod.iterrows()}
                best_ns_per_model[model] = _find_best_n(size_dict, SUBSET_SIZES)
            best_n = max(best_ns_per_model.values())
            models_str = ", ".join(f"{m}={n}" for m, n in sorted(best_ns_per_model.items()))
            print(f"  {metric:30s}: recommended N={best_n}  ({models_str})")
    print("\nDone.")


if __name__ == "__main__":
    main()
