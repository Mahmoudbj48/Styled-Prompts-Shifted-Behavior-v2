"""
utils/ — shared utilities for the styled-prompts experiment suite.

Provides:
  - data:             dataset loading helpers (TruthfulQA, MMLU, Alpaca, BBQ, HarmBench, GSM8K, NQ)
  - models:           model loading and batched text generation via HuggingFace
  - styles:           prompt style transformation functions (politeness, spacing, punctuation, etc.)
  - metrics:          evaluation metrics (similarity, confidence, activations, safety, etc.)
  - plots:            NeurIPS-style visualisation utilities (moved to Plots/)
  - latex_plots:      LaTeX figure string generation for paper publication
  - llm_style_cache:  caching layer for LLM-generated prompt rewrites
"""
