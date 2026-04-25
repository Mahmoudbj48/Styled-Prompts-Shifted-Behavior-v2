"""
experiments/run_closed_models.py
=================================
Run experiments on closed API models (GPT, Gemini, Claude) across all styles.

Usage:
    python experiments/run_closed_models.py \\
        --models gpt-5.4 gemini-2.5-flash claude-sonnet-4-6 \\
        --datasets truthful_qa alpaca simpleqa_verified \\
        --sample_size 16

Features:
    - Supports all 6 styles (spacing, punctuation, letter_case, politeness, length_variation, inter_vs_imper)
    - Computes BLEU, BERTScore, Confidence (when available), Mirroring
    - Uses cached prompts from existing experiments
    - Saves per-experiment results in separate folders
"""

import argparse
import os
import sys
import json
import yaml
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import get_llm_client
from utils.closed_model_metrics import compute_all_metrics
from utils.data import load_dataset_by_name
from utils.styles import (
    apply_spacing,
    apply_punctuation,
    apply_letter_case,
    apply_politeness,
    apply_length_variation,
    apply_interrogative,
)


# ══════════════════════════════════════════════════════════════════════════════
# STYLE CONFIGURATIONS
# ══════════════════════════════════════════════════════════════════════════════

STYLE_CONFIGS = {
    "spacing": {
        "strengths": [0, 1, 5, 20, 50, 100],
        "place": "global",
        "has_mirroring": True,
        "apply_fn": lambda text, strength, place: apply_spacing(text, strength, place),
    },
    "punctuation": {
        "strengths": [0, 1, 3, 5, 10, 20],
        "place": "global",
        "has_mirroring": True,
        "apply_fn": lambda text, strength, place: apply_punctuation(text, strength, place),
    },
    "letter_case": {
        "strengths": [0, 10, 25, 50, 75, 100],
        "place": "global",
        "has_mirroring": True,
        "apply_fn": lambda text, strength, place: apply_letter_case(text, strength, place),
    },
    "politeness": {
        "strengths": [-10, -6, -2, 0, 2, 6, 10],
        "place": "global",
        "has_mirroring": True,
        "apply_fn": lambda text, strength, place: apply_politeness(text, strength, place),
    },
    "length_variation": {
        "strengths": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
        "place": "global",
        "has_mirroring": True,
        "apply_fn": lambda text, strength, place: apply_length_variation(text, strength),
    },
    "inter_vs_imper": {
        "strengths": ["interrogative", "imperative"],
        "place": "global",
        "has_mirroring": False,
        "apply_fn": lambda text, strength, place: apply_interrogative(text, mode=strength),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_config() -> Dict[str, Any]:
    """Load project config.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_prompt_text(item: Dict[str, Any]) -> str:
    """Extract prompt text from dataset item"""
    if "question" in item and item["question"]:
        return str(item["question"])
    if "prompt" in item and item["prompt"]:
        return str(item["prompt"])
    return str(item)


def load_dataset_prompts(
    dataset_name: str,
    config: Dict[str, Any],
    sample_size: int,
) -> List[Dict[str, Any]]:
    """Load dataset prompts (uses same cache as local experiments)"""
    
    if dataset_name not in config["datasets"]:
        raise ValueError(f"Dataset '{dataset_name}' not found in config.yaml")
    
    dataset_config = config["datasets"][dataset_name]
    
    items = load_dataset_by_name(
        dataset_name,
        sample_size=sample_size,
        seed=int(config["defaults"].get("random_seed", 42)),
        config_name=dataset_config.get("config_name"),
        split=dataset_config.get("split", "validation"),
    )
    
    return items


def create_output_dir(model: str, dataset: str, style: str) -> str:
    """Create output directory for this experiment"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "results",
        "closed_models",
    )
    
    # Sanitize model name for directory
    model_safe = model.replace(".", "_").replace("-", "_")
    
    run_dir = os.path.join(
        base_dir,
        f"run_{model_safe}_{dataset}_{style}_{timestamp}",
    )
    
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir


# ══════════════════════════════════════════════════════════════════════════════
# CORE EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_style_experiment(
    model_name: str,
    dataset_name: str,
    style_name: str,
    items: List[Dict[str, Any]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    judge_provider: str = "openai",
    judge_model: str = "gpt-4o-mini",
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Run one style experiment for one model on one dataset.
    
    Returns DataFrame with all results.
    """
    
    print(f"\n{'='*80}")
    print(f"Running: {model_name} | {dataset_name} | {style_name}")
    print(f"{'='*80}\n")
    
    # Get style config
    if style_name not in STYLE_CONFIGS:
        raise ValueError(f"Unknown style: {style_name}")
    
    style_config = STYLE_CONFIGS[style_name]
    strengths = style_config["strengths"]
    place = style_config["place"]
    apply_fn = style_config["apply_fn"]
    has_mirroring = style_config["has_mirroring"]
    
    # Initialize LLM client
    client = get_llm_client(model_name)
    
    print(f"Model: {model_name}")
    print(f"Supports logprobs: {client.supports_logprobs}")
    print(f"Dataset: {dataset_name} ({len(items)} prompts)")
    print(f"Style: {style_name}")
    print(f"Strengths: {strengths}")
    print(f"Place: {place}")
    print(f"Has mirroring: {has_mirroring}\n")
    
    # Extract prompts
    prompts = [get_prompt_text(item) for item in items]
    categories = [item.get("category", "Unknown") for item in items]
    
    rows = []
    
    # Progress bar
    total = len(prompts) * len(strengths)
    pbar = tqdm(total=total, desc=f"{style_name}", unit="prompt")
    
    for strength in strengths:
        # Apply style to all prompts
        styled_prompts = [apply_fn(p, strength, place) for p in prompts]
        
        for i, (prompt_orig, prompt_styled) in enumerate(zip(prompts, styled_prompts)):
            
            # Generate baseline response
            baseline_resp = client.complete(
                prompt_orig,
                max_tokens=max_tokens,
                temperature=temperature,
                return_logprobs=client.supports_logprobs,
            )
            
            # Generate styled response
            styled_resp = client.complete(
                prompt_styled,
                max_tokens=max_tokens,
                temperature=temperature,
                return_logprobs=client.supports_logprobs,
            )
            
            # Compute metrics
            metrics = compute_all_metrics(
                baseline_prompt=prompt_orig,
                styled_prompt=prompt_styled,
                baseline_response=baseline_resp["text"],
                styled_response=styled_resp["text"],
                baseline_logprobs=baseline_resp["logprobs"],
                styled_logprobs=styled_resp["logprobs"],
                style=style_name,
                strength=strength,
                place=place,
                compute_confidence=client.supports_logprobs,
                compute_similarity=True,
                compute_mirroring_metric=has_mirroring,
                judge_provider=judge_provider,
                judge_model=judge_model,
                device=device,
            )
            
            # Build row
            row = {
                "model": model_name,
                "dataset": dataset_name,
                "style": style_name,
                "prompt_id": i,
                "place": place,
                "strength": strength,
                "category": categories[i],
                "prompt_orig": prompt_orig,
                "prompt_styled": prompt_styled,
                "response_baseline": baseline_resp["text"],
                "response_styled": styled_resp["text"],
                "tokens_baseline": baseline_resp["tokens_used"],
                "tokens_styled": styled_resp["tokens_used"],
            }
            
            # Add metrics
            row.update(metrics)
            
            rows.append(row)
            pbar.update(1)
    
    pbar.close()
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run closed-model experiments across all styles"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names (e.g., gpt-5.4 gemini-2.5-flash claude-sonnet-4-6)",
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names (e.g., truthful_qa alpaca simpleqa_verified)",
    )
    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=16,
        help="Number of prompts per dataset (default: 16)",
    )
    
    parser.add_argument(
        "--styles",
        nargs="+",
        default=None,
        help="Specific styles to run (default: all)",
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Max tokens for LLM responses (default: 512)",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    
    parser.add_argument(
        "--judge_provider",
        type=str,
        default="openai",
        choices=["openai", "gemini"],
        help="LLM provider for mirroring judge (default: openai)",
    )
    
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4o-mini",
        help="Model for mirroring judge (default: gpt-4o-mini)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for BERTScore computation (default: cpu)",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Determine which styles to run
    styles_to_run = args.styles if args.styles else list(STYLE_CONFIGS.keys())
    
    print(f"\n{'='*80}")
    print("CLOSED MODEL EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Styles: {styles_to_run}")
    print(f"Sample size: {args.sample_size}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Mirroring judge: {args.judge_provider}/{args.judge_model}")
    print(f"{'='*80}\n")
    
    # Run experiments
    for model_name in args.models:
        print(f"\n{'#'*80}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*80}\n")
        
        for dataset_name in args.datasets:
            print(f"\n{'='*80}")
            print(f"DATASET: {dataset_name}")
            print(f"{'='*80}\n")
            
            # Load dataset prompts
            items = load_dataset_prompts(dataset_name, config, args.sample_size)
            
            for style_name in styles_to_run:
                # Create output directory
                output_dir = create_output_dir(model_name, dataset_name, style_name)
                
                # Run experiment
                df = run_style_experiment(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    style_name=style_name,
                    items=items,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    judge_provider=args.judge_provider,
                    judge_model=args.judge_model,
                    device=args.device,
                )
                
                # Save results
                output_csv = os.path.join(output_dir, "full_results_all_models.csv")
                df.to_csv(output_csv, index=False)
                
                print(f"\n✓ Saved: {output_csv}")
                print(f"  Rows: {len(df)}")
                print(f"  Columns: {len(df.columns)}\n")
                
                # Save metadata
                metadata = {
                    "model": model_name,
                    "dataset": dataset_name,
                    "style": style_name,
                    "sample_size": args.sample_size,
                    "num_prompts": len(items),
                    "num_strengths": len(STYLE_CONFIGS[style_name]["strengths"]),
                    "total_rows": len(df),
                    "timestamp": datetime.now().isoformat(),
                    "config": {
                        "max_tokens": args.max_tokens,
                        "temperature": args.temperature,
                        "judge_provider": args.judge_provider,
                        "judge_model": args.judge_model,
                    },
                }
                
                metadata_path = os.path.join(output_dir, "metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()