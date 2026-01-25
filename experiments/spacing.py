"""
Spacing Style Experiment

Tests how adding random spaces affects model behavior across multiple metrics.

Usage:
    python experiments/spacing.py --model llama --dataset truthful_qa --sample_size 128
    python experiments/spacing.py --model gemma --dataset truthful_qa --sample_size 10
"""

import argparse
import os
import sys
import yaml
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_dataset_by_name
from utils.models import load_model, generate_response
from utils.metrics import compute_bleu, compute_bertscore, compute_confidence
from utils.styles import apply_spacing


def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_experiment(model_name, dataset_name, sample_size=None):
    """
    Run spacing experiment: test how random spacing affects model behavior.
    
    Args:
        model_name (str): Model identifier from config (e.g., 'llama', 'gemma', 'qwen')
        dataset_name (str): Dataset name (e.g., 'truthful_qa', 'mmlu')
        sample_size (int): Number of prompts to test (None = use config default)
    """
    print(f"\n{'='*70}")
    print(f"SPACING EXPERIMENT")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}\n")
    
    # Load config
    config = load_config()
    
    # Get model path from config
    if model_name not in config['models']:
        raise ValueError(f"Model '{model_name}' not found in config. Available: {list(config['models'].keys())}")
    model_path = config['models'][model_name]
    
    # Get dataset config
    if dataset_name not in config['datasets']:
        raise ValueError(f"Dataset '{dataset_name}' not found in config. Available: {list(config['datasets'].keys())}")
    dataset_config = config['datasets'][dataset_name]
    
    if sample_size is None:
        sample_size = dataset_config['sample_size']
    
    # Get style strength levels from config
    strength_levels = config['style_levels']['spacing']
    
    print(f"Configuration:")
    print(f"  - Model Path: {model_path}")
    print(f"  - Sample Size: {sample_size}")
    print(f"  - Strength Levels: {strength_levels}")
    print(f"  - Max New Tokens: {config['defaults']['max_new_tokens']}\n")
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model(
        model_path,
        device_map=config['defaults']['device_map'],
        dtype=config['defaults']['dtype']
    )
    
    # Load dataset
    print("\nLoading dataset...")
    prompts = load_dataset_by_name(
        dataset_name,
        sample_size=sample_size,
        seed=config['defaults']['random_seed'],
        config_name=dataset_config.get('config_name'),
        split=dataset_config.get('split', 'validation')
    )
    
    print(f"\n{'='*70}")
    print(f"Starting experiment: {len(prompts)} prompts × {len(strength_levels)} strength levels")
    print(f"Total iterations: {len(prompts) * len(strength_levels)}")
    print(f"{'='*70}\n")
    
    # Store results
    results = []
    
    # Progress bar
    total_iterations = len(prompts) * len(strength_levels)
    pbar = tqdm(total=total_iterations, desc="Processing", unit="iter")
    
    for i, item in enumerate(prompts):
        prompt_orig = item['question']
        
        # Generate baseline response (original prompt)
        response_orig = generate_response(
            model, tokenizer, prompt_orig,
            max_new_tokens=config['defaults']['max_new_tokens']
        )
        
        # Test each strength level
        for strength in strength_levels:
            # Apply spacing style
            prompt_pert = apply_spacing(prompt_orig, strength)
            
            # Generate perturbed response
            response_pert = generate_response(
                model, tokenizer, prompt_pert,
                max_new_tokens=config['defaults']['max_new_tokens']
            )
            
            # Compute quality metrics
            bleu = compute_bleu(response_orig, response_pert)
            bertscore = compute_bertscore(response_orig, response_pert, device=str(model.device))
            
            # Compute confidence metrics
            conf_metrics = compute_confidence(model, tokenizer, prompt_orig, prompt_pert, response_orig)
            
            # Store result
            results.append({
                'prompt_id': i,
                'strength': strength,
                'category': item.get('category', 'Unknown'),
                'bleu': bleu,
                'bertscore': bertscore,
                'delta_log_prob': conf_metrics['delta_log_prob'],
                'entropy_shift': conf_metrics['entropy_shift'],
                'jsd_drift': conf_metrics['jsd_drift'],
                'prompt_orig': prompt_orig,
                'prompt_pert': prompt_pert,
                'response_orig': response_orig,
                'response_pert': response_pert
            })
            
            pbar.update(1)
    
    pbar.close()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"spacing_{model_name}_{dataset_name}_{timestamp}.csv")
    
    # Save full results
    df.to_csv(output_path, index=False)
    print(f"\n✓ Full results saved to: {output_path}")
    
    # Create and save summary statistics
    summary = df.groupby('strength')[['bleu', 'bertscore', 'delta_log_prob', 'entropy_shift', 'jsd_drift']].agg(['mean', 'std']).reset_index()
    summary_path = os.path.join(output_dir, f"spacing_{model_name}_{dataset_name}_{timestamp}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"✓ Summary statistics saved to: {summary_path}")
    
    # Print summary to console
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS (Mean ± Std)")
    print(f"{'='*70}")
    
    summary_display = df.groupby('strength')[['bleu', 'bertscore', 'delta_log_prob', 'entropy_shift', 'jsd_drift']].mean()
    print(summary_display.round(3).to_string())
    print(f"{'='*70}\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run spacing style experiment")
    parser.add_argument("--model", type=str, default="llama", 
                       help="Model name from config (llama, gemma, qwen)")
    parser.add_argument("--dataset", type=str, default="truthful_qa", 
                       help="Dataset name (truthful_qa, mmlu)")
    parser.add_argument("--sample_size", type=int, default=None, 
                       help="Number of prompts to test (default: from config)")
    
    args = parser.parse_args()
    
    run_experiment(args.model, args.dataset, args.sample_size)


if __name__ == "__main__":
    main()