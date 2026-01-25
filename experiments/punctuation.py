"""
Punctuation Style Experiment

Tests how adding random punctuation marks affects model behavior.

Usage:
    python experiments/punctuation.py --model llama --dataset truthful_qa --sample_size 128
    python experiments/punctuation.py --model gemma --dataset truthful_qa --sample_size 10
"""

import argparse
import os
import sys
import yaml
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_dataset_by_name
from utils.models import load_model, generate_response
from utils.metrics import compute_bleu, compute_bertscore, compute_confidence
from utils.metrics import get_layer_activations, reduce_activations_2d
from utils.styles import apply_punctuation


def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_experiment(model_name, dataset_name, sample_size=None):
    """
    Run punctuation experiment: test how random punctuation affects model behavior.
    
    Args:
        model_name (str): Model identifier from config (e.g., 'llama', 'gemma', 'qwen')
        dataset_name (str): Dataset name (e.g., 'truthful_qa', 'mmlu')
        sample_size (int): Number of prompts to test (None = use config default)
    """
    print(f"\n{'='*70}")
    print(f"PUNCTUATION EXPERIMENT")
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
    strength_levels = config['style_levels']['punctuation']
    
    print(f"Configuration:")
    print(f"  - Model Path: {model_path}")
    print(f"  - Sample Size: {sample_size}")
    print(f"  - Strength Levels: {strength_levels}")
    print(f"  - Max New Tokens: {config['defaults']['max_new_tokens']}\n")
    
    # Create organized output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    style_dir = os.path.join(base_results_dir, "punctuation")
    run_dir = os.path.join(style_dir, f"run_{model_name}_{dataset_name}_{timestamp}")
    plot_dir = os.path.join(run_dir, "plots")
    
    # Create all directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Output directory: {run_dir}\n")
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model(
        model_path,
        device_map=config['defaults']['device_map'],
        dtype=config['defaults']['dtype']
    )
    print(f"Running on device: {model.device}\n")
    
    # Load dataset
    print("Loading dataset...")
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
    
    # Cache for activations (for 2D visualization later)
    activations_cache = {s: [] for s in strength_levels}
    
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
        
        # Get activation for original prompt (once per prompt)
        act_orig = get_layer_activations(model, tokenizer, prompt_orig, layer_idx=-1)
        
        # Test each strength level
        for strength in strength_levels:
            # Apply punctuation style
            prompt_pert = apply_punctuation(prompt_orig, strength)
            
            # Generate perturbed response
            response_pert = generate_response(
                model, tokenizer, prompt_pert,
                max_new_tokens=config['defaults']['max_new_tokens']
            )
            
            # Get activation for perturbed prompt (cached for later)
            act_pert = get_layer_activations(model, tokenizer, prompt_pert, layer_idx=-1)
            activations_cache[strength].append(act_pert)
            
            # Compute activation similarity
            activation_similarity = F.cosine_similarity(
                act_orig.unsqueeze(0), 
                act_pert.unsqueeze(0)
            ).item()
            
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
                'activation_similarity': activation_similarity,
                'prompt_orig': prompt_orig,
                'prompt_pert': prompt_pert,
                'response_orig': response_orig,
                'response_pert': response_pert
            })
            
            pbar.update(1)
    
    pbar.close()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save full results
    output_path = os.path.join(run_dir, "full_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✓ Full results saved to: {output_path}")
    
    # Create and save summary statistics
    summary = df.groupby('strength')[
        ['bleu', 'bertscore', 'delta_log_prob', 'entropy_shift', 'jsd_drift', 'activation_similarity']
    ].agg(['mean', 'std']).reset_index()
    summary_path = os.path.join(run_dir, "summary_statistics.csv")
    summary.to_csv(summary_path, index=False)
    print(f"✓ Summary statistics saved to: {summary_path}")
    
    # === Generate 2D Activation Visualizations ===
    print("\nGenerating 2D activation visualizations...")
    
    # Sample size for visualization (to avoid cluttering)
    viz_sample_size = min(50, len(prompts))
    
    # Save 2D coordinates for each strength level
    for strength in strength_levels:
        activations = activations_cache[strength][:viz_sample_size]
        
        if len(activations) > 1:  # Need at least 2 points
            # Reduce to 2D using tsne
            coords_2d = reduce_activations_2d(activations, method='tsne', seed=config['defaults']['random_seed'])
            
            # Save coordinates
            coords_df = pd.DataFrame({
                'prompt_id': list(range(len(coords_2d))),
                'strength': strength,
                'x': coords_2d[:, 0],
                'y': coords_2d[:, 1]
            })
            coords_path = os.path.join(run_dir, f"activation_coords_strength_{strength}.csv")
            coords_df.to_csv(coords_path, index=False)
    
    print(f"✓ 2D activation coordinates saved for each strength level")
    
    # Create combined visualization plot
    fig, ax = plt.subplots(figsize=(12, 8))
    colors_map = plt.cm.viridis(np.linspace(0, 1, len(strength_levels)))
    
    for idx, strength in enumerate(strength_levels):
        activations = activations_cache[strength][:viz_sample_size]
        
        if len(activations) > 1:
            coords_2d = reduce_activations_2d(activations, method='tsne', seed=config['defaults']['random_seed'])
            
            ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                       label=f'Strength {strength}', 
                       alpha=0.6, 
                       color=colors_map[idx],
                       s=50)
    
    ax.set_xlabel("t-SNE Component 1", fontsize=12)
    ax.set_ylabel("t-SNE Component 2", fontsize=12)
    ax.set_title(f"Activation Space (Last Layer) - Punctuation Style\nModel: {model_name}", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plot_file = os.path.join(plot_dir, "activation_space_2d.png")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ 2D activation plot saved to: {plot_file}")
    
    # Print summary to console
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS (Mean)")
    print(f"{'='*70}")
    
    summary_display = df.groupby('strength')[
        ['bleu', 'bertscore', 'delta_log_prob', 'entropy_shift', 'jsd_drift', 'activation_similarity']
    ].mean()
    print(summary_display.round(3).to_string())
    print(f"{'='*70}\n")
    
    print(f"✓ All outputs saved to: {run_dir}\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run punctuation style experiment")
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