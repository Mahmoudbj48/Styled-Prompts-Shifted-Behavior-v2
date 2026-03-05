"""
BBQ Bias Evaluation — Proof of Concept

Tests style sensitivity on demographic bias using BBQ (Bias Benchmark for QA).

Proof-of-Concept Mode:
    - Single model (Llama 3.1-8B)
    - Gender_identity category only
    - Small sample (32 examples)
    - Subset of strengths (0, 50, 100)
    - Global placement only

Full Evaluation Mode:
    - All strengths from config
    - All placements (global, prefix, suffix)
    - Larger sample (128+ examples)

Usage:
    # Proof of Concept (Quick Test)
    python experiments/bbq_bias_poc.py --model L3.1-8B --sample_size 32 --strengths 0 50 100 --global-only
    
    # Full Evaluation (All Strengths/Placements)
    python experiments/bbq_bias_poc.py --model L3.1-8B --sample_size 128
    
    # Resume interrupted run
    python experiments/bbq_bias_poc.py --model L3.1-8B --sample_size 32 --resume
"""

import os
import sys
import yaml
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_bbq_hf
from utils.metrics import compute_bias_score_bbq
from utils.models import load_model
from utils.styles import apply_spacing, apply_punctuation, apply_letter_case, apply_politeness


# Only Gender_identity for PoC
BBQ_CATEGORIES = ["Gender_identity"]


def load_config(config_path="config.yaml"):
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_style_function(style_name):
    """Map style name to function."""
    style_map = {
        'spacing': apply_spacing,
        'punctuation': apply_punctuation,
        'letter_case': apply_letter_case,
        'politeness': apply_politeness
    }
    return style_map.get(style_name)


def load_all_bbq_data(categories, sample_size=32, seed=42):
    """Load BBQ data for specified categories (ambiguous only)."""
    print("="*80)
    print("LOADING BBQ DATA")
    print("="*80)
    
    all_data = {}
    
    for category in tqdm(categories, desc="Loading categories"):
        try:
            examples = load_bbq_hf(
                sample_size=sample_size * 4,  # Load extra to ensure enough after filtering
                category=category,
                seed=seed,
                split='test'
            )
            
            # Filter for ambiguous context only
            ambig_examples = [
                ex for ex in examples 
                if ex['meta'].get('_bbq_config', '').lower().endswith('ambig')
            ]
            
            all_data[category] = ambig_examples[:sample_size]
            print(f"  {category}: {len(all_data[category])} examples")
            
        except Exception as e:
            print(f"  WARNING: Could not load {category}: {e}")
            all_data[category] = []
    
    return all_data


def create_experiment_configs(config, model_alias, strength_subset=None, global_only=False):
    """
    Generate experiment configurations for a SINGLE MODEL.
    
    Args:
        config: Full config dict
        model_alias: Model alias to run (e.g., 'L3.1-8B')
        strength_subset: List of strengths to test (e.g., [0, 50, 100])
        global_only: If True, only test 'global' placement
    
    Returns:
        list: Experiment configs
    """
    experiments = []
    
    if model_alias not in config['models']:
        raise ValueError(f"Model '{model_alias}' not found in config. Available: {list(config['models'].keys())}")
    
    model_path = config['models'][model_alias]
    style_levels = config['style_levels']
    style_positions = config['style_positions']
    
    # For each style
    for style_name in ['spacing', 'punctuation', 'letter_case', 'politeness']:
        strengths = style_levels.get(style_name, [0])
        
        # Filter strengths if subset provided
        if strength_subset is not None:
            strengths = [s for s in strengths if s in strength_subset]
        
        placements = style_positions.get(style_name, ['global'])
        
        # Filter to global only if requested
        if global_only:
            placements = ['global']
        
        # For each strength
        for strength in strengths:
            # For each placement
            for placement in placements:
                # For each category
                for category in BBQ_CATEGORIES:
                    experiments.append({
                        'model_alias': model_alias,
                        'model_path': model_path,
                        'style': style_name,
                        'strength': strength,
                        'placement': placement,
                        'category': category
                    })
    
    return experiments


def run_single_experiment(model, tokenizer, bbq_data, exp_config, results_csv, max_new_tokens=50):
    """Run a single experiment and append to CSV."""
    category = exp_config['category']
    style_name = exp_config['style']
    strength = exp_config['strength']
    placement = exp_config['placement']
    
    examples = bbq_data.get(category, [])
    
    if not examples:
        print(f"    WARNING: No examples for {category}")
        return
    
    style_fn_base = get_style_function(style_name)
    if style_fn_base is None:
        print(f"    WARNING: Unknown style {style_name}")
        return
    
    style_fn = lambda prompt: style_fn_base(prompt, strength, place=placement)
    
    try:
        results = compute_bias_score_bbq(
            model,
            tokenizer,
            examples,
            style_fn=style_fn,
            max_new_tokens=max_new_tokens
        )
        
        summary_row = pd.DataFrame([{
            'model_alias': exp_config['model_alias'],
            'model_path': exp_config['model_path'],
            'category': category,
            'style': style_name,
            'strength': strength,
            'placement': placement,
            'bias_score': results['bias_score'],
            'accuracy': results['accuracy'],
            'num_examples': results['num_examples'],
            'num_biased': results['num_biased'],
            'raw_bias': results['by_context']['ambig']['raw_bias'],
            'neg_target': results['by_context']['ambig']['counts']['neg_target'],
            'neg_non_target': results['by_context']['ambig']['counts']['neg_non_target'],
            'nonneg_target': results['by_context']['ambig']['counts']['nonneg_target'],
            'nonneg_non_target': results['by_context']['ambig']['counts']['nonneg_non_target'],
            'total_non_unknown': results['by_context']['ambig']['counts']['total_non_unknown'],
        }])
        
        if not os.path.exists(results_csv):
            summary_row.to_csv(results_csv, index=False, mode='w')
        else:
            summary_row.to_csv(results_csv, index=False, mode='a', header=False)
        
        print(f"    Bias: {results['bias_score']:>7.2f} | Acc: {results['accuracy']:.3f}")
        
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()


def check_existing_experiments(results_csv):
    """Check which experiments have already been run."""
    if not os.path.exists(results_csv):
        return set()
    
    try:
        df = pd.read_csv(results_csv)
        completed = set(
            tuple(row) for row in 
            df[['model_alias', 'category', 'style', 'strength', 'placement']].values
        )
        return completed
    except:
        return set()


def plot_bias_results(results_csv, out_dir):
    """Generate bias score plots."""
    df = pd.read_csv(results_csv)
    
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Apply clean style
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    
    # Plot 1: Bias score by style
    fig, ax = plt.subplots(figsize=(10, 6))
    
    styles = sorted(df['style'].unique())
    colors = plt.cm.tab10(range(len(styles)))
    
    for i, style in enumerate(styles):
        subset = df[df['style'] == style].sort_values('strength')
        if len(subset) > 0:
            ax.plot(subset['strength'], subset['bias_score'], 
                    marker='o', label=style, linewidth=2.5, 
                    markersize=8, color=colors[i])
    
    ax.set_xlabel('Style Strength', fontsize=13, fontweight='bold')
    ax.set_ylabel('Bias Score', fontsize=13, fontweight='bold')
    ax.set_title('BBQ Bias Score vs Style Strength (Gender Identity)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'bias_by_style.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Bias score by placement (if multiple placements)
    if len(df['placement'].unique()) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        placements = sorted(df['placement'].unique())
        colors = plt.cm.Set2(range(len(placements)))
        
        for i, placement in enumerate(placements):
            subset = df[df['placement'] == placement].groupby('strength')['bias_score'].mean()
            if len(subset) > 0:
                ax.plot(subset.index, subset.values, 
                        marker='o', label=placement, linewidth=2.5, 
                        markersize=8, color=colors[i])
        
        ax.set_xlabel('Style Strength', fontsize=13, fontweight='bold')
        ax.set_ylabel('Average Bias Score', fontsize=13, fontweight='bold')
        ax.set_title('BBQ Bias Score by Placement (Averaged Across Styles)', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'bias_by_placement.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Heatmap (style x strength)
    pivot = df.groupby(['style', 'strength'])['bias_score'].mean().unstack()
    
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
        
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        
        ax.set_xlabel('Strength', fontsize=12, fontweight='bold')
        ax.set_ylabel('Style', fontsize=12, fontweight='bold')
        ax.set_title('BBQ Bias Score Heatmap', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Bias Score', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'bias_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\n✓ Plots saved to {plots_dir}/")


def print_summary_statistics(results_csv):
    """Print summary statistics."""
    df = pd.read_csv(results_csv)
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal experiments: {len(df)}")
    print(f"Categories: {df['category'].unique().tolist()}")
    print(f"Styles: {df['style'].unique().tolist()}")
    print(f"Strengths: {sorted(df['strength'].unique().tolist())}")
    print(f"Placements: {df['placement'].unique().tolist()}")
    
    print("\n" + "-"*80)
    print("Bias Score by Style:")
    print("-"*80)
    style_summary = df.groupby('style')['bias_score'].agg(['mean', 'std', 'min', 'max'])
    print(style_summary.to_string())
    
    print("\n" + "-"*80)
    print("Bias Score by Strength:")
    print("-"*80)
    strength_summary = df.groupby('strength')['bias_score'].agg(['mean', 'std', 'min', 'max'])
    print(strength_summary.to_string())
    
    if len(df['placement'].unique()) > 1:
        print("\n" + "-"*80)
        print("Bias Score by Placement:")
        print("-"*80)
        placement_summary = df.groupby('placement')['bias_score'].agg(['mean', 'std', 'min', 'max'])
        print(placement_summary.to_string())


def main():
    parser = argparse.ArgumentParser(description="BBQ bias evaluation - Proof of Concept")
    parser.add_argument('--model', type=str, required=True, 
                        help='Model alias from config (e.g., L3.1-8B)')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume from existing results')
    parser.add_argument('--sample_size', type=int, default=32, 
                        help='Samples per category (default: 32 for PoC)')
    parser.add_argument('--strengths', nargs='+', type=int, default=None,
                        help='Subset of strengths to test (e.g., --strengths 0 50 100)')
    parser.add_argument('--global-only', action='store_true',
                        help='Only test global placement (skip prefix/suffix)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Validate model exists
    if args.model not in config['models']:
        print(f"ERROR: Model '{args.model}' not found in config.")
        print(f"Available models: {list(config['models'].keys())}")
        sys.exit(1)
    
    model_path = config['models'][args.model]
    
    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    poc_suffix = "_poc" if args.sample_size <= 64 else ""
    results_dir = f"results/bbq_bias_gender/{args.model}{poc_suffix}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    results_csv = os.path.join(results_dir, "bias_scores.csv")
    
    print("\n" + "="*80)
    print(f"BBQ BIAS EVALUATION - {args.model}")
    if args.sample_size <= 64:
        print("(PROOF OF CONCEPT MODE)")
    print("="*80)
    print(f"Model: {args.model} ({model_path})")
    print(f"Category: Gender_identity")
    print(f"Sample size: {args.sample_size}")
    if args.strengths:
        print(f"Strengths: {args.strengths}")
    if args.global_only:
        print(f"Placement: global only")
    print(f"Results: {results_dir}")
    
    # Load BBQ data once
    bbq_data = load_all_bbq_data(BBQ_CATEGORIES, sample_size=args.sample_size, seed=42)
    
    # Generate experiment configurations
    experiments = create_experiment_configs(
        config, 
        args.model, 
        strength_subset=args.strengths,
        global_only=args.global_only
    )
    total_experiments = len(experiments)
    
    print(f"\nTotal experiments for {args.model}: {total_experiments}")
    
    # Check for existing results
    completed = check_existing_experiments(results_csv) if args.resume else set()
    remaining = total_experiments - len(completed)
    
    if args.resume and completed:
        print(f"\nResuming from existing results:")
        print(f"  Completed: {len(completed)}")
        print(f"  Remaining: {remaining}")
    
    # Load model
    print(f"\n{'='*80}")
    print(f"LOADING MODEL: {args.model}")
    print(f"{'='*80}")
    
    try:
        model, tokenizer = load_model(
            model_path,
            device_map=config['defaults']['device_map'],
            dtype=config['defaults']['dtype']
        )
        print(f"✓ Model loaded on {model.device}")
    except Exception as e:
        print(f"✗ ERROR loading model: {e}")
        sys.exit(1)
    
    # Run experiments
    print(f"\n{'='*80}")
    print("RUNNING EXPERIMENTS")
    print(f"{'='*80}")
    
    for i, exp in enumerate(tqdm(experiments, desc=f"{args.model} experiments")):
        # Check if already completed
        exp_key = (exp['model_alias'], exp['category'], exp['style'], exp['strength'], exp['placement'])
        
        if args.resume and exp_key in completed:
            continue
        
        # Print experiment info
        print(f"\n[{i+1}/{total_experiments}] {exp['category']} | {exp['style']} | strength={exp['strength']} | {exp['placement']}")
        
        # Run experiment
        run_single_experiment(
            model,
            tokenizer,
            bbq_data,
            exp,
            results_csv,
            max_new_tokens=config['defaults']['max_new_tokens']
        )
    
    # Generate plots and summary
    if os.path.exists(results_csv):
        plot_bias_results(results_csv, results_dir)
        print_summary_statistics(results_csv)
    
    # Final summary
    print("\n" + "="*80)
    print("✓ EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Results: {results_csv}")
    print(f"Plots: {results_dir}/plots/")


if __name__ == "__main__":
    main()