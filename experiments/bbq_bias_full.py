"""
BBQ Bias Evaluation — Gender Identity Category.

Tests all style/strength/placement combinations for a single model and measures
demographic bias sensitivity using the BBQ (Bias Benchmark for QA) dataset.

Inputs:
    - BBQ Gender_identity category (via utils.data.load_bbq_hf, ambiguous context only)
    - config.yaml for model paths, style levels, and style positions

Outputs (saved to results/bias_bbq_politeness/run_YYYYMMDD_HHMMSS/):
    - bias_results.csv: bias score, accuracy, and raw counts per
      (model, category, style, strength, placement) combination

Run:
  python experiments/bbq_bias_full.py --model L3.2-1B

Important flags:
    --model   Model alias from config.yaml (e.g. L3.2-1B, G-7B)
    --sample_size  Number of BBQ examples per category (default: 128)
    --seed         Random seed (default: 42)
"""

import os
import sys
import yaml
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_bbq_hf
from utils.metrics import compute_bias_score_bbq
from utils.models import load_model
from utils.styles import apply_spacing, apply_punctuation, apply_letter_case, apply_politeness


# Only Gender
BBQ_CATEGORIES = [
    "Gender_identity"
]


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


def load_all_bbq_data(categories, sample_size=128, seed=42):
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
                split='test'  # BBQ uses 'test' split for main data
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


def create_experiment_configs(config, model_alias):
    """
    Generate experiment configurations for a SINGLE MODEL.
    
    Args:
        config: Full config dict
        model_alias: Model alias to run (e.g., 'L3.2-1B')
    
    Returns:
        list: Experiment configs for this model only
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
        placements = style_positions.get(style_name, ['global'])
        
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


def main():
    parser = argparse.ArgumentParser(description="BBQ bias evaluation for single model")
    parser.add_argument('--model', type=str, required=True, 
                        help='Model alias from config (e.g., L3.2-1B, G-2B)')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--resume', action='store_true', help='Resume from existing results')
    parser.add_argument('--sample_size', type=int, default=128, help='Samples per category')
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
    results_dir = f"results/bbq_bias_age_gender/{args.model}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    results_csv = os.path.join(results_dir, "bias_scores.csv")
    
    print("\n" + "="*80)
    print(f"BBQ BIAS EVALUATION - {args.model}")
    print("="*80)
    print(f"Model: {args.model} ({model_path})")
    print(f"Categories: Age, Gender_identity")
    print(f"Sample size: {args.sample_size} per category")
    print(f"Results: {results_dir}")
    
    # Load BBQ data once
    bbq_data = load_all_bbq_data(BBQ_CATEGORIES, sample_size=args.sample_size, seed=42)
    
    # Generate experiment configurations for this model only
    experiments = create_experiment_configs(config, args.model)
    total_experiments = len(experiments)
    
    print(f"\nTotal experiments for {args.model}: {total_experiments}")
    
    # Calculate expected counts
    num_styles = 4
    num_strengths = sum(len(config['style_levels'][s]) for s in ['spacing', 'punctuation', 'letter_case', 'politeness'])
    num_placements = 3
    num_categories = len(BBQ_CATEGORIES)
    
    print(f"  Styles: {num_styles}")
    print(f"  Total strength levels across all styles: {num_strengths}")
    print(f"  Placements: {num_placements}")
    print(f"  Categories: {num_categories}")
    
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
    
    # Final summary
    print("\n" + "="*80)
    print("✓ EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Results: {results_csv}")
    
    # Print summary statistics
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        print(f"\nTotal experiments completed: {len(df)}")
        print(f"Categories tested: {df['category'].unique().tolist()}")
        print(f"Styles tested: {df['style'].unique().tolist()}")
        
        print("\nBias score summary:")
        print(df.groupby('style')['bias_score'].describe()[['mean', 'std', 'min', 'max']])


if __name__ == "__main__":
    main()