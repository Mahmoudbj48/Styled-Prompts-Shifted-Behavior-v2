"""
Comprehensive BBQ Bias Evaluation
Tests all combinations of:
- 6 models
- 4 styles (spacing, punctuation, letter_case, politeness)
- Multiple strengths per style
- 3 placements (prefix, suffix, global)
- 9 bias categories
- 128 prompts per category (ambiguous context only)
"""

import os
import sys
import yaml
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_bbq_hf
from utils.metrics import compute_bias_score_bbq
from utils.models import load_model
from utils.styles import apply_spacing, apply_punctuation, apply_letter_case, apply_politeness


# BBQ Categories (all 9)
BBQ_CATEGORIES = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Religion",
    "SES",  # Socio-economic status
    "Sexual_orientation"
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
    """
    Load BBQ data for all categories (ambiguous only).
    
    Returns:
        dict: {category_name: list_of_examples}
    """
    print("="*80)
    print("LOADING BBQ DATA")
    print("="*80)
    
    all_data = {}
    
    for category in tqdm(categories, desc="Loading categories"):
        try:
            # Load ambiguous context only
            examples = load_bbq_hf(
                sample_size=sample_size * 2,  # Load extra, then filter
                category=category,
                seed=seed
            )
            
            # Filter for ambiguous context only
            ambig_examples = [
                ex for ex in examples 
                if ex['meta'].get('_bbq_config', '').lower().endswith('ambig')
            ]
            
            # Take first sample_size
            all_data[category] = ambig_examples[:sample_size]
            
            print(f"  {category}: {len(all_data[category])} ambiguous examples")
            
        except Exception as e:
            print(f"  WARNING: Could not load {category}: {e}")
            all_data[category] = []
    
    total_examples = sum(len(v) for v in all_data.values())
    print(f"\nTotal examples loaded: {total_examples}")
    
    return all_data


def create_experiment_configs(config):
    """
    Generate all experiment configurations.
    
    Returns:
        list: List of dicts, each representing one experiment config
    """
    experiments = []
    
    models = config['models']
    style_levels = config['style_levels']
    style_positions = config['style_positions']
    
    # For each model
    for model_alias, model_path in models.items():
        
        # For each style
        for style_name in ['spacing', 'punctuation', 'letter_case', 'politeness']:
            
            # Get strengths for this style
            strengths = style_levels.get(style_name, [0])
            
            # Get placements for this style
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


def run_single_experiment(
    model, 
    tokenizer, 
    bbq_data, 
    exp_config, 
    results_csv,
    details_csv,
    max_new_tokens=50
):
    """
    Run a single experiment configuration and append results to CSV.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        bbq_data: Dict of {category: examples}
        exp_config: Dict with experiment parameters
        results_csv: Path to summary CSV
        details_csv: Path to detailed CSV
        max_new_tokens: Max tokens for generation
    """
    category = exp_config['category']
    style_name = exp_config['style']
    strength = exp_config['strength']
    placement = exp_config['placement']
    
    # Get examples for this category
    examples = bbq_data.get(category, [])
    
    if not examples:
        print(f"    WARNING: No examples for {category}, skipping...")
        return
    
    # Create style function
    style_fn_base = get_style_function(style_name)
    
    if style_fn_base is None:
        print(f"    WARNING: Unknown style {style_name}, skipping...")
        return
    
    # Wrap with strength and placement
    style_fn = lambda prompt: style_fn_base(prompt, strength, place=placement)
    
    # Compute bias score
    try:
        results = compute_bias_score_bbq(
            model,
            tokenizer,
            examples,
            style_fn=style_fn,
            max_new_tokens=max_new_tokens
        )
        
        # Prepare summary row
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
        
        # Append to summary CSV
        if not os.path.exists(results_csv):
            summary_row.to_csv(results_csv, index=False, mode='w')
        else:
            summary_row.to_csv(results_csv, index=False, mode='a', header=False)
        
        # Prepare detailed results
        details_df = pd.DataFrame(results['details'])
        details_df['model_alias'] = exp_config['model_alias']
        details_df['model_path'] = exp_config['model_path']
        details_df['style'] = style_name
        details_df['strength'] = strength
        details_df['placement'] = placement
        
        # Append to details CSV
        if not os.path.exists(details_csv):
            details_df.to_csv(details_csv, index=False, mode='w')
        else:
            details_df.to_csv(details_csv, index=False, mode='a', header=False)
        
        # Print summary
        print(f"    Bias: {results['bias_score']:>7.2f} | Acc: {results['accuracy']:.3f} | Biased: {results['num_biased']:>3}/{results['num_examples']}")
        
    except Exception as e:
        print(f"    ERROR: {e}")
        # Log error to CSV
        error_row = pd.DataFrame([{
            'model_alias': exp_config['model_alias'],
            'model_path': exp_config['model_path'],
            'category': category,
            'style': style_name,
            'strength': strength,
            'placement': placement,
            'bias_score': None,
            'accuracy': None,
            'num_examples': len(examples),
            'num_biased': None,
            'raw_bias': None,
            'neg_target': None,
            'neg_non_target': None,
            'nonneg_target': None,
            'nonneg_non_target': None,
            'total_non_unknown': None,
            'error': str(e)
        }])
        
        if not os.path.exists(results_csv):
            error_row.to_csv(results_csv, index=False, mode='w')
        else:
            error_row.to_csv(results_csv, index=False, mode='a', header=False)


def check_existing_experiments(results_csv):
    """
    Check which experiments have already been run.
    
    Returns:
        set: Set of tuples (model_alias, category, style, strength, placement)
    """
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
    parser = argparse.ArgumentParser(description="Run comprehensive BBQ bias evaluation")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume from existing results')
    parser.add_argument('--sample_size', type=int, default=128, help='Samples per category')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/bbq_bias_comprehensive/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    results_csv = os.path.join(results_dir, "bias_scores.csv")
    details_csv = os.path.join(results_dir, "detailed_predictions.csv")
    
    print("\n" + "="*80)
    print("BBQ BIAS COMPREHENSIVE EVALUATION")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Sample size per category: {args.sample_size}")
    
    # Load all BBQ data once
    bbq_data = load_all_bbq_data(BBQ_CATEGORIES, sample_size=args.sample_size, seed=config['defaults']['random_seed'])
    
    # Generate experiment configurations
    experiments = create_experiment_configs(config)
    total_experiments = len(experiments)
    
    print(f"\nTotal experiments to run: {total_experiments}")
    print(f"  Models: {len(config['models'])}")
    print(f"  Styles: 4 (spacing, punctuation, letter_case, politeness)")
    print(f"  Categories: {len(BBQ_CATEGORIES)}")
    
    # Check for existing results
    completed = check_existing_experiments(results_csv) if args.resume else set()
    remaining = total_experiments - len(completed)
    
    if args.resume and completed:
        print(f"\nResuming from existing results:")
        print(f"  Completed: {len(completed)}")
        print(f"  Remaining: {remaining}")
    
    # Group experiments by model to avoid reloading
    experiments_by_model = {}
    for exp in experiments:
        model_alias = exp['model_alias']
        if model_alias not in experiments_by_model:
            experiments_by_model[model_alias] = []
        experiments_by_model[model_alias].append(exp)
    
    # Run experiments
    print("\n" + "="*80)
    print("STARTING EXPERIMENTS")
    print("="*80)
    
    global_progress = 0
    
    for model_alias, model_experiments in experiments_by_model.items():
        print(f"\n{'='*80}")
        print(f"MODEL: {model_alias}")
        print(f"{'='*80}")
        
        model_path = config['models'][model_alias]
        
        # Load model
        print(f"Loading model: {model_path}")
        try:
            model, tokenizer = load_model(
                model_path,
                device_map=config['defaults']['device_map'],
                dtype=config['defaults']['dtype']
            )
            print(f"✓ Model loaded on {model.device}")
        except Exception as e:
            print(f"✗ ERROR loading model: {e}")
            print(f"Skipping all experiments for {model_alias}")
            continue
        
        # Run experiments for this model
        for exp in tqdm(model_experiments, desc=f"{model_alias} experiments"):
            global_progress += 1
            
            # Check if already completed
            exp_key = (
                exp['model_alias'],
                exp['category'],
                exp['style'],
                exp['strength'],
                exp['placement']
            )
            
            if args.resume and exp_key in completed:
                continue
            
            # Print experiment info
            print(f"\n[{global_progress}/{total_experiments}] " +
                  f"{exp['model_alias']} | {exp['category']} | " +
                  f"{exp['style']} | strength={exp['strength']} | {exp['placement']}")
            
            # Run experiment
            run_single_experiment(
                model,
                tokenizer,
                bbq_data,
                exp,
                results_csv,
                details_csv,
                max_new_tokens=config['defaults']['max_new_tokens']
            )
        
        # Free model memory
        del model, tokenizer
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\n✓ Completed all experiments for {model_alias}")
    
    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results saved to: {results_dir}")
    print(f"  Summary: {results_csv}")
    print(f"  Details: {details_csv}")
    
    # Print summary statistics
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        print(f"\nTotal experiments completed: {len(df)}")
        print(f"Models tested: {df['model_alias'].nunique()}")
        print(f"Categories tested: {df['category'].nunique()}")
        print(f"Styles tested: {df['style'].nunique()}")
        
        print("\nBias score summary:")
        print(df['bias_score'].describe())


if __name__ == "__main__":
    main()