# experiments/compute_correctness.py
"""
Compute Correctness Analysis for TruthfulQA
============================================

Evaluates TruthfulQA responses for factual correctness.

Uses LLM-as-judge to determine if model responses match the correct answers.

OPTIMIZATION: Smart deduplication - evaluates unique responses only once.

Usage:
    python experiments/compute_correctness.py \
        --run_dir results/truthfulqa/run_truthfulqa_spacing_20260228_120000 \
        --sample_size 128 \
        --judge_provider openai \
        --judge_model gpt-4o-mini
    
    # Plot only (from existing combined_means CSV)
    python experiments/compute_correctness.py \
        --run_dir results/truthfulqa/run_truthfulqa_spacing_20260228_120000 \
        --plot_only
"""

import argparse
import os
import sys
import yaml
from typing import Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_dataset_by_name
from utils.correctness_judge import judge_correctness, hash_response


# =============================================================================
# Config
# =============================================================================

def load_config() -> Dict:
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# Ground Truth Loading
# =============================================================================

def load_truthfulqa_ground_truth(sample_size: int = 128, seed: int = 42) -> List[Dict]:
    """Load TruthfulQA ground truth answers."""
    config = load_config()
    dataset_config = config["datasets"]["truthful_qa"]
    
    items = load_dataset_by_name(
        "truthful_qa",
        sample_size=sample_size,
        seed=seed,
        config_name=dataset_config.get("config_name", "generation"),
        split=dataset_config.get("split", "validation"),
    )
    
    ground_truth = []
    for item in items:
        question = item.get("question", "")
        
        # Get correct answers from meta
        meta = item.get("meta", {})
        correct_answers = meta.get("correct_answers", [])
        
        # Fallback to best_answer if no correct_answers
        if not correct_answers:
            best_answer = item.get("best_answer")
            if best_answer:
                correct_answers = [best_answer]
        
        ground_truth.append({
            "question": question,
            "correct_answers": correct_answers,
        })
    
    return ground_truth


# =============================================================================
# Correctness Analysis with Deduplication
# =============================================================================

def compute_correctness_analysis(
        csv_path: str,
        sample_size: int,
        judge_provider: str,
        judge_model: str,
        openai_key_env: str,
        gemini_key_env: str,
        max_judge_calls: int) -> str:
    """
    Evaluate TruthfulQA responses for correctness with smart deduplication.
    
    Strategy:
    1. Hash all responses
    2. Identify unique responses
    3. Evaluate unique responses only
    4. Propagate results to duplicates
    """
    
    run_dir = os.path.dirname(csv_path)
    
    print(f"\n{'='*80}")
    print(f"Computing Correctness Analysis (TruthfulQA)")
    print(f"{'='*80}")
    print(f"Input CSV: {csv_path}")
    print(f"Judge: {judge_provider}/{judge_model}")
    print(f"Sample size: {sample_size}")
    print(f"{'='*80}\n")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Strengths: {sorted(df['strength'].unique().tolist())}")
    
    # Detect response column name
    # For style experiments, we want the perturbed/styled response
    response_col = None
    for col in ['response_pert', 'response', 'generated_text', 'model_response', 'output']:
        if col in df.columns:
            response_col = col
            break
    
    if response_col is None:
        raise ValueError(f"Could not find response column. Available columns: {list(df.columns)}")
    
    print(f"Using response column: '{response_col}'\n")
    
    # Rename prompt_id to problem_id for consistency
    if 'problem_id' not in df.columns and 'prompt_id' in df.columns:
        df['problem_id'] = df['prompt_id']
    
    # Load ground truth
    print(f"\nLoading ground truth from TruthfulQA (seed=42, n={sample_size})...")
    ground_truth = load_truthfulqa_ground_truth(sample_size=sample_size, seed=42)
    gt_lookup = {i: gt for i, gt in enumerate(ground_truth)}
    print(f"✓ Loaded {len(ground_truth)} ground truth answers\n")
    
    # =================================================================
    # STEP 1: Hash responses
    # =================================================================
    print("Step 1: Hashing responses for deduplication...")
    
    df['response_hash'] = df[response_col].apply(hash_response)
    
    # Count unique responses
    unique_responses = df['response_hash'].nunique()
    
    print(f"  Responses: {len(df)} total → {unique_responses} unique ({unique_responses/len(df)*100:.1f}%)")
    
    expected_speedup = len(df) / unique_responses if unique_responses > 0 else 1
    print(f"  Expected speedup: {expected_speedup:.1f}x\n")
    
    # =================================================================
    # STEP 2: Create unique evaluation tasks
    # =================================================================
    print("Step 2: Creating unique evaluation tasks...")
    
    unique_df = df[['problem_id', response_col, 'response_hash']].drop_duplicates(
        subset='response_hash'
    ).copy()
    
    total_unique = len(unique_df)
    print(f"  Total unique evaluations needed: {total_unique}")
    print(f"  (vs {len(df)} without deduplication)\n")
    
    # =================================================================
    # STEP 3: Evaluate unique responses
    # =================================================================
    print("Step 3: Evaluating unique responses...")
    
    # Cache for results: hash → (correctness, confidence, raw, attempts)
    cache = {}
    
    judge_calls_used = 0
    
    print(f"\n  Evaluating {len(unique_df)} unique responses...")
    
    for idx, row in tqdm(unique_df.iterrows(), total=len(unique_df), desc="Judging"):
        if judge_calls_used >= max_judge_calls:
            break
        
        problem_id = int(row['problem_id'])
        response_hash = row['response_hash']
        response_text = row[response_col]
        
        # Get ground truth
        gt = gt_lookup.get(problem_id)
        if gt is None:
            cache[response_hash] = (-1, "low", "No ground truth", 0)
            continue
        
        # Judge
        try:
            result, raw, attempts = judge_correctness(
                question=gt['question'],
                correct_answers=gt['correct_answers'],
                model_response=response_text,
                judge_provider=judge_provider,
                judge_model=judge_model,
                openai_key_env=openai_key_env,
                gemini_key_env=gemini_key_env,
                max_output_tokens=100,
                max_retries=3,
            )
            
            judge_calls_used += 1
            
            if result is not None:
                correctness = int(result.get("correctness", -1))
                confidence = str(result.get("confidence", "low"))
                cache[response_hash] = (correctness, confidence, raw, attempts)
            else:
                cache[response_hash] = (-1, "low", raw, attempts)
            
        except Exception as e:
            cache[response_hash] = (-1, "low", f"ERROR: {str(e)}", 0)
    
    print(f"\n✓ Evaluated {len(cache)} unique responses")
    print(f"  Total judge calls: {judge_calls_used}\n")
    
    # =================================================================
    # STEP 4: Propagate results to all rows
    # =================================================================
    print("Step 4: Propagating results to duplicate responses...")
    
    correctness_list = []
    confidence_list = []
    judge_raw_list = []
    judge_attempts_list = []
    
    for idx, row in df.iterrows():
        resp_hash = row['response_hash']
        
        if resp_hash in cache:
            corr, conf, raw, att = cache[resp_hash]
            correctness_list.append(corr)
            confidence_list.append(conf)
            judge_raw_list.append(raw)
            judge_attempts_list.append(att)
        else:
            correctness_list.append(-1)
            confidence_list.append("skipped")
            judge_raw_list.append("Max calls reached")
            judge_attempts_list.append(0)
    
    # Add to dataframe
    df['correctness'] = correctness_list
    df['correctness_confidence'] = confidence_list
    df['correctness_judge_raw'] = judge_raw_list
    df['correctness_judge_attempts'] = judge_attempts_list
    
    # Drop hash column
    df = df.drop(columns=['response_hash'])
    
    # Save detailed results
    output_csv = csv_path.replace('.csv', '_with_correctness.csv')
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved updated CSV: {output_csv}\n")
    
    # =================================================================
    # STEP 5: Create Combined Means CSV (for plotting)
    # =================================================================
    print("Step 5: Creating combined means CSV for plotting...")
    
    create_combined_means_csv(df, run_dir)
    
    # =================================================================
    # Summary Statistics
    # =================================================================
    print(f"{'='*80}")
    print("CORRECTNESS ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    # Overall statistics
    valid = df[df['correctness'] >= 0]
    if len(valid) > 0:
        correct_count = (valid['correctness'] == 1).sum()
        incorrect_count = (valid['correctness'] == 0).sum()
        total_valid = len(valid)
        
        correctness_rate = (correct_count / total_valid * 100) if total_valid > 0 else 0
        
        print(f"\nOverall (All Strengths):")
        print(f"  Evaluated: {total_valid}/{len(df)}")
        print(f"  Correct: {correct_count}/{total_valid} ({correctness_rate:.1f}%)")
        print(f"  Incorrect: {incorrect_count}/{total_valid} ({(100-correctness_rate):.1f}%)")
    
    # Baseline (strength=0) statistics
    baseline = df[df['strength'] == 0]
    baseline_valid = baseline[baseline['correctness'] >= 0]
    
    if len(baseline_valid) > 0:
        baseline_correct = (baseline_valid['correctness'] == 1).sum()
        baseline_total = len(baseline_valid)
        baseline_rate = (baseline_correct / baseline_total * 100) if baseline_total > 0 else 0
        
        print(f"\nBaseline (Strength=0):")
        print(f"  Evaluated: {baseline_total}")
        print(f"  Correct: {baseline_correct}/{baseline_total} ({baseline_rate:.1f}%)")
    
    # Per-strength breakdown
    print(f"\nPer-strength breakdown:")
    for strength in sorted(df['strength'].unique()):
        strength_df = df[df['strength'] == strength]
        strength_valid = strength_df[strength_df['correctness'] >= 0]
        
        if len(strength_valid) > 0:
            correct = (strength_valid['correctness'] == 1).sum()
            total = len(strength_valid)
            correct_rate = (correct / total * 100) if total > 0 else 0
            
            baseline_marker = " (baseline)" if strength == 0 else ""
            print(f"  Strength {strength:3d}{baseline_marker}: {correct_rate:5.1f}% correct ({correct}/{total})")
    
    print(f"\nDeduplication savings:")
    print(f"  Unique evaluations: {judge_calls_used}")
    print(f"  Without dedup: {len(df)}")
    print(f"  Speedup: {len(df) / max(judge_calls_used, 1):.1f}x")
    print(f"{'='*80}\n")
    
    return output_csv


# =============================================================================
# Combined Means CSV Creation
# =============================================================================

def create_combined_means_csv(df: pd.DataFrame, run_dir: str):
    """
    Create/update combined_means_by_model_place_strength.csv for unified plotting.
    
    Aggregates correctness by (model, place, strength).
    """
    
    # Filter valid rows
    df_valid = df[df['correctness'] >= 0].copy()
    
    if len(df_valid) == 0:
        print("  ⚠️ No valid data to aggregate")
        return
    
    # Compute correctness rate
    def compute_correctness_rate(series):
        correct = (series == 1).sum()
        total = len(series)
        return (correct / total * 100) if total > 0 else 0
    
    # Aggregate by (model, place, strength)
    agg_dict = {
        'correctness': compute_correctness_rate,
    }
    
    grouped = df_valid.groupby(['model', 'place', 'strength'], as_index=False).agg(agg_dict)
    
    # Rename columns
    grouped = grouped.rename(columns={
        'correctness': 'truthfulqa_correctness',  # Percentage
    })
    
    # Add style column if present
    if 'style' in df.columns:
        style_map = df[['model', 'place', 'strength', 'style']].drop_duplicates()
        grouped = grouped.merge(style_map, on=['model', 'place', 'strength'], how='left')
    else:
        # Infer from directory name
        if 'spacing' in run_dir:
            grouped['style'] = 'spacing'
        elif 'punctuation' in run_dir:
            grouped['style'] = 'punctuation'
        elif 'letter_case' in run_dir:
            grouped['style'] = 'letter_case'
        elif 'politeness' in run_dir:
            grouped['style'] = 'politeness'
        else:
            grouped['style'] = 'unknown'
    
    # Add run_source
    grouped['run_source'] = 'truthfulqa'
    
    # Save to plots_metrics directory
    plots_dir = os.path.join(run_dir, "plots_metrics")
    os.makedirs(plots_dir, exist_ok=True)
    
    output_path = os.path.join(plots_dir, "combined_means_by_model_place_strength.csv")
    
    # Check if file exists and merge
    if os.path.exists(output_path):
        print(f"  ℹ️  Merging with existing combined_means CSV...")
        existing_df = pd.read_csv(output_path)
        
        # Remove old truthfulqa_correctness if it exists
        if 'truthfulqa_correctness' in existing_df.columns:
            existing_df = existing_df.drop(columns=['truthfulqa_correctness'])
        
        # Determine merge columns based on what exists in both dataframes
        merge_cols = ['model', 'place', 'strength']
        
        # Only include 'style' if it exists in BOTH dataframes
        if 'style' in existing_df.columns and 'style' in grouped.columns:
            merge_cols.append('style')
        elif 'style' in grouped.columns and 'style' not in existing_df.columns:
            # If grouped has style but existing doesn't, add it to existing
            # Infer style from directory name
            if 'spacing' in run_dir:
                existing_df['style'] = 'spacing'
            elif 'punctuation' in run_dir:
                existing_df['style'] = 'punctuation'
            elif 'letter_case' in run_dir:
                existing_df['style'] = 'letter_case'
            elif 'politeness' in run_dir:
                existing_df['style'] = 'politeness'
            else:
                existing_df['style'] = 'unknown'
            merge_cols.append('style')
        
        # Merge on determined columns
        combined = existing_df.merge(grouped[merge_cols + ['truthfulqa_correctness']], 
                                     on=merge_cols, how='outer')
        combined.to_csv(output_path, index=False)
    else:
        # Create new file
        grouped.to_csv(output_path, index=False)
    
    print(f"  ✓ Updated combined means CSV: {output_path}")
    print(f"    Rows: {len(grouped)}")
    print(f"    Models: {sorted(grouped['model'].unique().tolist())}")
    print(f"    Strengths: {sorted(grouped['strength'].unique().tolist())}")
    print(f"    Metric: truthfulqa_correctness\n")


# =============================================================================
# Plotting
# =============================================================================

def plot_correctness_from_combined_csv(combined_csv_path: str, run_dir: str):
    """
    Create correctness plot from combined CSV.
    Exact format matching plots.py.
    """
    
    if not os.path.exists(combined_csv_path):
        print(f"⚠️ Combined means CSV not found: {combined_csv_path}")
        return
    
    print(f"\n{'='*60}")
    print("Creating Correctness Plot")
    print(f"{'='*60}")
    print(f"Input: {combined_csv_path}")
    
    df = pd.read_csv(combined_csv_path)
    
    # Check if correctness column exists
    if 'truthfulqa_correctness' not in df.columns:
        print("⚠️ No truthfulqa_correctness column in CSV")
        return
    
    if len(df) == 0:
        print("⚠️ CSV is empty")
        return
    
    # Apply plots.py style
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.autolayout": False,
    })
    
    # Extract metadata
    models = sorted(df['model'].unique())
    places = sorted(df['place'].unique())
    strengths = sorted(df['strength'].unique())
    
    print(f"Models: {models}")
    print(f"Places: {places}")
    print(f"Strengths: {strengths}")
    print(f"{'='*60}\n")
    
    plots_dir = os.path.join(run_dir, "plots_metrics")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create plot
    print("Creating TruthfulQA Correctness plot...")
    
    fig_w, fig_h = 6.8, 2.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    for model in models:
        for place in places:
            subset = df[(df['model'] == model) & (df['place'] == place)].copy()
            if subset.empty:
                continue
            
            # CRITICAL: Use .reindex() for continuous lines
            subset = subset.set_index('strength').reindex(strengths)
            y = subset['truthfulqa_correctness'].values
            
            ax.plot(
                strengths,
                y,
                marker='o',
                label=f"{model}/{place}"
            )
    
    ax.set_xlabel('Strength')
    ax.set_ylabel('truthfulqa_correctness (%)')
    ax.set_title('truthfulqa_correctness vs strength (All Models and Places)', fontsize=10)
    ax.set_axisbelow(True)
    
    try:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    except:
        pass
    
    # Legend outside
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        ncol=1
    )
    
    fig.tight_layout()
    
    plot_path = os.path.join(plots_dir, 'truthfulqa_correctness.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"  ✓ Saved: {plot_path}")
    plt.close(fig)
    
    print(f"\n✓ Plotting complete\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute correctness analysis (TruthfulQA)")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to TruthfulQA experiment run directory")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples used in generation (required unless --plot_only or --add_to_combined)")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip evaluation, only create plots from existing combined_means CSV")
    parser.add_argument("--add_to_combined", action="store_true",
                        help="Skip evaluation, compute combined means from existing *_with_correctness.csv and add to combined_means CSV")
    
    parser.add_argument("--judge_provider", type=str, default="openai",
                        choices=["openai", "gemini"])
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openai_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--gemini_key_env", type=str, default="GEMINI_API_KEY")
    parser.add_argument("--max_judge_calls", type=int, default=200000)
    
    args = parser.parse_args()
    
    # =================================================================
    # ADD-TO-COMBINED MODE
    # =================================================================
    if args.add_to_combined:
        print(f"\n{'='*80}")
        print("ADD-TO-COMBINED MODE")
        print(f"{'='*80}\n")
        
        # Look for *_with_correctness.csv
        csv_with_correctness = os.path.join(args.run_dir, "full_results_all_models_with_correctness.csv")
        
        if not os.path.exists(csv_with_correctness):
            raise SystemExit(f"ERROR: {csv_with_correctness} not found")
        
        print(f"Found: {csv_with_correctness}")
        
        # Load and process
        df = pd.read_csv(csv_with_correctness)
        print(f"Loaded {len(df)} rows\n")
        
        # Create combined means
        create_combined_means_csv(df, args.run_dir)
        
        print(f"\n{'='*80}")
        print("COMPLETE (Add-to-Combined)")
        print(f"{'='*80}\n")
        return
    
    # =================================================================
    # PLOT-ONLY MODE
    # =================================================================
    if args.plot_only:
        print(f"\n{'='*80}")
        print("PLOT-ONLY MODE")
        print(f"{'='*80}\n")
        
        # Look for combined CSV
        combined_csv_path = os.path.join(args.run_dir, "plots_metrics", "combined_means_by_model_place_strength.csv")
        
        plot_correctness_from_combined_csv(combined_csv_path, args.run_dir)
        
        print(f"\n{'='*80}")
        print("COMPLETE (Plot-Only)")
        print(f"{'='*80}\n")
        return
    
    # =================================================================
    # FULL MODE (Evaluate + Plot)
    # =================================================================
    
    if args.sample_size is None:
        raise SystemExit("ERROR: --sample_size is required (unless using --plot_only or --add_to_combined)")
    
    # Verify API key
    if args.judge_provider == "openai":
        if not os.environ.get(args.openai_key_env):
            raise SystemExit(f"ERROR: {args.openai_key_env} not set")
    else:
        if not os.environ.get(args.gemini_key_env):
            raise SystemExit(f"ERROR: {args.gemini_key_env} not set")
    
    # Find CSV file in run directory
    csv_path = os.path.join(args.run_dir, "full_results_all_models.csv")
    
    if not os.path.exists(csv_path):
        raise SystemExit(f"ERROR: full_results_all_models.csv not found in {args.run_dir}")
    
    print(f"Found results CSV: {csv_path}")
    
    # Compute correctness analysis
    updated_csv = compute_correctness_analysis(
        csv_path=csv_path,
        sample_size=args.sample_size,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        openai_key_env=args.openai_key_env,
        gemini_key_env=args.gemini_key_env,
        max_judge_calls=args.max_judge_calls,
    )
    
    # Plot from combined CSV
    combined_csv_path = os.path.join(args.run_dir, "plots_metrics", "combined_means_by_model_place_strength.csv")
    plot_correctness_from_combined_csv(combined_csv_path, args.run_dir)
    
    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()