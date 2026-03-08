# experiments/compute_cot_analysis.py
"""
Compute CoT Step Counting and Correctness Analysis
===================================================

Evaluates CoT responses for:
1. Number of reasoning steps
2. Answer correctness

UPDATED: Only evaluates response_styled (since strength=0 is identical to original).
Uses strength=0 as baseline for comparison.

Creates combined_means_by_model_place_strength.csv for unified plotting.

OPTIMIZATION: Smart deduplication - evaluates unique responses only once.

Usage:
    # Full analysis (evaluate + plot)
    python experiments/compute_cot_analysis.py \
        --run_dir results/cot_responses/run_gsm8k_spacing_20260228_120000 \
        --sample_size 28 \
        --judge_provider openai \
        --judge_model gpt-4o-mini
    
    # Plot only (from existing combined_means CSV)
    python experiments/compute_cot_analysis.py \
        --run_dir results/cot_responses/run_gsm8k_spacing_20260228_120000 \
        --plot_only
"""

import argparse
import os
import sys
import yaml
import json
from typing import Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_dataset_by_name
from utils.cot_judge import judge_cot_response, hash_response


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

def load_gsm8k_ground_truth(sample_size: int = 28, seed: int = 42) -> List[Dict]:
    """Load GSM8K ground truth answers."""
    config = load_config()
    dataset_config = config["datasets"]["gsm8k"]
    
    items = load_dataset_by_name(
        "gsm8k",
        sample_size=sample_size,
        seed=seed,
        config_name=dataset_config.get("config_name", "main"),
        split=dataset_config.get("split", "test"),
    )
    
    ground_truth = []
    for item in items:
        question = item.get("question", "")
        answer = item.get("answer", "")
        
        # GSM8K answers are in format "#### 42"
        # Extract the numeric answer
        if "####" in answer:
            answer_value = answer.split("####")[-1].strip()
        else:
            answer_value = answer.strip()
        
        ground_truth.append({
            "question": question,
            "answer": answer_value,
        })
    
    return ground_truth


# =============================================================================
# CoT Analysis with Deduplication (STYLED ONLY)
# =============================================================================

def compute_cot_analysis(
        csv_path: str,
        sample_size: int,
        judge_provider: str,
        judge_model: str,
        openai_key_env: str,
        gemini_key_env: str,
        max_judge_calls: int) -> str:
    """
    Evaluate CoT responses with smart deduplication.
    
    UPDATED: Only evaluates response_styled.
    Strength=0 serves as baseline (identical to original).
    
    Strategy:
    1. Hash all styled responses
    2. Identify unique responses
    3. Evaluate unique responses only
    4. Propagate results to duplicates
    """
    
    run_dir = os.path.dirname(csv_path)
    
    print(f"\n{'='*80}")
    print(f"Computing CoT Analysis (Styled Responses Only)")
    print(f"{'='*80}")
    print(f"Input CSV: {csv_path}")
    print(f"Judge: {judge_provider}/{judge_model}")
    print(f"Sample size: {sample_size}")
    print(f"Baseline: strength=0 (identical to original)")
    print(f"{'='*80}\n")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"Total rows: {len(df)}")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Strengths: {sorted(df['strength'].unique().tolist())}")
    
    # Load ground truth
    print(f"\nLoading ground truth from GSM8K (seed=42, n={sample_size})...")
    ground_truth = load_gsm8k_ground_truth(sample_size=sample_size, seed=42)
    gt_lookup = {i: gt for i, gt in enumerate(ground_truth)}
    print(f"✓ Loaded {len(ground_truth)} ground truth answers\n")
    
    # =================================================================
    # STEP 1: Hash styled responses only
    # =================================================================
    print("Step 1: Hashing styled responses for deduplication...")
    
    df['response_hash_styled'] = df['response_styled'].apply(hash_response)
    
    # Count unique responses
    unique_styled = df['response_hash_styled'].nunique()
    
    print(f"  Styled responses: {len(df)} total → {unique_styled} unique ({unique_styled/len(df)*100:.1f}%)")
    
    expected_speedup = len(df) / unique_styled if unique_styled > 0 else 1
    print(f"  Expected speedup: {expected_speedup:.1f}x\n")
    
    # =================================================================
    # STEP 2: Create unique evaluation tasks
    # =================================================================
    print("Step 2: Creating unique evaluation tasks...")
    
    # Styled responses only
    unique_styled_df = df[['problem_id', 'response_styled', 'response_hash_styled']].drop_duplicates(
        subset='response_hash_styled'
    ).copy()
    
    total_unique = len(unique_styled_df)
    print(f"  Total unique evaluations needed: {total_unique}")
    print(f"  (vs {len(df)} without deduplication)\n")
    
    # =================================================================
    # STEP 3: Evaluate unique responses
    # =================================================================
    print("Step 3: Evaluating unique styled responses...")
    
    # Cache for results: hash → (steps, correct_answer, raw, attempts)
    cache_styled = {}
    
    judge_calls_used = 0
    
    print(f"\n  Evaluating {len(unique_styled_df)} unique styled responses...")
    
    for idx, row in tqdm(unique_styled_df.iterrows(), total=len(unique_styled_df), desc="Styled"):
        if judge_calls_used >= max_judge_calls:
            break
        
        problem_id = int(row['problem_id'])
        response_hash = row['response_hash_styled']
        response_text = row['response_styled']
        
        # Get ground truth
        gt = gt_lookup.get(problem_id)
        if gt is None:
            cache_styled[response_hash] = (-1, "error", "No ground truth", 0)
            continue
        
        # Judge
        try:
            result, raw, attempts = judge_cot_response(
                question=gt['question'],
                ground_truth=gt['answer'],
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
                steps = int(result.get("steps", -1))
                correct = str(result.get("correct_answer", "error"))
                cache_styled[response_hash] = (steps, correct, raw, attempts)
            else:
                cache_styled[response_hash] = (-1, "parse_error", raw, attempts)
            
        except Exception as e:
            cache_styled[response_hash] = (-1, "error", f"ERROR: {str(e)}", 0)
    
    print(f"\n✓ Evaluated {len(cache_styled)} unique styled responses")
    print(f"  Total judge calls: {judge_calls_used}\n")
    
    # =================================================================
    # STEP 4: Propagate results to all rows
    # =================================================================
    print("Step 4: Propagating results to duplicate responses...")
    
    steps_styled = []
    correct_styled = []
    judge_raw_styled = []
    judge_attempts_styled = []
    
    for idx, row in df.iterrows():
        hash_styled = row['response_hash_styled']
        
        # Styled
        if hash_styled in cache_styled:
            s, c, raw, att = cache_styled[hash_styled]
            steps_styled.append(s)
            correct_styled.append(c)
            judge_raw_styled.append(raw)
            judge_attempts_styled.append(att)
        else:
            steps_styled.append(-1)
            correct_styled.append("skipped")
            judge_raw_styled.append("Max calls reached")
            judge_attempts_styled.append(0)
    
    # Add to dataframe
    df['cot_steps'] = steps_styled
    df['cot_correct'] = correct_styled
    df['cot_judge_raw'] = judge_raw_styled
    df['cot_judge_attempts'] = judge_attempts_styled
    
    # Drop hash column
    df = df.drop(columns=['response_hash_styled'])
    
    # Save detailed results
    output_csv = os.path.join(run_dir, "results_with_cot_analysis.csv")
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
    print("COT ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    # Overall statistics
    valid = df[df['cot_steps'] >= 0]
    if len(valid) > 0:
        mean_steps = valid['cot_steps'].mean()
        correct_1 = (valid['cot_correct'] == '1').sum()
        correct_0 = (valid['cot_correct'] == '0').sum()
        correct_na = (valid['cot_correct'] == 'na').sum()
        total_valid = len(valid)
        
        print(f"\nOverall (All Strengths):")
        print(f"  Evaluated: {total_valid}/{len(df)}")
        print(f"  Average steps: {mean_steps:.2f}")
        print(f"  Correct: {correct_1}/{total_valid} ({correct_1/total_valid*100:.1f}%)")
        print(f"  Incorrect: {correct_0}/{total_valid} ({correct_0/total_valid*100:.1f}%)")
        print(f"  Incomplete/NA: {correct_na}/{total_valid} ({correct_na/total_valid*100:.1f}%)")
    
    # Baseline (strength=0) statistics
    baseline = df[df['strength'] == 0]
    baseline_valid = baseline[baseline['cot_steps'] >= 0]
    
    if len(baseline_valid) > 0:
        mean_steps_baseline = baseline_valid['cot_steps'].mean()
        correct_1_baseline = (baseline_valid['cot_correct'] == '1').sum()
        total_baseline = len(baseline_valid)
        
        print(f"\nBaseline (Strength=0, identical to original):")
        print(f"  Evaluated: {total_baseline}")
        print(f"  Average steps: {mean_steps_baseline:.2f}")
        print(f"  Correct: {correct_1_baseline}/{total_baseline} ({correct_1_baseline/total_baseline*100:.1f}%)")
    
    # Per-strength breakdown
    print(f"\nPer-strength breakdown:")
    for strength in sorted(df['strength'].unique()):
        strength_df = df[df['strength'] == strength]
        strength_valid = strength_df[strength_df['cot_steps'] >= 0]
        
        if len(strength_valid) > 0:
            avg_steps = strength_valid['cot_steps'].mean()
            correct = (strength_valid['cot_correct'] == '1').sum()
            total = len(strength_valid)
            correct_rate = (correct / total * 100) if total > 0 else 0
            
            baseline_marker = " (baseline)" if strength == 0 else ""
            print(f"  Strength {strength:3d}{baseline_marker}: {avg_steps:.2f} steps, {correct_rate:5.1f}% correct")
    
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
    Create combined_means_by_model_place_strength.csv for unified plotting.
    
    Aggregates metrics by (model, place, strength) for compatibility with plots.py
    """
    
    # Filter valid rows
    df_valid = df[df['cot_steps'] >= 0].copy()
    
    if len(df_valid) == 0:
        print("  ⚠️ No valid data to aggregate")
        return
    
    # Compute correctness rate (excluding 'na' responses)
    def compute_correctness_rate(series):
        correct = (series == '1').sum()
        total_answered = ((series == '1') | (series == '0')).sum()
        return (correct / total_answered * 100) if total_answered > 0 else np.nan
    
    # Aggregate by (model, place, strength)
    agg_dict = {
        'cot_steps': 'mean',
        'cot_correct': compute_correctness_rate,
    }
    
    grouped = df_valid.groupby(['model', 'place', 'strength'], as_index=False).agg(agg_dict)
    
    # Rename columns to match expected format
    grouped = grouped.rename(columns={
        'cot_steps': 'cot_steps',           # Average CoT steps
        'cot_correct': 'cot_correctness',   # Correctness rate (%)
    })
    
    # Add style column if present in original data
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
    grouped['run_source'] = 'cot_reasoning'
    
    # Reorder columns to match standard format
    cols_order = ['model', 'place', 'strength', 'style', 'cot_steps', 'cot_correctness', 'run_source']
    # Keep only columns that exist
    cols_order = [c for c in cols_order if c in grouped.columns]
    grouped = grouped[cols_order]
    
    # Save to plots_metrics directory
    plots_dir = os.path.join(run_dir, "plots_metrics")
    os.makedirs(plots_dir, exist_ok=True)
    
    output_path = os.path.join(plots_dir, "combined_means_by_model_place_strength.csv")
    grouped.to_csv(output_path, index=False)
    
    print(f"  ✓ Created combined means CSV: {output_path}")
    print(f"    Rows: {len(grouped)}")
    print(f"    Models: {sorted(grouped['model'].unique().tolist())}")
    print(f"    Strengths: {sorted(grouped['strength'].unique().tolist())}")
    print(f"    Metrics: cot_steps, cot_correctness\n")


# =============================================================================
# Plotting (Exact plots.py format)
# =============================================================================

def plot_cot_from_combined_csv(combined_csv_path: str, run_dir: str):
    """
    Create CoT plots from already-combined CSV.
    Exact format matching plots.py (politeness/surface variants).
    """
    
    if not os.path.exists(combined_csv_path):
        print(f"⚠️ Combined means CSV not found: {combined_csv_path}")
        return
    
    print(f"\n{'='*60}")
    print("Creating CoT Plots")
    print(f"{'='*60}")
    print(f"Input: {combined_csv_path}")
    
    df = pd.read_csv(combined_csv_path)
    
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
    
    # =================================================================
    # Plot 1: CoT Steps
    # =================================================================
    
    if 'cot_steps' in df.columns:
        print("Creating CoT Steps plot...")
        
        fig_w, fig_h = 6.8, 2.8
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        
        for model in models:
            for place in places:
                subset = df[(df['model'] == model) & (df['place'] == place)].copy()
                if subset.empty:
                    continue
                
                # CRITICAL: Use .reindex() for continuous lines
                subset = subset.set_index('strength').reindex(strengths)
                y = subset['cot_steps'].values
                
                ax.plot(
                    strengths,
                    y,
                    marker='o',
                    label=f"{model}/{place}"
                )
        
        ax.set_xlabel('Strength')
        ax.set_ylabel('cot_steps')
        ax.set_title('cot_steps vs strength (GSM8K) (All Models and Places)', fontsize=10)  # ← TITLE
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
        
        plot_path = os.path.join(plots_dir, 'cot_steps.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        
        print(f"  ✓ Saved: {plot_path}")
        plt.close(fig)
    
    # =================================================================
    # Plot 2: CoT Correctness
    # =================================================================
    
    if 'cot_correctness' in df.columns:
        print("Creating CoT Correctness plot...")
        
        fig_w, fig_h = 6.8, 2.8
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        
        for model in models:
            for place in places:
                subset = df[(df['model'] == model) & (df['place'] == place)].copy()
                if subset.empty:
                    continue
                
                # CRITICAL: Use .reindex() for continuous lines
                subset = subset.set_index('strength').reindex(strengths)
                y = subset['cot_correctness'].values
                
                ax.plot(
                    strengths,
                    y,
                    marker='o',
                    label=f"{model}/{place}"
                )
        
        ax.set_xlabel('Strength')
        ax.set_ylabel('cot_correctness')
        ax.set_title('cot_correctness vs strength (GSM8K) (All Models and Places)', fontsize=10)  # ← TITLE
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
        
        plot_path = os.path.join(plots_dir, 'cot_correctness.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        
        print(f"  ✓ Saved: {plot_path}")
        plt.close(fig)
    
    print(f"\n✓ Plotting complete\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute CoT analysis (styled responses only)")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to CoT experiment run directory")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples used in generation (required unless --plot_only)")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip evaluation, only create plots from existing combined_means CSV")
    
    parser.add_argument("--judge_provider", type=str, default="openai",
                        choices=["openai", "gemini"])
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openai_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--gemini_key_env", type=str, default="GEMINI_API_KEY")
    parser.add_argument("--max_judge_calls", type=int, default=200000)
    
    args = parser.parse_args()
    
    # =================================================================
    # PLOT-ONLY MODE
    # =================================================================
    if args.plot_only:
        print(f"\n{'='*80}")
        print("PLOT-ONLY MODE")
        print(f"{'='*80}\n")
        
        # Look for combined CSV
        combined_csv_path = os.path.join(args.run_dir, "plots_metrics", "combined_means_by_model_place_strength.csv")
        
        plot_cot_from_combined_csv(combined_csv_path, args.run_dir)
        
        print(f"\n{'='*80}")
        print("COMPLETE (Plot-Only)")
        print(f"{'='*80}\n")
        return
    
    # =================================================================
    # FULL MODE (Evaluate + Plot)
    # =================================================================
    
    if args.sample_size is None:
        raise SystemExit("ERROR: --sample_size is required (unless using --plot_only)")
    
    # Verify API key
    if args.judge_provider == "openai":
        if not os.environ.get(args.openai_key_env):
            raise SystemExit(f"ERROR: {args.openai_key_env} not set")
    else:
        if not os.environ.get(args.gemini_key_env):
            raise SystemExit(f"ERROR: {args.gemini_key_env} not set")
    
    # Find CSV file in run directory
    csv_candidates = [
        os.path.join(args.run_dir, "all_models_responses.csv"),
    ]
    
    # Also check for model-specific CSVs
    try:
        csv_files = [f for f in os.listdir(args.run_dir) if f.endswith('_responses.csv')]
        csv_candidates.extend([os.path.join(args.run_dir, f) for f in csv_files])
    except:
        pass
    
    csv_path = None
    for candidate in csv_candidates:
        if os.path.exists(candidate):
            csv_path = candidate
            break
    
    if csv_path is None:
        raise SystemExit(f"ERROR: No responses CSV found in {args.run_dir}")
    
    print(f"Found responses CSV: {csv_path}")
    
    # Compute CoT analysis
    updated_csv = compute_cot_analysis(
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
    plot_cot_from_combined_csv(combined_csv_path, args.run_dir)
    
    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()