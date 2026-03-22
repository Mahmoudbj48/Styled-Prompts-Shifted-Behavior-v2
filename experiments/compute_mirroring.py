# experiments/compute_mirroring.py
"""
Compute Mirroring Rates for Surface Noise Experiments
======================================================

Reads CSV files from surface noise experiments (spacing, punctuation, letter_case)
and computes mirroring rates using deterministic detection.

Outputs:
    - Updated CSV with 'mirroring_verdict' and 'mirroring_debug' columns
    - Mirroring rate plot saved in plots_metrics/

Run:
    python experiments/compute_mirroring.py \
        --input results/spacing/run_multi_truthful_qa_20260223_221345/full_results_all_models.csv \
        --style spacing
"""

import argparse
import os
import sys
import json
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.surface_mirroring_detector import detect_mirroring


def compute_mirroring_for_csv(
        csv_path: str,
        style: str) -> str:
    """
    Add mirroring detection to existing CSV.
    
    Returns:
        Path to updated CSV
    """
    
    # Determine run directory from CSV path
    run_dir = os.path.dirname(csv_path)
    
    print(f"\n{'='*80}")
    print(f"Computing Mirroring Rates: {style.upper()}")
    print(f"{'='*80}")
    print(f"Input CSV: {csv_path}")
    print(f"Run directory: {run_dir}")
    print(f"{'='*80}\n")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} rows")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Strengths: {sorted(df['strength'].unique().tolist())}")
    print(f"Places: {df['place'].unique().tolist()}\n")
    
    # Add mirroring columns
    mirroring_verdicts = []
    mirroring_debug_info = []
    
    print("Detecting mirroring...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        
        original_response = str(row.get('response_orig', ''))
        styled_response = str(row.get('response_pert', ''))
        strength = int(row.get('strength', 0))
        place = str(row.get('place', 'global'))
        
        # Detect mirroring
        try:
            is_mirroring, debug_info = detect_mirroring(
                original_response=original_response,
                styled_response=styled_response,
                style=style,
                strength=strength,
                place=place
            )
            
            mirroring_verdicts.append("YES" if is_mirroring else "NO")
            mirroring_debug_info.append(json.dumps(debug_info))
            
        except Exception as e:
            print(f"\n⚠️  Error at row {idx}: {e}")
            mirroring_verdicts.append("ERROR")
            mirroring_debug_info.append(json.dumps({"error": str(e)}))
    
    # Add to dataframe
    df['mirroring_verdict'] = mirroring_verdicts
    df['mirroring_debug'] = mirroring_debug_info
    
    # Save updated CSV with descriptive name
    output_csv = os.path.join(run_dir, f"results_with_mirroring.csv")
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved updated CSV: {output_csv}")
    
    # Compute summary statistics
    print(f"\n{'='*80}")
    print("MIRRORING SUMMARY")
    print(f"{'='*80}")
    
    total = len(df)
    mirroring_count = (df['mirroring_verdict'] == 'YES').sum()
    mirroring_rate = (mirroring_count / total) * 100 if total > 0 else 0
    
    print(f"Total responses: {total}")
    print(f"Mirroring detected: {mirroring_count} ({mirroring_rate:.1f}%)")
    
    # Per-strength breakdown
    print(f"\nPer-strength breakdown:")
    for strength in sorted(df['strength'].unique()):
        strength_df = df[df['strength'] == strength]
        strength_mirroring = (strength_df['mirroring_verdict'] == 'YES').sum()
        strength_total = len(strength_df)
        strength_rate = (strength_mirroring / strength_total * 100) if strength_total > 0 else 0
        print(f"  Strength {strength:3d}: {strength_mirroring:4d}/{strength_total:4d} ({strength_rate:5.1f}%)")
    
    # Per-model breakdown
    print(f"\nPer-model breakdown:")
    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]
        model_mirroring = (model_df['mirroring_verdict'] == 'YES').sum()
        model_total = len(model_df)
        model_rate = (model_mirroring / model_total * 100) if model_total > 0 else 0
        print(f"  {model:15s}: {model_mirroring:4d}/{model_total:4d} ({model_rate:5.1f}%)")
    
    print(f"{'='*80}\n")
    
    return output_csv


def plot_mirroring_rates(
        csv_path: str,
        style: str):
    """
    Create mirroring rate plot (similar to other metric plots).
    Saves in plots_metrics/ subdirectory.
    """
    
    df = pd.read_csv(csv_path)
    
    # Determine run directory and plots directory
    run_dir = os.path.dirname(csv_path)
    plots_dir = os.path.join(run_dir, "plots_metrics")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Compute mirroring rate per (model, strength, place)
    grouped = df.groupby(['model', 'strength', 'place']).agg({
        'mirroring_verdict': lambda x: (x == 'YES').sum() / len(x) * 100
    }).reset_index()
    grouped.rename(columns={'mirroring_verdict': 'mirroring_rate'}, inplace=True)
    
    # Get unique values
    models = sorted(df['model'].unique())
    places = sorted(df['place'].unique())
    strengths = sorted(df['strength'].unique())
    
    # Create plot
    n_places = len(places)
    fig, axes = plt.subplots(1, n_places, figsize=(6 * n_places, 5), squeeze=False)
    axes = axes[0]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for place_idx, place in enumerate(places):
        ax = axes[place_idx]
        
        for model_idx, model in enumerate(models):
            # Get data for this model/place
            data = grouped[(grouped['model'] == model) & (grouped['place'] == place)]
            
            if len(data) > 0:
                data_sorted = data.sort_values('strength')
                ax.plot(
                    data_sorted['strength'],
                    data_sorted['mirroring_rate'],
                    marker='o',
                    label=model,
                    color=colors[model_idx],
                    linewidth=2,
                    markersize=8
                )
        
        ax.set_xlabel('Strength', fontsize=12)
        ax.set_ylabel('Mirroring Rate (%)', fontsize=12)
        ax.set_title(f'Place: {place}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)
        
        if place_idx == 0:
            ax.legend(loc='best', fontsize=10)
    
    plt.suptitle(f'Mirroring Rate: {style.title()}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save plot with descriptive name
    plot_path = os.path.join(plots_dir, f'mirroring_rate.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {plot_path}")
    
    plt.close()


def main():
    """Parse CLI arguments and run the mirroring detection and plotting pipeline."""
    parser = argparse.ArgumentParser(description="Compute mirroring rates for surface noise experiments")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to CSV file (e.g., full_results_all_models.csv)")
    parser.add_argument("--style", type=str, required=True,
                        choices=["spacing", "punctuation", "letter_case"],
                        help="Style type")
    
    args = parser.parse_args()
    
    # Compute mirroring
    updated_csv = compute_mirroring_for_csv(
        csv_path=args.input,
        style=args.style
    )
    
    # Create plot
    plot_mirroring_rates(
        csv_path=updated_csv,
        style=args.style
    )
    
    run_dir = os.path.dirname(updated_csv)
    
    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")
    print(f"Updated CSV: {updated_csv}")
    print(f"Plot: {run_dir}/plots_metrics/mirroring_rate.png")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()