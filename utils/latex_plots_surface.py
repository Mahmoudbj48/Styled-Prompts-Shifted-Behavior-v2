"""
utils/latex_plots_surface.py — LaTeX figure string generation for surface noise styles.

Generates LaTeX figure environments for spacing, letter_case, and punctuation
experiments across TruthfulQA and Natural Questions datasets.

Usage:
    # Generate LaTeX for all surface styles
    python utils/latex_plots_surface.py
    
    # Generate LaTeX for specific style only
    python utils/latex_plots_surface.py --style spacing
    
    # Generate LaTeX for specific dataset only
    python utils/latex_plots_surface.py --dataset truthfulqa
    
    # Output to file
    python utils/latex_plots_surface.py > surface_appendix.tex

Outputs:
    - Complete LaTeX appendix section for surface noise styles
    - Follows same structure as politeness appendix
    - Ready to paste into Overleaf
"""

import argparse

# Metric labels (same as politeness)
METRIC_LABELS = {
    "activation_similarity": "Activation Similarity",
    "bleu": "BLEU Score",
    "bertscore_response": "BERTScore (Response)",
    "delta_log_prob": "$\\Delta$ Log-Prob",
    "entropy_shift": "Entropy Shift",
    "jsd_drift": "JSD Drift",
    "silhouette": "Silhouette Score",
    "asr": "ASR",
}

# Surface noise styles configuration
SURFACE_STYLES = {
    "spacing": {
        "title": "Spacing",
        "latex_label": "spacing",
    },
    "letter_case": {
        "title": "Letter Case",
        "latex_label": "letter_case",
    },
    "punctuation": {
        "title": "Punctuation",
        "latex_label": "punctuation",
    },
}

# Behavioral axes (same structure as politeness)
AXES_META = {
    "activation_geometry": {
        "title": "Activation Geometry",
        "metrics": ["activation_similarity"],
        "desc": "representational drift via cosine similarity between last-nonpad-token hidden-state vectors",
    },
    "generation_quality": {
        "title": "Generation Quality",
        "metrics": ["bleu", "bertscore_response"],
        "desc": "output stability distinguishing surface-form rewriting from meaning-level drift",
    },
    "confidence_uncertainty": {
        "title": "Confidence and Uncertainty",
        "metrics": ["delta_log_prob", "entropy_shift"],
        "desc": "prompt sensitivity measuring changes in predictive likelihood and predictive sharpness",
    },
    "safety_refusal": {
        "title": "Safety and Refusal",
        "metrics": ["asr", "silhouette"],
        "desc": "safety robustness and representational separability between benign and harmful inputs",
    },
}


def generate_surface_style_results(style_filter=None, dataset_filter=None) -> str:
    """
    Generate LaTeX results section for surface noise styles.
    
    Args:
        style_filter: If specified, only generate for this style (spacing/letter_case/punctuation)
        dataset_filter: If specified, only generate for this dataset (truthfulqa/nq)
    
    Returns:
        str: LaTeX string containing all figure environments
    """
    # Model subsets
    all_models = ["G-2B", "G-7B", "L3.1-8B", "L3.2-3B", "Q2.5-1.5B", "Q2.5-7B"]
    big_models = ["G-7B", "L3.1-8B", "Q2.5-7B"]
    places = ["global", "prefix", "suffix"]
    
    latex_output = []
    
    # Filter styles if requested
    styles_to_process = SURFACE_STYLES.keys()
    if style_filter:
        if style_filter not in SURFACE_STYLES:
            raise ValueError(f"Unknown style: {style_filter}. Choose from {list(SURFACE_STYLES.keys())}")
        styles_to_process = [style_filter]
    
    # Process each style
    for style_key in styles_to_process:
        style_info = SURFACE_STYLES[style_key]
        style_title = style_info["title"]
        style_label = style_info["latex_label"]
        
        latex_output.append(f"\\section{{{style_title} Results}}")
        latex_output.append(f"\\label{{sec:surface_{style_label}}}")
        latex_output.append("")
        
        # Process each behavioral axis
        for axis_key, axis_info in AXES_META.items():
            latex_output.append(f"\\subsection{{{axis_info['title']}}}")
            latex_output.append(f"\\label{{sec:surface_{style_label}_{axis_key}}}")
            latex_output.append("")
            
            # TruthfulQA subsection (or HarmBench for safety)
            if axis_key == "safety_refusal":
                # Safety experiments
                if dataset_filter is None or dataset_filter == "harmbench":
                    latex_output.append(f"\\subsubsection{{HarmBench + Alpaca}}")
                    base_dir = f"imgs/safety_{style_label}"
                    
                    for metric in axis_info['metrics']:
                        latex_output.append(generate_metric_block(
                            metric=metric,
                            models=all_models,
                            places=places,
                            info=axis_info,
                            style=style_title,
                            ds="HarmBench" if metric == "asr" else "HarmBench + Alpaca",
                            base_dir=base_dir,
                        ))
            else:
                # TruthfulQA
                if dataset_filter is None or dataset_filter == "truthfulqa":
                    latex_output.append(f"\\subsubsection{{TruthfulQA}}")
                    base_dir = f"imgs/{style_label}_truthfulqa"
                    
                    for metric in axis_info['metrics']:
                        latex_output.append(generate_metric_block(
                            metric=metric,
                            models=all_models,
                            places=places,
                            info=axis_info,
                            style=style_title,
                            ds="TruthfulQA",
                            base_dir=base_dir,
                        ))
                
                # Natural Questions
                if dataset_filter is None or dataset_filter == "nq":
                    latex_output.append(f"\\subsubsection{{Natural Questions}}")
                    base_dir = f"imgs/{style_label}_nq"
                    
                    for metric in axis_info['metrics']:
                        latex_output.append(generate_metric_block(
                            metric=metric,
                            models=big_models,
                            places=places,
                            info=axis_info,
                            style=style_title,
                            ds="Natural Questions",
                            base_dir=base_dir,
                        ))
            
            latex_output.append("")
        
        latex_output.append("\\newpage")
        latex_output.append("")
    
    return "\n".join(latex_output)


def generate_metric_block(metric, models, places, info, style, ds, base_dir):
    """
    Generate all figure blocks for one metric (line per model, line per place, radar).
    
    Args:
        metric: Metric key from METRIC_LABELS
        models: List of model aliases
        places: List of placement strings
        info: Axis metadata dict from AXES_META
        style: Style family label for captions
        ds: Dataset name for captions
        base_dir: Base image directory path
    
    Returns:
        str: LaTeX string with all figure environments for this metric
    """
    label = METRIC_LABELS.get(metric, metric)
    desc = info.get('desc', '')
    ds_slug = ds.lower().replace(" ", "_").replace("+", "")
    
    blocks = [
        generate_grid(metric, "line_per_model", models, label, desc, style, ds, base_dir, ds_slug),
        generate_grid(metric, "line_per_place", places, label, desc, style, ds, base_dir, ds_slug),
        generate_radar_pair(metric, label, desc, style, ds, base_dir, ds_slug),
    ]
    return "\n".join(blocks)


def generate_grid(metric, p_type, items, label, desc, style, ds, base_dir, ds_slug):
    """
    Generate a LaTeX figure with a grid of subfigures (one per model or placement).
    
    Args:
        metric: Metric key
        p_type: Plot type string ("line_per_model" or "line_per_place")
        items: List of items (model aliases or placement strings)
        label: Human-readable metric label
        desc: Short metric description
        style: Style family label
        ds: Dataset name
        base_dir: Base image directory
        ds_slug: URL-safe dataset slug for LaTeX labels
    
    Returns:
        str: LaTeX figure environment string
    """
    latex = ["\\begin{figure}[H]", "    \\centering"]
    
    for i, item in enumerate(items):
        path = f"{base_dir}/{p_type}/{metric}_{p_type}__{item}.png"
        latex.append(f"    \\begin{{subfigure}}[b]{{0.31\\textwidth}}")
        latex.append(f"        \\centering")
        latex.append(f"        \\includegraphics[width=\\textwidth]{{{path}}}")
        latex.append(f"        \\caption{{{item}}}")
        latex.append(f"    \\end{{subfigure}}")
        
        if (i + 1) % 3 == 0:
            latex.append("    \\\\ \\vspace{0.2cm}")
        else:
            latex.append("    \\hfill")
    
    # Remove trailing hfill if last row incomplete
    if len(items) % 3 != 0:
        latex = latex[:-1]
    
    cap = f"\\textbf{{{label} vs. {style} ({ds}).}} {label} grouped by {p_type.split('_')[-1]}."
    latex.extend([
        "    \\\\ \\vspace{0.2cm}",
        f"    \\caption{{{cap}}}",
        f"    \\label{{fig:{ds_slug}_{metric}_{p_type}}}",
        "\\end{figure}"
    ])
    
    return "\n".join(latex)


def generate_radar_pair(metric, label, desc, style, ds, base_dir, ds_slug):
    """
    Generate a LaTeX figure with two side-by-side radar plots.
    
    Args:
        metric: Metric key
        label: Human-readable metric label
        desc: Short metric description
        style: Style family label
        ds: Dataset name
        base_dir: Base image directory
        ds_slug: URL-safe dataset slug for LaTeX labels
    
    Returns:
        str: LaTeX figure environment string
    """
    path_m = f"{base_dir}/radar/{metric}_radar_axes_models.png"
    path_p = f"{base_dir}/radar/{metric}_radar_axes_places.png"
    
    return f"""\\begin{{figure}}[H]
    \\centering
    \\begin{{subfigure}}[b]{{0.48\\textwidth}}
        \\centering
        \\includegraphics[width=\\textwidth]{{{path_m}}}
        \\caption{{Axes: Models}}
    \\end{{subfigure}}
    \\hfill
    \\begin{{subfigure}}[b]{{0.48\\textwidth}}
        \\centering
        \\includegraphics[width=\\textwidth]{{{path_p}}}
        \\caption{{Axes: Positions}}
    \\end{{subfigure}}
    \\caption{{\\textbf{{{label} Radar Analysis ({ds}).}} {label} across axes for {style}.}}
    \\label{{fig:{ds_slug}_{metric}_radar}}
\\end{{figure}}"""


def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX for surface noise style experiments'
    )
    parser.add_argument(
        '--style',
        choices=['spacing', 'letter_case', 'punctuation'],
        help='Generate LaTeX for specific style only'
    )
    parser.add_argument(
        '--dataset',
        choices=['truthfulqa', 'nq', 'harmbench'],
        help='Generate LaTeX for specific dataset only'
    )
    
    args = parser.parse_args()
    
    latex_code = generate_surface_style_results(
        style_filter=args.style,
        dataset_filter=args.dataset
    )
    
    print(latex_code)


if __name__ == "__main__":
    main()