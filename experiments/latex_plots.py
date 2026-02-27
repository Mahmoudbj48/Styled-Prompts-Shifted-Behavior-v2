import os

# Definitions strictly from your paper text
AXES_META = {
    "activation_geometry": {
        "title": "Activation Geometry",
        "metrics": ["activation_similarity"],
        "desc": "representational drift via cosine similarity between last-nonpad-token hidden-state vectors",
        "ds": "TruthfulQA"
    },
    "generation_quality": {
        "title": "Generation Quality",
        "metrics": ["bleu", "bertscore_response"],
        "desc": "output stability distinguishing surface-form rewriting from meaning-level drift",
        "ds": "TruthfulQA"
    },
    "confidence_uncertainty": {
        "title": "Confidence and Uncertainty",
        "metrics": ["delta_log_prob", "entropy_shift"],
        "desc": "prompt sensitivity measuring changes in predictive likelihood and predictive sharpness",
        "ds": "TruthfulQA"
    },
    "style_mirroring": {
        "title": "Style Mirroring",
        "metrics": ["mirroring_rate"],
        "desc": "the proportion of instances where stylistic variation propagates into response-level behavior",
        "ds": "TruthfulQA"
    },
    "safety_refusal": {
        "title": "Safety and Refusal",
        "metrics": ["asr", "silhouette"],
        "desc": "safety robustness and representational separability between benign and harmful inputs",
        "ds": "HarmBench + Alpaca"
    }
}

# Display names for metrics
METRIC_LABELS = {
    "activation_similarity": "Activation Similarity",
    "bleu": "BLEU Score",
    "bertscore_response": "BERTScore (Response)",
    "delta_log_prob": "$\\Delta$ Log-Prob",
    "entropy_shift": "Entropy Shift",
    "mirroring_rate": "Mirroring Rate (MR)",
    "asr": "ASR",
    "silhouette": "Silhouette Score"
}

def generate_paper_results_code(style_name="Politeness", style_folder="polite"):
    models = ["G-2B", "G-7B", "L3.1-8B", "L3.2-3B", "Q2.5-1.5B", "Q2.5-7B"]
    places = ["global", "prefix", "suffix"]

    latex_output = []

    for axis_key, info in AXES_META.items():
        # Main Section for the Behavioral Axis
        latex_output.append(f"\\section{{{info['title']}}}")
        latex_output.append(f"\\label{{sec:{axis_key}}}")

        # Subsection for the Style
        latex_output.append(f"\\subsection{{{style_name} Results}}")

        for metric in info['metrics']:
            m_label = METRIC_LABELS.get(metric, metric)

            # Base directory logic (safety metrics are in safety_polite)
            base_dir = "imgs/combined_plots/safety_polite" if metric in ["asr", "silhouette"] else f"imgs/combined_plots/{style_folder}"

            # 1. Line per Model Grid
            latex_output.append(generate_grid(metric, "line_per_model", models, m_label, info, style_name, base_dir))

            # 2. Line per Place Grid
            latex_output.append(generate_grid(metric, "line_per_place", places, m_label, info, style_name, base_dir))

            # 3. Dual Radar Comparison
            latex_output.append(generate_radar_pair(metric, m_label, info, style_name, base_dir))

            # 4. Ridge Plot
            latex_output.append(generate_ridge(metric, m_label, info, style_name, base_dir))

        latex_output.append("\\newpage")

    return "\n".join(latex_output)

def generate_grid(metric, p_type, items, label, info, style, base_dir):
    latex = ["\\begin{figure}[H]", "    \\centering"]
    for i, item in enumerate(items):
        path = f"{base_dir}/{p_type}/{metric}_{p_type}__{item}.png"
        latex.append(f"    \\begin{{subfigure}}[b]{{0.31\\textwidth}}")
        latex.append(f"        \\centering")
        latex.append(f"        \\includegraphics[width=\\textwidth]{{{path}}}")
        latex.append(f"        \\caption{{{item}}}")
        latex.append(f"    \\end{{subfigure}}")
        if (i + 1) % 3 == 0: latex.append("    \\\\ \\vspace{0.2cm}")
        else: latex.append("    \\hfill")

    caption = (f"\\textbf{{{label} vs. {style} Strength ({info['ds']}).}} "
               f"Comparison of {info['desc']} grouped by {p_type.split('_')[-1]}. ")
    latex.extend([f"    \\caption{{{caption}}}", f"    \\label{{fig:{metric}_{p_type}}}", "\\end{figure}"])
    return "\n".join(latex)

def generate_radar_pair(metric, label, info, style, base_dir):
    path_m = f"{base_dir}/{metric}_radar_axes_models.png"
    path_p = f"{base_dir}/{metric}_radar_axes_places.png"
    latex = [
        "\\begin{figure}[H]", "    \\centering",
        f"    \\begin{{subfigure}}[b]{{0.48\\textwidth}}", f"        \\includegraphics[width=\\textwidth]{{{path_m}}}", "        \\caption{Axes: Models}", "    \\end{subfigure} \\hfill",
        f"    \\begin{{subfigure}}[b]{{0.48\\textwidth}}", f"        \\includegraphics[width=\\textwidth]{{{path_p}}}", "        \\caption{Axes: Positions}", "    \\end{subfigure}",
        f"    \\caption{{\\textbf{{{label} Radar Profile ({info['ds']}).}} {label} visualized across model and positional axes under {style.lower()} perturbations.}}",
        "\\end{figure}"
    ]
    return "\n".join(latex)

def generate_ridge(metric, label, info, style, base_dir):
    path = f"{base_dir}/ridge_plots/{metric}_ridge.png"
    latex = [
        "\\begin{figure}[H]", "    \\centering",
        f"    \\includegraphics[width=0.8\\textwidth]{{{path}}}",
        f"    \\caption{{\\textbf{{{label} Distribution ({info['ds']}).}} Ridge plot showing the density of {info['desc']} as {style.lower()} strength increases.}}",
        "\\end{figure}"
    ]
    return "\n".join(latex)

# Print everything
print(generate_paper_results_code())