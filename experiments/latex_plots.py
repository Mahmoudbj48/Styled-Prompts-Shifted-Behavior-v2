import os

# Formal Axis Definitions grounded in your paper
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
    "style_mirroring": {
        "title": "Style Mirroring",
        "metrics": ["mirroring_rate"],
        "desc": "the proportion of instances where stylistic variation propagates into response-level behavior",
    },
    "safety_refusal": {
        "title": "Safety and Refusal",
        "metrics": ["asr", "silhouette"],
        "desc": "safety robustness and representational separability between benign and harmful inputs",
    }
}

METRIC_LABELS = {
    "activation_similarity": "Activation Similarity",
    "bleu": "BLEU Score",
    "bertscore_response": "BERTScore (Response)",
    "delta_log_prob": "$\\Delta$ Log-Prob",
    "entropy_shift": "Entropy Shift",
    "mirroring_rate": "Mirroring Rate (MR)",
    "asr": "ASR",
    "silhouette": "Silhouette Score",
    "bertscore_prompt": "BERTScore (Prompt)"
}

def generate_full_paper_results():
    # Model subsets
    all_models = ["G-2B", "G-7B", "L3.1-8B", "L3.2-3B", "Q2.5-1.5B", "Q2.5-7B"]
    big_models = ["G-7B", "L3.1-8B", "Q2.5-7B"]
    places = ["global", "prefix", "suffix"]

    latex_output = []

    # 1. MAIN BEHAVIORAL AXES
    for axis_key, info in AXES_META.items():
        latex_output.append(f"\\section{{{info['title']}}}")
        latex_output.append(f"\\label{{sec:{axis_key}}}")

        # Subsection: TruthfulQA (or HarmBench for safety)
        ds_main = "HarmBench" if axis_key == "safety_refusal" else "TruthfulQA"
        base_main = "imgs/safety_polite" if axis_key == "safety_refusal" else "imgs/polite"

        latex_output.append(f"\\subsection{{Politeness Results ({ds_main})}}")
        for metric in info['metrics']:
            latex_output.append(generate_metric_block(metric, all_models, places, info, "Politeness", ds_main, base_main))

        # Subsection: Natural Questions (Excluding Safety)
        if axis_key != "safety_refusal":
            latex_output.append(f"\\subsection{{Politeness Results (Natural Questions)}}")
            for metric in info['metrics']:
                latex_output.append(generate_metric_block(metric, big_models, places, info, "Politeness", "Natural Questions", "imgs/polite_NQ"))

        latex_output.append("\\newpage")

    # 2. SEMANTIC PRESERVATION SECTION
    latex_output.append("\\section{Semantic Preservation Analysis}")
    latex_output.append("\\label{sec:semantic_preservation}")

    # Combined Summary Plot for BERTScore Prompt
    latex_output.append("\\subsection{Global BERTScore Prompt Summary}")
    latex_output.append(f"""\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{imgs/bertscore_prompt_line.png}}
    \\caption{{\\textbf{{Prompt Semantic Preservation (All Datasets).}} This plot illustrates the semantic similarity between neutral and styled prompts across TruthfulQA and Natural Questions. All variants utilized in this study satisfy the preservation threshold of 0.85.}}
    \\label{{fig:global_bertscore_preservation}}
\\end{{figure}}""")

    # Grid details for TruthfulQA
    latex_output.append("\\subsection{Politeness BERTScore (Prompt) Details}")
    info_bs = {"desc": "semantic similarity of the prompt to ensure content preservation"}
    latex_output.append(generate_metric_block("bertscore_prompt", all_models, places, info_bs, "Politeness", "TruthfulQA", "imgs/polite"))

    # Grid details for NQ
    latex_output.append(generate_metric_block("bertscore_prompt", big_models, places, info_bs, "Politeness", "Natural Questions", "imgs/polite_NQ"))

    return "\n".join(latex_output)

def generate_metric_block(metric, models, places, info, style, ds, base_dir):
    label = METRIC_LABELS.get(metric, metric)
    desc = info.get('desc', '')
    ds_slug = ds.lower().replace(" ", "_")

    blocks = [
        generate_grid(metric, "line_per_model", models, label, desc, style, ds, base_dir, ds_slug),
        generate_grid(metric, "line_per_place", places, label, desc, style, ds, base_dir, ds_slug),
        generate_radar_pair(metric, label, desc, style, ds, base_dir, ds_slug),
        generate_ridge(metric, label, desc, style, ds, base_dir, ds_slug)
    ]
    return "\n".join(blocks)

def generate_grid(metric, p_type, items, label, desc, style, ds, base_dir, ds_slug):
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

    cap = f"\\textbf{{{label} vs. {style} ({ds}).}} {label} grouped by {p_type.split('_')[-1]}."
    latex.extend([f"    \\caption{{{cap}}}", f"    \\label{{fig:{ds_slug}_{metric}_{p_type}}}", "\\end{figure}"])
    return "\n".join(latex)

def generate_radar_pair(metric, label, desc, style, ds, base_dir, ds_slug):
    path_m = f"{base_dir}/{metric}_radar_axes_models.png"
    path_p = f"{base_dir}/{metric}_radar_axes_places.png"
    return f"""\\begin{{figure}}[H]
    \\centering
    \\begin{{subfigure}}[b]{{0.48\\textwidth}} \\includegraphics[width=\\textwidth]{{{path_m}}} \\caption{{Axes: Models}} \\end{{subfigure}} \\hfill
    \\begin{{subfigure}}[b]{{0.48\\textwidth}} \\includegraphics[width=\\textwidth]{{{path_p}}} \\caption{{Axes: Positions}} \\end{{subfigure}}
    \\caption{{\\textbf{{{label} Radar Analysis ({ds}).}} {label} across axes for {style}.}}
    \\label{{fig:{ds_slug}_{metric}_radar}}
\\end{{figure}}"""

def generate_ridge(metric, label, desc, style, ds, base_dir, ds_slug):
    path = f"{base_dir}/ridge_plots/{metric}_ridge.png"
    return f"""\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{{path}}}
    \\caption{{\\textbf{{{label} Distribution ({ds}).}} Ridge plot of {desc} as {style.lower()} strength increases.}}
    \\label{{fig:{ds_slug}_{metric}_ridge}}
\\end{{figure}}"""

if __name__ == "__main__":
    print(generate_full_paper_results())