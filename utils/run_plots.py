"""
Usage:
  python utils/run_plots.py \
      --runs results/politeness/run_*/summary.csv \
      --out_dir results/combined_plots/polite \
      --dataset_name "TruthfulQA" \
      --style_name "politeness"

  python utils/run_plots.py \
      --runs results/combined_plots/safety_polite/combined_means_by_model_place_strength.csv \
      --out_dir results/combined_plots/safety_polite

  # length_variation with extra structuredness plot:
  python utils/run_plots.py \
      --runs results/length_variation/run_a/summary.csv results/length_variation/run_b/summary.csv \
      --row_runs results/length_variation/run_a/full_results_all_models.csv results/length_variation/run_b/full_results_all_models.csv \
      --out_dir results/combined_plots/length_variation \
      --dataset_name "TruthfulQA" \
      --style_name "length_variation"

Plots produced per metric:
  - Line TYPE 1 : all models + places together     (<metric>_line.png)
  - Line TYPE 2 : per model, all places            (line_per_model/<metric>_line_per_model__<model>.png)
  - Line TYPE 3 : per place, all models            (line_per_place/<metric>_line_per_place__<place>.png)
  - Radar A     : axes=places, colors=models       (radar/<metric>_radar_axes_places.png)
  - Radar B     : axes=models,  colors=places      (radar/<metric>_radar_axes_models.png)
  - Radar C     : axes=metrics, all model×place    (radar/all_metrics_radar_axes_metrics.png)
  - Ridge       : distribution per strength        (ridge_plots/<metric>_ridge.png)

Extra behavior:
  - When --style_name length_variation is used, this script also tries to
    generate the structuredness-specific plots from utils.plots under
    <out_dir>/plots_structuredness.
  - Those extra plots use --row_runs if provided; otherwise they fall back to --runs.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _maybe_generate_structuredness_plots(args) -> None:
    """
    For length_variation runs, also generate plots_structuredness/
    length_ratio_boxplot.png when row-level CSVs are available.
    """
    if args.style_name != "length_variation":
        return

    from utils.plots import load_results_csvs, make_structuredness_plots

    row_inputs = args.row_runs if args.row_runs else args.runs

    try:
        df_rows = load_results_csvs(row_inputs)
    except Exception as e:
        print(f"[WARN] Could not load row-level inputs for structuredness plots: {e}")
        return

    if args.models is not None and "model" in df_rows.columns:
        df_rows = df_rows[df_rows["model"].isin(args.models)]
    if args.places is not None and "place" in df_rows.columns:
        df_rows = df_rows[df_rows["place"].isin(args.places)]

    required = {"strength", "prompt_orig", "prompt_pert"}
    missing = [c for c in sorted(required) if c not in df_rows.columns]
    if missing:
        print(
            "[WARN] Skipping plots_structuredness: row-level inputs are missing "
            f"required columns {missing}. "
            "Pass --row_runs with full_results_all_models.csv files."
        )
        return

    struct_dir = os.path.join(args.out_dir, "plots_structuredness")
    os.makedirs(struct_dir, exist_ok=True)

    try:
        make_structuredness_plots(
            df_rows,
            struct_dir,
            dataset_name=args.dataset_name or "",
            models_filter=args.models,
            include_title=True,
            save_pdf=args.save_pdf,
        )
        print(f"[DONE] Structuredness-specific plots saved to: {struct_dir}")
    except Exception as e:
        print(f"[WARN] Structuredness-specific plots failed: {e}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help=(
            "Inputs for aggregate plots. Typically summary.csv files, "
            "combined_means_by_model_place_strength.csv files, or run directories "
            "that load_all_runs(...) can resolve."
        ),
    )
    parser.add_argument(
        "--row_runs",
        nargs="+",
        default=None,
        help=(
            "Optional row-level inputs used only for structuredness-specific plots "
            "when --style_name length_variation. Typically full_results_all_models.csv "
            "files or run directories containing them."
        ),
    )

    parser.add_argument("--out_dir", required=True,
                        help="Directory to save all plots.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Restrict to these models.")
    parser.add_argument("--places", nargs="+", default=None,
                        help="Restrict to these placement positions.")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Restrict to these metrics (optional; defaults to all allowed).")
    parser.add_argument("--radar_norm", default="minmax",
                        choices=["none", "minmax", "zscore"],
                        help="Normalisation for radar plots.")
    parser.add_argument("--dataset_name", default=None,
                        choices=["TruthfulQA", "Natural Questions"],
                        help="Dataset label shown in plot titles.")
    parser.add_argument("--style_name", default=None,
                        help="Style type (e.g. politeness, spacing, letter_case, "
                             "punctuation, length_variation, inter_vs_imper). "
                             "Added to all plot titles; also controls which strength "
                             "value is excluded from radar averages.")
    parser.add_argument("--save_pdf", action="store_true",
                        help="Also save each plot as PDF.")
    args = parser.parse_args()

    from utils.plots import (
        load_all_runs,
        select_metrics,
        aggregate_plot_metric_lines,
        plot_metric_lines_per_model,
        plot_metric_lines_per_place,
        plot_radar_places_axes,
        plot_radar_models_axes,
        plot_radar_metrics_axes,
        plot_metric_ridge,
        ALLOWED_METRICS,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_all_runs(args.runs)

    if args.models is not None:
        df = df[df["model"].isin(args.models)]
    if args.places is not None:
        df = df[df["place"].isin(args.places)]

    if df.empty:
        raise SystemExit("[ERROR] No data left after filtering.")

    metrics = select_metrics(df)
    if args.metrics is not None:
        metrics = [m for m in metrics if m in set(args.metrics)]

    # ── Line plots ────────────────────────────────────────────────────────────
    for metric in metrics:
        print(f"[PLOT] {metric}")

        aggregate_plot_metric_lines(
            df, metric,
            out_path=os.path.join(args.out_dir, f"{metric}_line.png"),
            models=args.models,
            places=args.places,
            save_pdf=args.save_pdf,
            dataset_name=args.dataset_name,
            style_name=args.style_name,
        )

        plot_metric_lines_per_model(
            df, metric,
            out_dir=os.path.join(args.out_dir, "line_per_model"),
            models=args.models,
            places=args.places,
            save_pdf=args.save_pdf,
            dataset_name=args.dataset_name,
            style_name=args.style_name,
        )

        plot_metric_lines_per_place(
            df, metric,
            out_dir=os.path.join(args.out_dir, "line_per_place"),
            models=args.models,
            places=args.places,
            save_pdf=args.save_pdf,
            dataset_name=args.dataset_name,
            style_name=args.style_name,
        )

    # ── Radar plots ───────────────────────────────────────────────────────────
    radar_dir = os.path.join(args.out_dir, "radar")
    os.makedirs(radar_dir, exist_ok=True)

    for metric in metrics:
        plot_radar_places_axes(
            df, metric,
            out_path=os.path.join(radar_dir, f"{metric}_radar_axes_places.png"),
            models=args.models,
            places=args.places,
            radar_norm=args.radar_norm,
            save_pdf=args.save_pdf,
            dataset_name=args.dataset_name,
            style_name=args.style_name,
        )

        plot_radar_models_axes(
            df, metric,
            out_path=os.path.join(radar_dir, f"{metric}_radar_axes_models.png"),
            models=args.models,
            places=args.places,
            radar_norm=args.radar_norm,
            save_pdf=args.save_pdf,
            dataset_name=args.dataset_name,
            style_name=args.style_name,
        )

    metrics_for_radar_c = [m for m in metrics if m in ALLOWED_METRICS]
    plot_radar_metrics_axes(
        df,
        metrics=metrics_for_radar_c,
        out_path=os.path.join(radar_dir, "all_metrics_radar_axes_metrics.png"),
        models=args.models,
        places=args.places,
        radar_norm=args.radar_norm,
        save_pdf=args.save_pdf,
        dataset_name=args.dataset_name,
        style_name=args.style_name,
    )

    # ── Ridge plots ───────────────────────────────────────────────────────────
    ridge_dir = os.path.join(args.out_dir, "ridge_plots")
    os.makedirs(ridge_dir, exist_ok=True)

    for metric in metrics:
        plot_metric_ridge(
            df, metric,
            out_path=os.path.join(ridge_dir, f"{metric}_ridge.png"),
            models=args.models,
            places=args.places,
            save_pdf=args.save_pdf,
            dataset_name=args.dataset_name,
            style_name=args.style_name,
        )

    _maybe_generate_structuredness_plots(args)

    print(f"\n[DONE] All plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()