"""


Usage:
  python utils/run_plots.py \
      --runs results/politeness/run_*/plots_metrics/combined_means_by_model_place_strength.csv \
      --out_dir results/combined_plots/polite \
      --dataset_name "TruthfulQA"  --style_name "politeness"

  python utils/run_plots.py \
      --runs results/combined_plots/safety_polite/combined_means_by_model_place_strength.csv \
      --out_dir results/combined_plots/safety_polite

Plots produced per metric:
  - Line TYPE 1 : all models + places together     (<metric>_line.png)
  - Line TYPE 2 : per model, all places            (line_per_model/<metric>_line_per_model__<model>.png)
  - Line TYPE 3 : per place, all models            (line_per_place/<metric>_line_per_place__<place>.png)
  - Radar A     : axes=places, colors=models       (radar/<metric>_radar_axes_places.png)
  - Radar B     : axes=models,  colors=places      (radar/<metric>_radar_axes_models.png)
  - Radar C     : axes=metrics, all model×place    (radar/all_metrics_radar_axes_metrics.png)
  - Ridge       : distribution per strength        (ridge_plots/<metric>_ridge.png)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Paths to combined_means_by_model_place_strength CSV files.")
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

        # TYPE 1: all models + all places
        aggregate_plot_metric_lines(
            df, metric,
            out_path=os.path.join(args.out_dir, f"{metric}_line.png"),
            models=args.models,
            places=args.places,
            save_pdf=args.save_pdf,
            dataset_name=args.dataset_name,
            style_name=args.style_name,
        )

        # TYPE 2: per model (one figure per model, lines = places)
        plot_metric_lines_per_model(
            df, metric,
            out_dir=os.path.join(args.out_dir, "line_per_model"),
            models=args.models,
            places=args.places,
            save_pdf=args.save_pdf,
            dataset_name=args.dataset_name,
            style_name=args.style_name,
        )

        # TYPE 3: per place (one figure per place, lines = models)
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
        # Radar A: axes=places, colors=models
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
        # Radar B: axes=models, colors=places
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

    # Radar C: axes=metrics, one line per (model, place)
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

    print(f"\n[DONE] All plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()