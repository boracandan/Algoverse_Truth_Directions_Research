"""
Batch driver for plot_figure_4_generalization.py.

Rather than invoking the parameterized script by hand once per model/condition-pair, this
runs it for every (model, left-condition, right-condition) combination in CANDIDATE_PAIRS
that is actually present in results_database.csv for that model -- a single "plug and run"
entry point. Pairs where either condition is missing for a given model are skipped (e.g.
granite-4.2-8b currently only has sentence-based-CoT, so none of its pairs run yet).
"""

import argparse
import pandas as pd

from plot_figure_4_generalization import (
    plot_single_layer_triplet,
    plot_multi_layer_comparison,
    slug,
)

TASK_ORDER = ["A1", "A2", "A3", "F0", "F1", "F2", "F3", "F4", "F5"]

# Condition pairs worth comparing, as (left, right). Add new pairs here as new conditions
# get added to the CSV -- nothing else needs to change.
CANDIDATE_PAIRS = [
    ("no-prompt", "cot-zero-shot"),
    ("no-prompt", "no-prompt-chat-template"),
    ("no-prompt", "sentence-based-CoT"),
    ("cot-zero-shot", "sentence-based-CoT"),
]


def main():
    parser = argparse.ArgumentParser(description="Run Figure 4 for every available model x condition-pair combination")
    parser.add_argument("--csv", default="experiments/results_database.csv", help="Path to results_database.csv")
    parser.add_argument("--output_dir", default="experiments/figures", help="Output directory for plots")
    parser.add_argument("--layers", nargs="+", type=int, default=[25, 30], help="Layers to plot")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    for model in sorted(df["model"].unique()):
        model_df = df[df["model"] == model]
        available = set(model_df["train_condition"].unique())
        model_slug = slug(model)

        for left, right in CANDIDATE_PAIRS:
            if left not in available or right not in available:
                print(f"Skipping {model}: {left} vs {right} (missing condition data)")
                continue

            pair_slug = f"{slug(left)}_vs_{slug(right)}"
            print(f"\n=== {model}: {left} vs {right} ===")

            for layer in args.layers:
                out_path = f"{args.output_dir}/figure_4_cross_task_{model_slug}_{pair_slug}_l{layer}_new.png"
                plot_single_layer_triplet(model_df, model, left, right, layer, TASK_ORDER, out_path)

            comp_path = f"{args.output_dir}/figure_4_cross_task_{model_slug}_comparison_{pair_slug}_new.png"
            plot_multi_layer_comparison(model_df, model, left, right, args.layers, TASK_ORDER, comp_path)


if __name__ == "__main__":
    main()
