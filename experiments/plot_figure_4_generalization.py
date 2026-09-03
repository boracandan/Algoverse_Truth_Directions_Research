"""
Figure 4 Replication: Cross-Task AUROC Generalization Matrices & Heatmaps
Model: DeepSeek-R1-Distill-8B
Conditions: any two conditions present in results_database.csv (pass via
--left-condition / --right-condition), e.g. "no-prompt" vs "cot-zero-shot",
or "no-prompt" vs "no-prompt-chat-template".
Layers: Layer 25 (Paper Parity) & Layer 30 (Late-Stage Convergence)

This replaces the old plot_figure_4_generalization.py / plot_figure_4_sentence_based_cot.py
pair -- those two scripts differed only in which condition was hardcoded as the "right"
side and in the output filenames, so they're consolidated into one parameterized script
here instead of duplicating the plotting logic per condition pair.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def slug(condition):
    """Turn a condition string into a filename-safe slug, e.g. 'no-prompt-chat-template'
    -> 'no_prompt_chat_template'."""
    return condition.replace("-", "_")


def label(condition):
    """Human-readable label for a condition, for plot titles/legends."""
    labels = {
        "no-prompt": "Plaintext (No-Prompt)",
        "no-prompt-chat-template": "No-Prompt (Chat Template)",
        "cot-zero-shot": "Chain-of-Thought (CoT Zero-Shot)",
        "sentence-based-CoT": "Sentence-Based CoT",
    }
    return labels.get(condition, condition)


def build_matrix(df, model_name, condition, layer, task_order):
    n = len(task_order)
    matrix = np.full((n, n), np.nan)

    sub_df = df[
        (df["model"] == model_name) &
        (df["train_condition"] == condition) &
        (df["test_condition"] == condition) &
        (df["layer"] == layer)
    ]

    for i, train_task in enumerate(task_order):
        for j, test_task in enumerate(task_order):
            row = sub_df[
                (sub_df["train_task"] == train_task) &
                (sub_df["test_task"] == test_task)
            ]
            if len(row) > 0:
                matrix[i, j] = float(row.iloc[0]["auroc"])

    return matrix


def plot_single_layer_triplet(df, model_name, left_condition, right_condition, layer, task_order, output_path):
    left_mat = build_matrix(df, model_name, left_condition, layer, task_order)
    right_mat = build_matrix(df, model_name, right_condition, layer, task_order)
    diff_mat = right_mat - left_mat

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    n = len(task_order)

    panels = [
        (label(left_condition), left_mat, "RdBu", 0.0, 1.0, "AUROC"),
        (label(right_condition), right_mat, "RdBu", 0.0, 1.0, "AUROC"),
        (rf"$\Delta$ Transfer Gain ({label(right_condition)} - {label(left_condition)})", diff_mat, "coolwarm", -0.4, 0.4, r"$\Delta$ AUROC")
    ]

    for ax, (title, mat, cmap, vmin, vmax, cbar_lbl) in zip(axes, panels):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                if not np.isnan(val):
                    if cmap == "RdBu":
                        txt_color = "white" if (val < 0.35 or val > 0.75) else "black"
                        txt = f"{val:.2f}"
                    else:
                        txt_color = "white" if abs(val) > 0.25 else "black"
                        txt = f"{val:+.2f}"
                    ax.text(j, i, txt, ha="center", va="center", color=txt_color, fontsize=8.5, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.04)
        cbar.set_label(cbar_lbl, fontsize=10.5, fontweight="semibold")

        ax.set_xticks(range(n))
        ax.set_xticklabels(task_order, fontsize=10, fontweight="medium")
        ax.set_yticks(range(n))
        ax.set_yticklabels(task_order, fontsize=10, fontweight="medium")
        ax.set_xlabel("Test Task", fontsize=11, fontweight="semibold")
        ax.set_ylabel("Train Task", fontsize=11, fontweight="semibold")
        ax.set_title(f"{title}\nLayer {layer} ({model_name})", fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_multi_layer_comparison(df, model_name, left_condition, right_condition, layers, task_order, output_path):
    fig, axes = plt.subplots(len(layers), 3, figsize=(21, 6.2 * len(layers)))
    n = len(task_order)

    for row_idx, layer in enumerate(layers):
        left_mat = build_matrix(df, model_name, left_condition, layer, task_order)
        right_mat = build_matrix(df, model_name, right_condition, layer, task_order)
        diff_mat = right_mat - left_mat

        panels = [
            (f"{label(left_condition)} | Layer {layer}", left_mat, "RdBu", 0.0, 1.0, "AUROC"),
            (f"{label(right_condition)} | Layer {layer}", right_mat, "RdBu", 0.0, 1.0, "AUROC"),
            (rf"$\Delta$ Transfer ({label(right_condition)} - {label(left_condition)}) | Layer {layer}", diff_mat, "coolwarm", -0.4, 0.4, r"$\Delta$ AUROC")
        ]

        for col_idx, (title, mat, cmap, vmin, vmax, cbar_lbl) in enumerate(panels):
            ax = axes[row_idx, col_idx] if len(layers) > 1 else axes[col_idx]
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

            for i in range(n):
                for j in range(n):
                    val = mat[i, j]
                    if not np.isnan(val):
                        if cmap == "RdBu":
                            txt_color = "white" if (val < 0.35 or val > 0.75) else "black"
                            txt = f"{val:.2f}"
                        else:
                            txt_color = "white" if abs(val) > 0.25 else "black"
                            txt = f"{val:+.2f}"
                        ax.text(j, i, txt, ha="center", va="center", color=txt_color, fontsize=8, fontweight="bold")

            cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.04)
            cbar.set_label(cbar_lbl, fontsize=10.5, fontweight="semibold")

            ax.set_xticks(range(n))
            ax.set_xticklabels(task_order, fontsize=9.5, fontweight="medium")
            ax.set_yticks(range(n))
            ax.set_yticklabels(task_order, fontsize=9.5, fontweight="medium")
            ax.set_xlabel("Test Task", fontsize=10.5, fontweight="semibold")
            ax.set_ylabel("Train Task", fontsize=10.5, fontweight="semibold")
            ax.set_title(f"{title} ({model_name})", fontsize=11.5, fontweight="bold", pad=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 4 Cross-Task Generalization Heatmaps")
    parser.add_argument("--csv", default="experiments/results_database.csv", help="Path to results_database.csv")
    parser.add_argument("--model", default="deepseek-r1-distill-8b", help="Model name identifier")
    parser.add_argument("--output_dir", default="experiments/figures", help="Output directory for plots")
    parser.add_argument("--left-condition", default="no-prompt", help="Condition for the left/baseline panel (must match a train_condition/test_condition value in the CSV)")
    parser.add_argument("--right-condition", default="cot-zero-shot", help="Condition for the right/comparison panel")
    parser.add_argument("--layers", nargs="+", type=int, default=[25, 30], help="Layers to plot (first two are used for the single-layer triplets; all are used in the multi-layer comparison)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    task_order = ["A1", "A2", "A3", "F0", "F1", "F2", "F3", "F4", "F5"]

    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Model: {args.model}")
    print(f"Conditions available: {df['train_condition'].unique()}")
    print(f"Comparing: {args.left_condition} vs {args.right_condition}")

    left_slug, right_slug = slug(args.left_condition), slug(args.right_condition)
    pair_slug = f"{left_slug}_vs_{right_slug}"

    for layer in args.layers:
        out_path = os.path.join(args.output_dir, f"figure_4_cross_task_deepseek_{pair_slug}_l{layer}_new.png")
        plot_single_layer_triplet(df, args.model, args.left_condition, args.right_condition, layer, task_order, out_path)

    comp_path = os.path.join(args.output_dir, f"figure_4_cross_task_comparison_{pair_slug}_new.png")
    plot_multi_layer_comparison(df, args.model, args.left_condition, args.right_condition, args.layers, task_order, comp_path)


if __name__ == "__main__":
    main()
