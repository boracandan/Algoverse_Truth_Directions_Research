"""
Figure 4 Replication: Cross-Task AUROC Generalization Matrices & Heatmaps
Model: DeepSeek-R1-Distill-8B
Conditions: Plaintext (no-prompt) vs Chain-of-Thought (cot-zero-shot)
Layers: Layer 25 (Paper Parity) & Layer 30 (Late-Stage Convergence)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def plot_single_layer_triplet(df, model_name, layer, task_order, output_path):
    plain_mat = build_matrix(df, model_name, "no-prompt", layer, task_order)
    cot_mat = build_matrix(df, model_name, "cot-zero-shot", layer, task_order)
    diff_mat = cot_mat - plain_mat
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    n = len(task_order)
    
    panels = [
        ("Plaintext Baseline (No-Prompt)", plain_mat, "RdBu", 0.0, 1.0, "AUROC"),
        ("Chain-of-Thought (CoT Zero-Shot)", cot_mat, "RdBu", 0.0, 1.0, "AUROC"),
        (r"$\Delta$ Transfer Gain (CoT - Plain)", diff_mat, "coolwarm", -0.4, 0.4, r"$\Delta$ AUROC")
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


def plot_multi_layer_comparison(df, model_name, layers, task_order, output_path):
    fig, axes = plt.subplots(len(layers), 3, figsize=(21, 6.2 * len(layers)))
    n = len(task_order)
    
    for row_idx, layer in enumerate(layers):
        plain_mat = build_matrix(df, model_name, "no-prompt", layer, task_order)
        cot_mat = build_matrix(df, model_name, "cot-zero-shot", layer, task_order)
        diff_mat = cot_mat - plain_mat
        
        panels = [
            (f"Plaintext (No-Prompt) | Layer {layer}", plain_mat, "RdBu", 0.0, 1.0, "AUROC"),
            (f"Chain-of-Thought (CoT) | Layer {layer}", cot_mat, "RdBu", 0.0, 1.0, "AUROC"),
            (rf"$\Delta$ Transfer (CoT - Plain) | Layer {layer}", diff_mat, "coolwarm", -0.4, 0.4, r"$\Delta$ AUROC")
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
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv)
    task_order = ["A1", "A2", "A3", "F0", "F1", "F2", "F3", "F4", "F5"]
    
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Model: {args.model}")
    print(f"Conditions: {df['train_condition'].unique()}")
    
    # 1. Plot Layer 25 Triplet
    l25_path = os.path.join(args.output_dir, "figure_4_cross_task_deepseek_l25.png")
    plot_single_layer_triplet(df, args.model, 25, task_order, l25_path)
    
    # 2. Plot Layer 30 Triplet
    l30_path = os.path.join(args.output_dir, "figure_4_cross_task_deepseek_l30.png")
    plot_single_layer_triplet(df, args.model, 30, task_order, l30_path)
    
    # 3. Plot Multi-Layer Comparison (Layer 25 and Layer 30 together)
    comp_path = os.path.join(args.output_dir, "figure_4_cross_task_comparison.png")
    plot_multi_layer_comparison(df, args.model, [25, 30], task_order, comp_path)


if __name__ == "__main__":
    main()
