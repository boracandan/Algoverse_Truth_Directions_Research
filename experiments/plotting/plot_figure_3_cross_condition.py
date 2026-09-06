"""
Figure 3 Replication: Cross-Condition Generalization (No-Prompt <-> Sentence-Based CoT)
Model: DeepSeek-R1-Distill-8B
Layers: All 33 Layers (0 through 32)
Tasks: All 9 Tasks (Arithmetic: A1-A3, Factual: F0-F5)

Replicates the spirit of Figure 3 from Poulis et al., "Testing the Limits of Truth Directions in LLMs".
Unlike Poulis (who tested only one direction), this generates figures for both directions:
  - Direction A (CoT -> No-Prompt): train_condition="sentence-based-CoT", test_condition="no-prompt-chat-template"
  - Direction B (No-Prompt -> CoT): train_condition="no-prompt-chat-template", test_condition="sentence-based-CoT"

Four distinct figures matching Poulis Fig 2/3 layout (shared axes: Layer vs AUROC):
  1. Arithmetic (A1-A3) -- Direction A (CoT -> No-Prompt)
  2. Arithmetic (A1-A3) -- Direction B (No-Prompt -> CoT)
  3. Factual (F0-F5) -- Direction A (CoT -> No-Prompt)
  4. Factual (F0-F5) -- Direction B (No-Prompt -> CoT)
Plus a consolidated 2x2 comprehensive comparison panel.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Canonical task groups
ARITH_TASKS = ["A1", "A2", "A3"]
FACT_TASKS = ["F0", "F1", "F2", "F3", "F4", "F5"]
ALL_TASKS = ARITH_TASKS + FACT_TASKS

# Standard repository style map
STYLE_MAP = {
    "A1": {"color": "#1f77b4", "marker": "*", "label": "A1: 1-Op Arithmetic"},
    "A2": {"color": "#ff7f0e", "marker": "X", "label": "A2: 2-Op Arithmetic"},
    "A3": {"color": "#17becf", "marker": "h", "label": "A3: 3-Op Arithmetic"},
    "F0": {"color": "#2ca02c", "marker": "o", "label": "F0: Atomic Facts"},
    "F1": {"color": "#9467bd", "marker": "s", "label": "F1: Negations"},
    "F2": {"color": "#8c564b", "marker": "^", "label": "F2: Conjunctions"},
    "F3": {"color": "#e377c2", "marker": "D", "label": "F3: 2-Count Cardinality"},
    "F4": {"color": "#d62728", "marker": "v", "label": "F4: 5-Count Cardinality"},
    "F5": {"color": "#7f7f7f", "marker": "P", "label": "F5: Set Comparison"},
}


def load_cross_condition_data(csv_path, model_name="deepseek-r1-distill-8b"):
    """Load results_database.csv and filter down to the cross-condition pairs."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Database CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Filter for the relevant model and same-task cross-condition evaluations
    mask = (
        (df["model"] == model_name) &
        (df["train_task"] == df["test_task"]) &
        (
            ((df["train_condition"] == "sentence-based-CoT") & (df["test_condition"] == "no-prompt-chat-template")) |
            ((df["train_condition"] == "no-prompt-chat-template") & (df["test_condition"] == "sentence-based-CoT"))
        )
    )
    cross_df = df[mask].copy()
    cross_df["layer"] = cross_df["layer"].astype(int)
    cross_df["auroc"] = cross_df["auroc"].astype(float)
    return cross_df


def plot_single_panel(ax, sub_df, tasks, title, show_ylabel=True):
    """Render a single shared-axis panel (Arithmetic or Factual)."""
    for task in tasks:
        cfg = STYLE_MAP[task]
        task_data = sub_df[sub_df["train_task"] == task].sort_values("layer")
        if len(task_data) > 0:
            ax.plot(
                task_data["layer"],
                task_data["auroc"],
                color=cfg["color"],
                marker=cfg["marker"],
                markersize=6,
                linewidth=2.2,
                alpha=0.9,
                label=cfg["label"],
            )

    # Chance level baseline
    ax.axhline(0.5, color="black", linestyle=":", alpha=0.5, linewidth=1.3, label="Chance (0.50)")

    ax.set_xlabel("Layer Index ($l$)", fontsize=12, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Cross-Condition AUROC", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim(0.35, 1.05)
    ax.set_xlim(-0.5, 32.5)
    ax.set_xticks(range(0, 33, 4))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)


def generate_figure(df, train_cond, test_cond, tasks, title, out_path):
    """Generate and save a standalone single-group figure."""
    sub_df = df[
        (df["train_condition"] == train_cond) &
        (df["test_condition"] == test_cond)
    ]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    plot_single_panel(ax, sub_df, tasks, title, show_ylabel=True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved figure: {out_path}")


def generate_all_figure_3_plots(csv_path="experiments/results_database.csv", output_dir="experiments/figures/figure_3"):
    """Generates the 4 required figures + 2x2 companion summary."""
    df = load_cross_condition_data(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loaded {len(df)} cross-condition rows from {csv_path}")

    # 1. Arithmetic Tasks -- Direction A (CoT -> No-Prompt)
    generate_figure(
        df,
        train_cond="sentence-based-CoT",
        test_cond="no-prompt-chat-template",
        tasks=ARITH_TASKS,
        title="Arithmetic Reasoning ($A_1-A_3$)\nDirection A: Train CoT $\\to$ Test No-Prompt",
        out_path=os.path.join(output_dir, "figure_3_arithmetic_cot_to_noprompt.png"),
    )

    # 2. Arithmetic Tasks -- Direction B (No-Prompt -> CoT)
    generate_figure(
        df,
        train_cond="no-prompt-chat-template",
        test_cond="sentence-based-CoT",
        tasks=ARITH_TASKS,
        title="Arithmetic Reasoning ($A_1-A_3$)\nDirection B: Train No-Prompt $\\to$ Test CoT",
        out_path=os.path.join(output_dir, "figure_3_arithmetic_noprompt_to_cot.png"),
    )

    # 3. Factual Tasks -- Direction A (CoT -> No-Prompt)
    generate_figure(
        df,
        train_cond="sentence-based-CoT",
        test_cond="no-prompt-chat-template",
        tasks=FACT_TASKS,
        title="Factual & Compositional Reasoning ($F_0-F_5$)\nDirection A: Train CoT $\\to$ Test No-Prompt",
        out_path=os.path.join(output_dir, "figure_3_factual_cot_to_noprompt.png"),
    )

    # 4. Factual Tasks -- Direction B (No-Prompt -> CoT)
    generate_figure(
        df,
        train_cond="no-prompt-chat-template",
        test_cond="sentence-based-CoT",
        tasks=FACT_TASKS,
        title="Factual & Compositional Reasoning ($F_0-F_5$)\nDirection B: Train No-Prompt $\\to$ Test CoT",
        out_path=os.path.join(output_dir, "figure_3_factual_noprompt_to_cot.png"),
    )

    # 5. Comprehensive 2x2 Comparison Grid
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), sharey=True)

    dir_a_df = df[(df["train_condition"] == "sentence-based-CoT") & (df["test_condition"] == "no-prompt-chat-template")]
    dir_b_df = df[(df["train_condition"] == "no-prompt-chat-template") & (df["test_condition"] == "sentence-based-CoT")]

    # Row 0: Arithmetic
    plot_single_panel(axes[0, 0], dir_a_df, ARITH_TASKS, "Arithmetic: Direction A (CoT $\\to$ No-Prompt)", show_ylabel=True)
    plot_single_panel(axes[0, 1], dir_b_df, ARITH_TASKS, "Arithmetic: Direction B (No-Prompt $\\to$ CoT)", show_ylabel=False)

    # Row 1: Factual
    plot_single_panel(axes[1, 0], dir_a_df, FACT_TASKS, "Factual: Direction A (CoT $\\to$ No-Prompt)", show_ylabel=True)
    plot_single_panel(axes[1, 1], dir_b_df, FACT_TASKS, "Factual: Direction B (No-Prompt $\\to$ CoT)", show_ylabel=False)

    plt.suptitle(
        "Figure 3: Cross-Condition Generalization of Truth Directions (DeepSeek-R1-Distill-8B)\n"
        "Directly Evaluating Transfer Between No-Prompt (Chat Template) and Sentence-Based CoT",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    comp_path = os.path.join(output_dir, "figure_3_cross_condition_comprehensive.png")
    plt.savefig(comp_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved 2x2 comprehensive figure: {comp_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Figure 3 Cross-Condition Generalization curves.")
    parser.add_argument("--csv", default="experiments/results_database.csv", help="Path to results_database.csv")
    parser.add_argument("--output_dir", default="experiments/figures/figure_3", help="Output directory for plots")
    args = parser.parse_args()

    generate_all_figure_3_plots(csv_path=args.csv, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
