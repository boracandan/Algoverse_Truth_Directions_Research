"""
Figure 5 Replication: 2D Representation Geometry of Truth Directions - Sentence-Based CoT
Evaluates Layer 25 internal representations across all 9 complexity tasks (F0-F5, A1-A3).
Grid layout matching paper:
  Row 1: A1, A2, A3
  Row 2: F0, F1, F2
  Row 3: F3, F4, F5
"""

from ast import literal_eval
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def fit_linear_probe(X_train, y_train, device="cuda" if torch.cuda.is_available() else "cpu"):
    """No-bias, train-mean-centered linear probe -- matches the paper's Appendix A.4 exactly
    (Adam, lr=1e-3, weight_decay=0.1, 1000 steps, BCEWithLogitsLoss, no bias term). Replaces
    sklearn's LogisticRegression (free intercept, no train-mean centering) so this probe
    uses the same methodology as train_probe_pytorch elsewhere in the project."""
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    train_mean = X_train_t.mean(dim=0)
    X_centered = (X_train_t - train_mean).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    probe = nn.Linear(X_train.shape[1], 1, bias=False).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=0.1)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(1000):
        optimizer.zero_grad()
        logits = probe(X_centered).squeeze(-1)
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimizer.step()

    w = probe.weight.detach().cpu().numpy().flatten()
    return w, train_mean.numpy()


def compute_task_projection(X_train, y_train, X_test, y_test):
    """
    Fits the no-bias linear probe to obtain the truth vector,
    then projects full distribution onto truth axis and orthogonal PCA.
    """
    w, _ = fit_linear_probe(X_train, y_train)

    # 1. Normalize truth direction vector
    norm_w = np.linalg.norm(w)
    v_hat = w / (norm_w if norm_w > 1e-9 else 1.0)

    # 2. Combine train + test sets for full distribution
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    # 3. Center the data
    X_centered = X_all - X_all.mean(axis=0)

    # 4. Horizontal coordinate: project onto normalized truth direction
    truth_coords = X_centered @ v_hat

    # 5. Vertical coordinate: remove truth component, then fit 1D PCA
    X_orthogonal = X_centered - np.outer(truth_coords, v_hat)
    pca = PCA(n_components=1, random_state=42)
    ortho_coords = pca.fit_transform(X_orthogonal).flatten()

    return truth_coords, ortho_coords, y_all


def plot_3x3_geometry(projections_dict, title, output_path):
    """
    Plots a 3x3 grid exactly matching Figure 5 in Poulis et al. (2024)
    Row 1: A1, A2, A3
    Row 2: F0, F1, F2
    Row 3: F3, F4, F5
    """
    grid_tasks = [
        ["A1", "A2", "A3"],
        ["F0", "F1", "F2"],
        ["F3", "F4", "F5"]
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))

    for row_idx in range(3):
        for col_idx in range(3):
            task = grid_tasks[row_idx][col_idx]
            ax = axes[row_idx, col_idx]

            if task in projections_dict:
                data = projections_dict[task]
                x = data["truth_axis"]
                y = data["ortho_axis"]
                labels = data["label"]

                # Color True as Blue (#1f77b4), False as Red (#d62728)
                colors = np.where(labels == 1, "#1f77b4", "#d62728")

                ax.scatter(x, y, c=colors, alpha=0.55, s=8, edgecolors="none")
                ax.set_title(task, fontsize=14, fontweight="bold")
                ax.grid(True, linestyle="--", alpha=0.35)

                if col_idx == 0 and row_idx == 1:
                    ax.set_ylabel("Orthogonal direction of max. variance", fontsize=13)
                if row_idx == 2 and col_idx == 1:
                    ax.set_xlabel("Truth direction", fontsize=13)
            else:
                ax.text(0.5, 0.5, f"No data for {task}", ha="center", va="center")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="False", markerfacecolor="#d62728", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="True", markerfacecolor="#1f77b4", markersize=8)
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01), fontsize=12)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def update_batch_size(pbar, current_batch_size, previous_rate):
    """Halve the batch size if throughput (rows/sec) has dropped by more than half since
    the last check -- a proxy for "sequences got a lot longer, back off." pbar.format_dict
    is tqdm's own snapshot of its stats; 'rate' is iterations/sec (None on the very first
    call, before tqdm has timing data yet)."""
    if current_batch_size == 1 or previous_rate is None:
        return current_batch_size, pbar.format_dict.get("rate")

    current_rate = pbar.format_dict.get("rate")
    if current_rate is not None and current_rate < previous_rate / 2:
        return max(1, current_batch_size // 2), current_rate

    return current_batch_size, previous_rate if previous_rate is not None else current_rate

def extract_activations(ids, model, layer, batch_size=4, desc="Batches"):
    ids = ids.apply(literal_eval)
    ids = [torch.tensor(id_list) for id_list in ids]
    lengths = torch.tensor([len(t) for t in ids])

    # Sort by length before batching so each batch is made of similar-length sequences --
    # without this, one unusually long statement in an otherwise-short batch forces all rows
    # in that batch to be padded up to its length, which can make that single batch
    # dramatically slower/more memory-hungry than the rest. Un-sort the results back to the
    # original row order at the end so they still line up with the caller's labels.
    sort_idx = torch.argsort(lengths)
    inverse_idx = torch.argsort(sort_idx)
    ids = [ids[i] for i in sort_idx.tolist()]
    lengths = lengths[sort_idx]

    final_token_activations = []

    previous_rate = None
    pbar = tqdm(total=len(ids), desc=desc, position=1, leave=False)
    i = 0
    n_batches = 0

    while i < len(ids):
        n_batches += 1
        if n_batches % 10 == 0:
            torch.cuda.empty_cache()
            batch_size, previous_rate = update_batch_size(pbar, batch_size, previous_rate)

        batch = ids[i : i + batch_size]
        batch_lengths = lengths[i : i + batch_size]

        max_len = int(batch_lengths.max())
        padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0, padding_side="left").to(model.device)

        # Mask built from real lengths, not from comparing token values against 0 -- so a
        # real token that happens to have id 0 is never mistaken for padding.
        position_idx = torch.arange(max_len).unsqueeze(0)
        attention_mask = (position_idx >= (max_len - batch_lengths).unsqueeze(1)).long().to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=padded, attention_mask=attention_mask, output_hidden_states=True)

        final_token_activation = outputs.hidden_states[layer][:, -1, :]
        final_token_activations.append(final_token_activation.cpu())

        pbar.update(len(batch))
        i += batch_size

    pbar.close()
    return torch.cat(final_token_activations, dim=0)[inverse_idx]

def generate_deepseek_geometry(full_model_name, dataset_path, layer_idx, output_path, cond_lbl):
    """
    Generates Figure 5 2D representation projections for DeepSeek R1 Distill 8B.
    """
    model_name = full_model_name.split("/")[1]
    model = AutoModelForCausalLM.from_pretrained(full_model_name, dtype=torch.float16, device_map="cuda")

    tasks = ["A3", "F5", "A1", "A2", "F0", "F1", "F2", "F3", "F4"]
    projections = {}

    # Load in data
    for task in tqdm(tasks, desc="Tasks", position=0):
        train_df = pd.read_csv(f"{dataset_path}/{task}_train.csv")[["extracted_statement_ids", "label"]]
        test_df = pd.read_csv(f"{dataset_path}/{task}_test.csv")[["extracted_statement_ids", "label"]]

        # Extract activations directly (not via a DataFrame column -- extract_activations
        # returns a genuine 2D tensor, and pandas rejects assigning a 2D array-like to a
        # single column).
        train_acts = extract_activations(train_df["extracted_statement_ids"], model, layer_idx, desc=f"{task} train").numpy()
        test_acts = extract_activations(test_df["extracted_statement_ids"], model, layer_idx, desc=f"{task} test").numpy()

        x, y, y_all = compute_task_projection(train_acts, np.array(train_df["label"]), test_acts, np.array(test_df["label"]))

        projections[task] = {
                    "truth_axis": x,
                    "ortho_axis": y,
                    "label": y_all
                }

    plot_3x3_geometry(
        projections,
        f"Figure 5: 2D Representation Geometry at Layer {layer_idx}\n({model_name} — {cond_lbl})",
        output_path
    )


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 5 2D Representation Geometry Plots")
    parser.add_argument("--layer", type=int, default=25, help="Layer index (default: 25)")
    parser.add_argument("--output_dir", default="experiments/figures/figure_5", help="Output folder")
    parser.add_argument("--dataset_path", required=True, help="Folder containing {task}_train.csv/{task}_test.csv, e.g. datasets/CoT_datasets/sentence_based_lexically_cleaned")
    parser.add_argument("--condition", required=True, help="Condition slug used in the output filename and the plot title's label, e.g. sentence-based-CoT, no-prompt-chat-template, filler-token-only")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Full HuggingFace model repo id, e.g. deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    slug = args.condition.replace("-", "_").replace(" ", "_")
    out_path = os.path.join(args.output_dir, f"figure_5_geometry_deepseek_{slug}_l{args.layer}.png")
    generate_deepseek_geometry(args.model, args.dataset_path, args.layer, out_path, args.condition)


if __name__ == "__main__":
    main()
