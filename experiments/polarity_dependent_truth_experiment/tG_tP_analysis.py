"""
F0/F1 Polarity Direction Analysis

Tests whether the F0<->F1 cross-task generalization failure (see
findings/f0_f1_polarity_investigation.txt) is explained by Burger, Hamprecht & Nadler's
t_G / t_P decomposition ("Truth is Universal: Robust Detection of Lies in LLMs", NeurIPS
2024, arXiv:2407.12831).

F0 and F1 are the only affirmative/negated-style pair in this project's task battery
(F0: country matches city, F1: country does NOT match city -- same underlying evidence,
inverted labeling convention), so they stand in for the paper's "topic pair" (one
affirmative dataset + its negated counterpart).

Per layer, this script:
  1. Trains a probe on F0 (w_F0) and a probe on F1 (w_F1), the normal way.
  2. Fits t_G (general truth direction) and t_P (polarity-sensitive direction) via the
     closed-form OLS solution from the paper's Eq. 3-5, using F0 (p=+1) and F1 (p=-1)
     pooled together.
  3. Orthonormalizes {t_G, t_P} into a basis {e1, e2} (matching the paper's own
     visualization basis, Appendix B) and decomposes each trained probe's *unit* direction
     onto that basis, reporting the t_G-component, the t_P-component, and the residual
     (out-of-subspace) magnitude.

Expected pattern if the t_G/t_P account explains the anomaly: both w_F0 and w_F1 should
carry a large relative t_P component (large |t_P-component| / |t_G-component| ratio) --
that's the component which flips sign across polarity and is the paper's proposed
mechanism for why a probe trained on one polarity fails to generalize to the other.
"""

from ast import literal_eval
import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def fit_linear_probe(X_train, y_train, device="cuda" if torch.cuda.is_available() else "cpu"):
    """No-bias, train-mean-centered linear probe -- matches the paper's Appendix A.4
    (Adam, lr=1e-3, weight_decay=0.1, 1000 steps, BCEWithLogitsLoss, no bias term). Same
    methodology used elsewhere in this project (plot_figure_5_geometry.py,
    Extract_Layers_*.ipynb's train_probe)."""
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


def extract_all_layer_activations(ids, model, batch_size=4, desc="Batches"):
    """Extracts the final-token activation at EVERY layer in one forward pass per batch
    (output_hidden_states=True), returning a list of [N, d] arrays, one per layer. Same
    length-sorted, left-padded batching as plot_figure_5_geometry.py's extract_activations
    -- avoids padding blowup from outlier-length sequences and un-sorts results back to the
    original row order at the end."""
    ids = ids.apply(literal_eval)
    ids = [torch.tensor(id_list) for id_list in ids]
    lengths = torch.tensor([len(t) for t in ids])

    sort_idx = torch.argsort(lengths)
    inverse_idx = torch.argsort(sort_idx)
    ids = [ids[i] for i in sort_idx.tolist()]
    lengths = lengths[sort_idx]

    num_layers = model.config.num_hidden_layers + 1
    per_layer_acts = [[] for _ in range(num_layers)]

    pbar = tqdm(total=len(ids), desc=desc, leave=False)
    i = 0
    while i < len(ids):
        batch = ids[i : i + batch_size]
        batch_lengths = lengths[i : i + batch_size]

        max_len = int(batch_lengths.max())
        padded = torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=0, padding_side="left"
        ).to(model.device)

        position_idx = torch.arange(max_len).unsqueeze(0)
        attention_mask = (position_idx >= (max_len - batch_lengths).unsqueeze(1)).long().to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=padded, attention_mask=attention_mask, output_hidden_states=True)

        for layer_idx, layer_hidden in enumerate(outputs.hidden_states):
            per_layer_acts[layer_idx].append(layer_hidden[:, -1, :].float().cpu())

        pbar.update(len(batch))
        i += batch_size
    pbar.close()

    # Cast to float32 on the way out -- the model runs in float16, but np.linalg.lstsq
    # (used in fit_tg_tp and decompose_probe) doesn't support float16 arrays.
    return [torch.cat(layer_acts, dim=0)[inverse_idx].numpy() for layer_acts in per_layer_acts]


def fit_tg_tp(X_f0, y_f0, X_f1, y_f1):
    """Closed-form OLS fit of t_G and t_P (Burger et al. 2024, Eq. 3-5):

        a_ij ~= mu + tau_ij * t_G + tau_ij * p_i * t_P

    where tau_ij = +1/-1 for true/false statements and p_i = +1 for F0 (the affirmative
    member of the pair) / -1 for F1 (the negated member). F0+F1 pooled together form the
    single topic pair available in this project's task battery. mu collapses to a single
    intercept term since there is only one topic. Solved directly via lstsq -- no iterative
    training needed, this is ordinary least squares.
    """
    tau_f0 = np.where(y_f0 == 1, 1.0, -1.0)
    tau_f1 = np.where(y_f1 == 1, 1.0, -1.0)
    p_f0 = np.ones_like(tau_f0)
    p_f1 = -np.ones_like(tau_f1)

    X = np.vstack([X_f0, X_f1])
    tau = np.concatenate([tau_f0, tau_f1])
    p = np.concatenate([p_f0, p_f1])

    design = np.column_stack([np.ones_like(tau), tau, tau * p])  # [N, 3]
    coeffs, _, _, _ = np.linalg.lstsq(design, X, rcond=None)     # [3, d]

    mu, tG, tP = coeffs[0], coeffs[1], coeffs[2]
    return tG, tP, mu


def decompose_probe(w, tG, tP):
    """Decomposes a trained probe's unit direction w_hat as a linear combination of the
    ACTUAL fitted t_G and t_P directions (unit-normalized, but not orthogonalized against
    each other): solves for (a, b) minimizing ||w_hat - a*tG_hat - b*tP_hat|| via ordinary
    least squares on the [tG_hat, tP_hat] design.

    This is deliberately NOT a Gram-Schmidt/orthonormal decomposition. Gram-Schmidt would
    redefine the "t_P axis" as t_P with its t_G-component subtracted out -- i.e. it
    measures alignment with a modified direction, not the real fitted t_P. Since t_G and
    t_P are not constrained to be orthogonal by the OLS fit in fit_tg_tp, that substitution
    can understate/distort how much of w_hat is actually explained by t_P whenever t_G and
    t_P are correlated. Solving the oblique (non-orthogonal) least-squares system instead
    gives the coefficients that best reconstruct w_hat using the real t_G and t_P
    directions as-is, correctly accounting for their correlation rather than ignoring it.
    """
    w_hat = w / (np.linalg.norm(w) + 1e-12)
    tG_hat = tG / np.linalg.norm(tG)
    tP_hat = tP / np.linalg.norm(tP)

    basis = np.column_stack([tG_hat, tP_hat])          # [d, 2], NOT orthonormalized
    (a, b), _, _, _ = np.linalg.lstsq(basis, w_hat, rcond=None)
    residual = float(np.linalg.norm(w_hat - a * tG_hat - b * tP_hat))

    return {
        "tG_component": float(a),
        "tP_component": float(b),
        "residual": residual,
        "tG_tP_cosine": float(tG_hat @ tP_hat),  # how non-orthogonal tG/tP actually are
        "tP_over_tG_ratio": abs(b) / (abs(a) + 1e-12),
    }


def load_cross_task_auroc(csv_path, condition, model_name):
    """Pulls the already-computed F0->F1 and F1->F0 cross-task AUROC per layer out of
    results_database.csv, if present, purely for context/overlay on the final plot --
    no recomputation needed since these numbers already exist from the extraction
    notebooks."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df = df[(df["train_condition"] == condition) & (df["test_condition"] == condition) & (df["model"] == model_name)]

    f0_to_f1 = df[(df["train_task"] == "F0") & (df["test_task"] == "F1")].set_index("layer")["auroc"]
    f1_to_f0 = df[(df["train_task"] == "F1") & (df["test_task"] == "F0")].set_index("layer")["auroc"]

    if f0_to_f1.empty and f1_to_f0.empty:
        return None
    return f0_to_f1, f1_to_f0


def run_analysis(model_name, dataset_path, condition, output_dir, results_csv, results_db_model_name=None):
    short_model_name = model_name.split("/")[1]
    # results_database.csv stores its own short model label (e.g. "deepseek-r1-distill-8b"),
    # which isn't guaranteed to match model_name.split("/")[1] verbatim -- pass
    # results_db_model_name explicitly when they differ (used only for the AUROC overlay).
    auroc_model_name = results_db_model_name or short_model_name
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="cuda")
    model.eval()

    print(f"Loading F0/F1 datasets from {dataset_path}...")
    f0_train = pd.read_csv(f"{dataset_path}/F0_train.csv")[["extracted_statement_ids", "label"]]
    f1_train = pd.read_csv(f"{dataset_path}/F1_train.csv")[["extracted_statement_ids", "label"]]

    print("Extracting F0 activations (all layers)...")
    f0_acts = extract_all_layer_activations(f0_train["extracted_statement_ids"], model, desc="F0 train")
    print("Extracting F1 activations (all layers)...")
    f1_acts = extract_all_layer_activations(f1_train["extracted_statement_ids"], model, desc="F1 train")

    y_f0 = f0_train["label"].to_numpy()
    y_f1 = f1_train["label"].to_numpy()

    num_layers = len(f0_acts)
    rows = []

    for layer_idx in tqdm(range(num_layers), desc="Layers"):
        X_f0 = f0_acts[layer_idx]
        X_f1 = f1_acts[layer_idx]

        w_f0, _ = fit_linear_probe(X_f0, y_f0)
        w_f1, _ = fit_linear_probe(X_f1, y_f1)

        tG, tP, _ = fit_tg_tp(X_f0, y_f0, X_f1, y_f1)

        decomp_f0 = decompose_probe(w_f0, tG, tP)
        decomp_f1 = decompose_probe(w_f1, tG, tP)

        rows.append({"layer": layer_idx, "probe_source": "F0", **decomp_f0})
        rows.append({"layer": layer_idx, "probe_source": "F1", **decomp_f1})

    results_df = pd.DataFrame(rows)
    out_csv = os.path.join(output_dir, f"f0_f1_polarity_decomposition_{short_model_name}_{condition}.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    plot_results(results_df, load_cross_task_auroc(results_csv, condition, auroc_model_name),
                 short_model_name, condition, output_dir)


def plot_results(results_df, cross_task_auroc, short_model_name, condition, output_dir):
    fig, ax1 = plt.subplots(figsize=(11, 6))

    for probe_source, color in [("F0", "#1f77b4"), ("F1", "#d62728")]:
        subset = results_df[results_df["probe_source"] == probe_source]
        ax1.plot(subset["layer"], subset["tP_over_tG_ratio"], marker="o", markersize=3,
                  color=color, label=f"|t_P component| / |t_G component| -- {probe_source}-trained probe")

    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("|t_P component| / |t_G component| (in trained probe)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.35)

    lines1, labels1 = ax1.get_legend_handles_labels()

    if cross_task_auroc is not None:
        f0_to_f1, f1_to_f0 = cross_task_auroc
        ax2 = ax1.twinx()
        if not f0_to_f1.empty:
            ax2.plot(f0_to_f1.index, f0_to_f1.values, linestyle="--", color="#1f77b4", alpha=0.5,
                      label="Cross-task AUROC: F0(train)->F1(test)")
        if not f1_to_f0.empty:
            ax2.plot(f1_to_f0.index, f1_to_f0.values, linestyle="--", color="#d62728", alpha=0.5,
                      label="Cross-task AUROC: F1(train)->F0(test)")
        ax2.set_ylabel("Cross-task AUROC (existing results_database.csv)", fontsize=12)
        ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines1, labels1 = lines1 + lines2, labels1 + labels2

    ax1.legend(lines1, labels1, loc="upper left", fontsize=9)
    fig.suptitle(f"F0/F1 Probe Decomposition onto t_G/t_P Basis\n({short_model_name} — {condition})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"f0_f1_polarity_decomposition_{short_model_name}_{condition}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="F0/F1 Polarity (t_G/t_P) Direction Decomposition Analysis")
    parser.add_argument("--dataset_path", required=True,
                         help="Folder containing F0_train.csv/F0_test.csv/F1_train.csv/F1_test.csv, "
                              "e.g. datasets/CoT_datasets/sentence_based_lexically_cleaned")
    parser.add_argument("--condition", required=True,
                         help="Condition slug used in output filenames/title, e.g. sentence-based-CoT, no-prompt")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                         help="Full HuggingFace model repo id")
    parser.add_argument("--output_dir", default="experiments/figures/polarity_analysis")
    parser.add_argument("--results_csv", default="experiments/results_database.csv",
                         help="Existing results_database.csv, used only to overlay already-computed "
                              "cross-task AUROC for context (optional -- skipped if not found)")
    parser.add_argument("--results_db_model_name", default=None,
                         help="Model label as stored in results_database.csv's 'model' column, if it "
                              "differs from --model's short name (e.g. 'deepseek-r1-distill-8b' vs. "
                              "'DeepSeek-R1-Distill-Llama-8B'). Defaults to --model's short name.")
    args = parser.parse_args()

    run_analysis(args.model, args.dataset_path, args.condition, args.output_dir, args.results_csv,
                 args.results_db_model_name)


if __name__ == "__main__":
    main()
