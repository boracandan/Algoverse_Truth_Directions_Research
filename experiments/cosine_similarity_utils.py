"""
Utilities for comparing saved probe weight vectors via cosine similarity, and for computing
those weight vectors in the first place when no notebook run has saved them yet.

Each of the layer_extraction_notebooks/Extract_Layers_Deepseek_*.ipynb notebooks calls
write_probe_weights(...) for every (task, layer) it trains a probe on, appending to a single
pickle file (weights_path, e.g. "{save_dir}/probe_weights.pkl") keyed by
(model, condition, task, layer) -> weight vector (np.ndarray, shape (hidden_dim,)).

The comparison half (load_weights/cosine_similarity/etc.) is CPU-only and needs nothing but
numpy/pandas. The compute half (compute_and_save_weights and its helpers) needs torch +
transformers and a GPU to be worth running -- those imports are deferred into the functions
that need them so importing this module for pure comparison work doesn't require a GPU
environment at all.
"""

import os
import pickle

import numpy as np
import pandas as pd

# condition -> (folder relative to a datasets/ root, text column to feed the tokenizer,
# whether that column holds a token-id list needing tokenizer.decode() first). Mirrors the
# DATA_DIR / text-source choice made in each layer_extraction_notebooks/*.ipynb file.
CONDITIONS = {
    "no-prompt": ("plain_dataset", "statement", False),
    "no-prompt-chat-template": ("plain_template_dataset", "extracted_statement_texts", False),
    "cot-zero-shot": (os.path.join("CoT_datasets", "lexically_cleaned"), "extracted_statement_ids", True),
    "sentence-based-CoT": (os.path.join("CoT_datasets", "sentence_based_lexically_cleaned"), "extracted_statement_ids", True),
}

DEFAULT_TASKS = ["A1", "A2", "A3", "F0", "F1", "F2", "F3", "F4", "F5"]


def load_weights(weights_path):
    """Loads the probe-weights pickle into a dict keyed by (model, condition, task, layer)."""
    with open(weights_path, "rb") as f:
        return pickle.load(f)


def cosine_similarity(vec_a, vec_b):
    """Cosine similarity between two weight vectors, in [-1, 1]."""
    a, b = np.asarray(vec_a), np.asarray(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 1e-12 else float("nan")


def get_similarity(weights_db, key_a, key_b):
    """Cosine similarity between the two probe directions stored under key_a and key_b,
    each a (model, condition, task, layer) tuple."""
    return cosine_similarity(weights_db[key_a], weights_db[key_b])


def build_similarity_table(weights_db, model, condition, layer, task_order):
    """Task x task cosine-similarity matrix for one (model, condition, layer), as a
    labeled DataFrame -- mirrors build_matrix() in plotting/plot_figure_4_generalization.py
    so it drops into the same kind of heatmap plotting code."""
    n = len(task_order)
    matrix = np.full((n, n), np.nan)

    for i, task_a in enumerate(task_order):
        key_a = (model, condition, task_a, layer)
        if key_a not in weights_db:
            continue
        for j, task_b in enumerate(task_order):
            key_b = (model, condition, task_b, layer)
            if key_b not in weights_db:
                continue
            matrix[i, j] = cosine_similarity(weights_db[key_a], weights_db[key_b])

    return pd.DataFrame(matrix, index=task_order, columns=task_order)


def _vec_to_str(vec):
    """Serializes a weight vector to a single space-separated string so it fits in one CSV
    cell. Round-trip with np.fromstring(s, sep=' ')."""
    return " ".join(map(str, np.asarray(vec).tolist()))


def append_weight_vectors(csv_path, weights_db, model, task_order=None):
    """Adds weight_i/weight_j columns onto an EXISTING cosine-similarity CSV (e.g.
    cosine_similarity_mahalanobis.csv) in place, rather than writing a separate file.

    Reads csv_path (expects condition,layer,task_i,task_j,cosine_similarity columns),
    looks up the probe weight vector for each (condition, layer, task_i/task_j) row in
    weights_db, and writes the same rows back out to csv_path with weight_i/weight_j
    appended (space-separated, see _vec_to_str). Rows whose vectors aren't found in
    weights_db get empty weight_i/weight_j cells rather than being dropped.
    """
    df = pd.read_csv(csv_path)
    task_order = task_order or DEFAULT_TASKS

    def _lookup(row, task_col):
        key = (model, row["condition"], row[task_col], row["layer"])
        vec = weights_db.get(key)
        return _vec_to_str(vec) if vec is not None else ""

    df["weight_i"] = df.apply(lambda row: _lookup(row, "task_i"), axis=1)
    df["weight_j"] = df.apply(lambda row: _lookup(row, "task_j"), axis=1)

    df.to_csv(csv_path, index=False)
    missing = int((df["weight_i"] == "").sum())
    print(f"✓ Updated {csv_path} with weight vectors for {len(df) - missing}/{len(df)} rows")
    return df


def cross_condition_similarity(weights_db, model, task, layer, condition_a, condition_b):
    """Cosine similarity between the same task's probe direction under two different
    conditions -- e.g. does the no-prompt F0 probe point the same way as the cot-zero-shot
    F0 probe, at a given layer?"""
    key_a = (model, condition_a, task, layer)
    key_b = (model, condition_b, task, layer)
    return cosine_similarity(weights_db[key_a], weights_db[key_b])


def write_probe_weights(task_name, layer_results, weights_path, condition, model_name):
    """Same helper the notebooks call -- appends each layer's weight vector into the pickle
    at weights_path, keyed by (model_name, condition, task_name, layer). Kept here too so
    compute_and_save_weights doesn't depend on notebook-only state."""
    if os.path.exists(weights_path):
        with open(weights_path, "rb") as f:
            weights_db = pickle.load(f)
    else:
        weights_db = {}

    for layer_idx, data in layer_results.items():
        weights_db[(model_name, condition, task_name, layer_idx)] = data["weights"]

    with open(weights_path, "wb") as f:
        pickle.dump(weights_db, f)
    print(f"  ✓ Saved {len(layer_results)} probe weight vectors for {task_name} ({condition})")


def _decode_ids(id_column, tokenizer):
    from ast import literal_eval
    return [
        tokenizer.decode(literal_eval(ids) if isinstance(ids, str) else ids)
        for ids in id_column
    ]


def _activations_all_layers(model, tokenizer, statements, batch_size=4):
    import torch
    statements = list(statements)
    num_layers = model.config.num_hidden_layers + 1
    activations_by_layer = [[] for _ in range(num_layers)]

    for i in range(0, len(statements), batch_size):
        batch = statements[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for layer_idx, layer_hidden in enumerate(outputs.hidden_states):
            activations_by_layer[layer_idx].append(layer_hidden[:, -1, :].cpu())

        del outputs, inputs
        torch.cuda.empty_cache()

    return [torch.cat(layer_acts, dim=0) for layer_acts in activations_by_layer]


def _train_probe(X_train, X_test, y_train, y_test, device):
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    X_train = torch.stack(list(X_train)).float().numpy()
    X_test = torch.stack(list(X_test)).float().numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    train_mean = X_train.mean(axis=0)
    X_train_c, X_test_c = X_train - train_mean, X_test - train_mean

    X_train_t = torch.tensor(X_train_c, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test_c, dtype=torch.float32, device=device)

    probe = nn.Linear(X_train.shape[1], 1, bias=False).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=0.1)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(1000):
        optimizer.zero_grad()
        loss = loss_fn(probe(X_train_t).squeeze(-1), y_train_t)
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        test_logits = probe(X_test_t).squeeze(-1).cpu().numpy()

    auroc = roc_auc_score(y_test, test_logits)
    w = probe.weight.detach().cpu().numpy().flatten()
    return {"weights": w, "train_mean": train_mean, "auroc": auroc}


def compute_and_save_weights(model_name, weights_path, datasets_root="datasets",
                              conditions=None, tasks=None, batch_size=4,
                              short_model_name=None, device=None):
    """Loads model_name ONCE, then for every condition in `conditions` (default: all of
    CONDITIONS) extracts activations and retrains a probe per task/layer, saving each
    resulting weight vector via write_probe_weights. This exists for the case where no
    layer_extraction_notebooks/*.ipynb run is still live to pull already-trained weights
    from -- it has to redo activation extraction (the expensive part; nothing about it was
    ever cached to disk), but loading the model once and sweeping every condition in one
    pass is still cheaper than the 4 separate model loads the 4 notebooks each do.

    Probes are retrained here (not reloaded from anywhere), using the exact same
    architecture/hyperparameters as the notebooks (no-bias Linear, train-mean centering,
    Adam lr=1e-3, weight_decay=0.1, 1000 steps, BCEWithLogitsLoss) -- but with no random
    seed fixed anywhere in that pipeline (here or in the notebooks), so the resulting AUROC
    will be very close to, but not bit-identical to, what's already in results_database.csv.
    Only weight vectors are written here; results_database.csv is left untouched.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tasks = tasks or DEFAULT_TASKS
    conditions = conditions or list(CONDITIONS.keys())
    short_model_name = short_model_name or model_name.split("/")[-1]

    print(f"Loading tokenizer + model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    print(f"✓ Model loaded on {model.device}")

    for condition in conditions:
        if condition not in CONDITIONS:
            raise KeyError(f"Unknown condition {condition!r}; add it to CONDITIONS or pass "
                            f"data_dir/text_column/decode_ids yourself and call the pieces directly.")
        rel_dir, text_column, needs_decode = CONDITIONS[condition]
        data_dir = os.path.join(datasets_root, rel_dir)
        print(f"\n=== {condition} ({data_dir}) ===")

        for task in tasks:
            print(f"  {task}...")
            train_df = pd.read_csv(os.path.join(data_dir, f"{task}_train.csv"))
            test_df = pd.read_csv(os.path.join(data_dir, f"{task}_test.csv"))

            if needs_decode:
                train_text = _decode_ids(train_df[text_column], tokenizer)
                test_text = _decode_ids(test_df[text_column], tokenizer)
            else:
                train_text = train_df[text_column]
                test_text = test_df[text_column]

            train_acts = _activations_all_layers(model, tokenizer, train_text, batch_size)
            test_acts = _activations_all_layers(model, tokenizer, test_text, batch_size)

            layer_results = {}
            for layer_idx in range(len(train_acts)):
                layer_results[layer_idx] = _train_probe(
                    train_acts[layer_idx], test_acts[layer_idx],
                    train_df["label"], test_df["label"], device,
                )

            write_probe_weights(task, layer_results, weights_path, condition, short_model_name)

            del train_acts, test_acts
            torch.cuda.empty_cache()

    print("\n✓ All conditions/tasks complete.")
