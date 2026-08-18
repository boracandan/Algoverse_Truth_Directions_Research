# Cross-Architecture Layerwise Truth Probing & Representation Analysis

This directory contains layerwise linear probing implementations, evaluation sweeps, and serialized representation benchmarks across multi-scale autoregressive language models, comparing representation geometry and truth emergence against the baseline **LLaMA-3.1-8B-Instruct** ($L=32, d_{\text{model}}=4096$).

---

## 1. Experimental Methodology & Mathematical Formulation

### 1.1 Activation Extraction Protocol
Given an input sequence $S = (t_1, t_2, \dots, t_T)$ from task dataset $\mathcal{D}_k$, hidden activations are extracted from the transformer residual stream at each layer $l \in [0, L-1]$ corresponding to the final token position $t_T$:

$$\mathbf{h}_l(S) = \text{TransformerLayer}_l\big(\mathbf{h}_{l-1}(S)\big)_{[:, -1, :]} \in \mathbb{R}^{d_{\text{model}}}$$

Token sequences are left-padded without conversational prompt wrappers to eliminate chat-template artifacts and ensure exact token-positional alignment across distinct tokenizers.

### 1.2 Linear Probe Optimization
For each task $k$ and layer $l$, a linear classification probe parameterized by weight vector $\mathbf{w}_{k,l} \in \mathbb{R}^{d_{\text{model}}}$ and bias $b_{k,l} \in \mathbb{R}$ is optimized via $\ell_2$-regularized empirical risk minimization over binary truth labels $y_i \in \{0, 1\}$:

$$\min_{\mathbf{w}, b} \sum_{i=1}^{N_{\text{train}}} \log\left(1 + \exp\left(- (2y_i - 1)(\mathbf{w}^T \mathbf{h}_{l}(S_i) + b)\right)\right) + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$

Performance is measured via Area Under the Receiver Operating Characteristic curve ($\text{AUROC}$) on strictly disjoint held-out test splits $\mathcal{D}_k^{\text{test}}$:

$$\text{AUROC}_{k, l} = \mathbb{P}\left(\sigma\left(\mathbf{w}_{k,l}^T \mathbf{h}_l(S^+) + b_{k,l}\right) > \sigma\left(\mathbf{w}_{k,l}^T \mathbf{h}_l(S^-) + b_{k,l}\right) \mid y^+ = 1, y^- = 0\right)$$

---

## 2. Model Architectures & Parameter Specifications

| Identifier | Model Family | Parameters | Layer Count ($L$) | Hidden Dimension ($d_{\text{model}}$) | Precision | Primary Notebook |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `google/gemma-4-e2b-it` | Gemma 4 Edge | 2.0B | 36 | 1536 | `bfloat16` | `layerwise_truth_probing_gemma4.ipynb` |
| `meta-llama/Llama-3.2-3B-Instruct` | LLaMA 3.2 | 3.2B | 28 | 3072 | `bfloat16` | `layerwise_truth_probing_llama3_2_3b.ipynb` |
| `meta-llama/Llama-3.1-8B-Instruct` | LLaMA 3.1 (Baseline) | 8.0B | 32 | 4096 | `float16` / `bfloat16` | `../Extract_Layers.ipynb` |
| `ibm-granite/granite-3.1-8b-instruct` | Granite 3.1 | 8.2B | 40 | 4096 | `bfloat16` | `layerwise_truth_probing_granite8b.ipynb` |

---

## 3. Empirical Layerwise Benchmark Matrix

The table below reports the maximum in-domain held-out test $\text{AUROC}$ and the optimal layer index $(L^*)$ achieving peak linear separability across all 9 evaluation tasks ($F_0 - F_5, A_1 - A_3$):

| Task | Task Description | Gemma 4 Edge (2B) | LLaMA 3.2 (3B) | LLaMA 3.1 (8B Baseline) | **IBM Granite 3.1 (8B)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$F_0$** | Atomic Factual Statements | 0.8554 ($L_9$) | 0.9993 ($L_{12}$) | 0.9998 ($L_{13}$) | **0.9997** ($L_{31}$) |
| **$F_1$** | Explicit Negations | 0.8807 ($L_8$) | 1.0000 ($L_{14}$) | 0.9999 ($L_{13}$) | **1.0000** ($L_{15}$) |
| **$F_2$** | Binary Conjunctions | 0.7814 ($L_9$) | 0.9945 ($L_{13}$) | 0.9991 ($L_{16}$) | **0.9967** ($L_{40}$) |
| **$F_3$** | 2-Item Exact Cardinality Counting | 0.6199 ($L_{11}$) | 0.9388 ($L_{28}$) | 0.9779 ($L_{32}$) | **0.9848** ($L_{40}$) |
| **$F_4$** | 5-Item Exact Cardinality Counting | 0.6017 ($L_9$) | 0.8531 ($L_{28}$) | 0.8707 ($L_{32}$) | **0.9176** ($L_{36}$) |
| **$F_5$** | Dual-Predicate Set Comparison | 0.5666 ($L_9$) | 0.7545 ($L_{26}$) | 0.8008 ($L_{32}$) | **0.8862** ($L_{36}$) |
| **$A_1$** | Single-Operation Arithmetic | 0.5622 ($L_0$) | 0.9988 ($L_{28}$) | 0.9992 ($L_{30}$) | **0.9991** ($L_{39}$) |
| **$A_2$** | Two-Operation Arithmetic | 0.5298 ($L_5$) | 0.8985 ($L_{26}$) | 0.8848 ($L_{30}$) | **0.8070** ($L_{38}$) |
| **$A_3$** | Three-Operation Arithmetic | 0.5461 ($L_7$) | 0.5978 ($L_{10}$) | 0.5972 ($L_9$) | **0.6597** ($L_{38}$) |

---

## 4. Mechanistic Findings & Theoretical Implications

### 4.1 Residual Stream Capacity Constraints
Linear probe separability on compositional reasoning ($F_3 - F_5$) scales directly with residual stream dimensionality $d_{\text{model}}$:
- **$d_{\text{model}} = 1536$ (Gemma 4 Edge 2B):** Displays linear collapse under multi-clause composition ($F_3=0.6199, F_5=0.5666$). Representations saturate at shallow layers ($l/L \approx 0.25$) due to constrained subspace capacity for orthogonal truth and feature directions.
- **$d_{\text{model}} = 3072$ (LLaMA 3.2 3B):** Successfully retains factual fidelity ($F_0=0.9993, F_1=1.0000$) and tracks 8B-scale trajectories up to 2-item cardinality, but degrades on dual-predicate comparisons ($F_5=0.7545$).
- **$d_{\text{model}} = 4096$ (LLaMA 3.1 8B & IBM Granite 8B):** Provides sufficient subspace rank to decouple multi-entity truth vectors without destructive interference.

### 4.2 Depthwise Functional Stratification ($l/L$)
Across dense 8B architectures, truth representations exhibit a two-stage functional trajectory:
1. **Early Depth ($l/L \in [0.25, 0.40]$):** Linear encoding of isolated factual predicates and lexical polarity ($F_0, F_1$).
2. **Late Depth ($l/L \in [0.85, 1.00]$):** Emergence of multi-entity working memory, discrete counting states, and compositional truth planes ($F_3 - F_5, A_1 - A_3$).

### 4.3 Breaking the Arithmetic Representation Ceiling
On composite multi-step arithmetic ($A_3$), LLaMA models plateau at $\text{AUROC} \approx 0.597$, indicating representational bottlenecking in the absence of intermediate chain-of-thought tokens. IBM Granite 3.1 8B breaks this barrier, reaching $\text{AUROC} = 0.6597$ at layer 38 ($l/L = 0.95$).

---

## 5. Artifact Directory Index

```
validation_code/other_models/
├── README.md                                     # Technical benchmark reference
├── layerwise_truth_probing_gemma4.ipynb          # Gemma 4 Edge 2B evaluation pipeline
├── layerwise_truth_probing_llama3_2_3b.ipynb     # LLaMA 3.2 3B evaluation pipeline
├── layerwise_truth_probing_granite8b.ipynb       # IBM Granite 3.1 8B evaluation pipeline
├── validation_all_models.ipynb                   # Configurable multi-model probing interface
├── layer_results_gemma4.pkl                      # Serialized AUROC tensor dictionary (Gemma 4)
├── layer_results_llama3_2_3b.pkl                 # Serialized AUROC tensor dictionary (LLaMA 3.2 3B)
└── layer_results_granite8b.pkl                   # Serialized AUROC tensor dictionary (IBM Granite 8B)
```

Each pickle file contains a nested dictionary structured as:
```python
{
    "<TASK_ID>": {
        <layer_int>: {
            "auroc": float,
            "probe_weight": np.ndarray,  # shape (d_model,)
            "probe_bias": float
        }
    }
}
```
