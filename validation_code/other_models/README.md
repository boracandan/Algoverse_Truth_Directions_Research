# Other Model Experiments & Evaluations

This directory organizes all comparative model truth-probing experiments evaluated alongside the primary baseline (`meta-llama/Llama-3.1-8B-Instruct`).

---

## 📁 Directory Overview

| File | Target Architecture | Scale / Precision | Description |
| :--- | :--- | :--- | :--- |
| **`layerwise_truth_probing_gemma4.ipynb`** | `google/gemma-4-e2b-it` | 2B / bfloat16 | Full 36-layer truth probing sweep across factual (F0–F5) and arithmetic (A1–A3) tasks on Google's edge model. |
| **`layerwise_truth_probing_llama3_2_3b.ipynb`** | `meta-llama/Llama-3.2-3B-Instruct` | 3B / bfloat16 | 28-layer sweep evaluating compact LLaMA 3.2 against 8B factual accuracy and arithmetic ceilings. |
| **`layerwise_truth_probing_granite8b.ipynb`** | `ibm-granite/granite-3.1-8b-instruct` | 8B / bfloat16 | 40-layer sweep evaluating IBM Granite 3.1 8B, testing compositional reasoning (F4, F5) and A3 arithmetic representations. |
| **`validation_all_models.ipynb`** | Multi-Model Parameterized | Variable | Parameterized notebook for middle-layer validation across candidate models. |

---

## 💾 Saved Layer Results (Pickle Artifacts)

- **`layer_results_gemma4.pkl`**: Layerwise in-domain AUROC scores across all 36 layers for Gemma 4 Edge 2B.
- **`layer_results_llama3_2_3b.pkl`**: Layerwise in-domain AUROC scores across all 28 layers for LLaMA 3.2 3B.
- **`layer_results_granite8b.pkl`**: Layerwise in-domain AUROC scores across all 40 layers for IBM Granite 3.1 8B.

*(Note: Baseline LLaMA 3.1 8B results remain in `../layer_results.pkl`).*
