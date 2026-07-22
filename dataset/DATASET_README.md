# Truth Directions Dataset (Poulis et al. + CoT Extension)

## Overview
This dataset implements the factual and arithmetic reasoning tasks from **Poulis et al. (2024) "Testing the Limits of Truth Directions in LLMs"**, extended by complexity levels F0–F5 (factual) and A1–A3 (arithmetic).

The dataset is designed for probing linear representations of truth in language models across varying task complexity, serving as the foundation for studying how chain-of-thought reasoning affects truth direction geometry.

---

## Task Hierarchy

### Factual Family (F0–F5)
Statements about geographic facts with increasing compositional complexity.

| Task | Complexity | Examples | Train | Test | Description |
|------|-----------|----------|-------|------|-------------|
| **F0** | Atomic facts | 160 | 112 | 48 | Single city-country facts: "Paris is in France" |
| **F1** | Negation | 160 | 112 | 48 | Negated facts: "Paris is not in [wrong country]" |
| **F2** | Conjunction | 80 | 56 | 24 | Two-clause conjunction: "Both [fact1] and [fact2]" |
| **F3** | Counting (2 items) | 2,000 | 1,400 | 600 | "Exactly N of {city1, city2} are in [country]" |
| **F4** | Counting (5 items) | 2,000 | 1,400 | 600 | "Exactly N of {5 cities} are in [country]" |
| **F5** | Dual counting (6 items) | 2,000 | 1,400 | 600 | "Exactly M in [country1] and N in [country2]" |

### Arithmetic Family (A1–A3)
Equations with increasing operation depth.

| Task | Complexity | Examples | Train | Test | Description |
|------|-----------|----------|-------|------|-------------|
| **A1** | Single operation | 1,000 | 700 | 300 | `a ± b = result` or `a × b = result` |
| **A2** | Two operations | 1,000 | 700 | 300 | `(a ⊕ b) ⊗ c = result` with mixed ±/×/ |
| **A3** | Three operations | 1,000 | 700 | 300 | `(a ⊕ b) ⊗ (c ⊘ d) = result` |

---

## Dataset Properties

### Label Distribution
All datasets are **balanced** (50% true / 50% false) within train and test splits.

### City Coverage
- **80 cities** across 10 countries (8 cities per country)
- Countries: France, Germany, United States, Japan, Brazil, India, China, United Kingdom, Australia, Greece

### Arithmetic Details
- Operands: integers [1, 99]
- Operations: +, −, ×, / (division restricted to exact integer results)
- Wrong answers: perturbed by random offset in {−5, ..., −1, 1, ..., 5}
- All results are integers

### Train/Test Split
- **F0–F2**: ~70% train / ~30% test
- **F3–F5**: ~70% train / ~30% test (1,400 / 600)
- **A1–A3**: ~70% train / ~30% test (700 / 300)

---

## File Format

Each CSV file contains two columns:

```csv
statement,label
"The city of Paris is in France.",True
"The city of Berlin is in France.",False
```

**Columns:**
- `statement` (string): The complete statement to evaluate
- `label` (boolean): Ground truth (True/False)

---

## Usage for CoT Experiments

### Baseline (Plain Statement)
Use the `statement` field directly:
```
Input: "The city of Paris is in France."
Model output (ask): "Yes" or "No"
Probe: activation at final input token
```

### Ask-Correct (Poulis et al.)
Wrap with explicit instruction:
```
Input: "The city of Paris is in France. Is this statement correct? Yes or No."
Model output: "Yes" or "No"
Probe: activation at the output token
```

### CoT (Proposed)
#### Zero-shot CoT
```
Input: "The city of Paris is in France. Let's think step by step. Answer with only 'Yes' or 'No'."
Model output: "[reasoning] Yes"
Probe: activation at token before "Yes", OR mean-pooled over all reasoning tokens
```

#### Few-shot CoT
```
Input: "[Example 1 with step-by-step reasoning]
[Example 2 with step-by-step reasoning]
[Example 3 with step-by-step reasoning]
Now, the city of Paris is in France. Let's think step by step. Answer with only 'Yes' or 'No'."
Model output: "[reasoning] Yes"
Probe: activation at token before "Yes", OR mean-pooled over all reasoning tokens
```

---

## Quality Notes

### F3–F5
- False examples use varied wrong counts, not always ±1
- Distractors (cities not in target country) are randomly sampled

### A1–A3
- Integer division constraint eliminates non-exact results
- Wrong answer offset is randomized to avoid trivial patterns
- Negative results are allowed (e.g., "3 - 19 = -16")

### Known Limitations
- **F0–F2 sample sizes** are smaller (~160 examples per task) due to limited city data; extend if needed by adding more cities/countries
- **A3 generation** filters for valid integer division; if error rates >30%, consider larger operand ranges

---

## Replication & Extension

To extend the dataset:
1. Add more cities per country in `cities_by_country`
2. Increase loop iterations in F3/F4/F5/A1/A2/A3 generation
3. Adjust operand ranges in arithmetic tasks
4. Maintain balanced train/test splits via the `split_dataset()` function

---

## References

- **Marks & Tegmark (2024)**: "The Geometry of Truth" — established baseline truth direction probes
- **Poulis et al. (2024)**: "Testing the Limits of Truth Directions in LLMs" — complexity hierarchy and generalization failures
- **Ying et al. (2024)**: "The Truthfulness Spectrum Hypothesis" — mean-pooling vs. last-token extraction for reasoning

---


