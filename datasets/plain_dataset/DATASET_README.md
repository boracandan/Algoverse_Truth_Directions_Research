# Truth Directions Dataset (Poulis et al. Reconstruction + Extensions)

## Overview
This dataset reconstructs the factual and arithmetic reasoning tasks from **Poulis et al. (2026) "Testing the Limits of Truth Directions in LLMs"**, spanning complexity levels F0–F5 (factual) and A1–A3 (arithmetic).

The factual tasks are built from a fresh GeoNames pull (`cities1000.txt` + `CountryInfo.txt`), not the original snapshot Poulis used, so exact counts differ from the source paper. Several methodological choices also deliberately diverge from Poulis's own generation code, documented below and justified empirically in `validation_code/check_trivial_cases_all_or_none_correct.ipynb`.

---

## Task Hierarchy

### Factual Family (F0–F5)
Statements about geographic facts with increasing compositional complexity.

| Task | Complexity | Total | Train | Test | Description |
|------|-----------|-------|-------|------|-------------|
| **F0** | Atomic facts | 1,706 | 1,194 | 512 | Single city-country facts: "The city of Paris is in France." |
| **F1** | Negation | 1,706 | 1,194 | 512 | Negated variants of F0: "The city of Paris is not in France." |
| **F2** | Conjunction | 1,706 | 1,194 | 512 | Two-clause conjunction of independent F0 statements |
| **F3** | Counting (2 items) | 1,998 | 1,398 | 600 | "Exactly N of the following cities are in [country]: A, B." |
| **F4** | Counting (5 items) | 1,992 | 1,394 | 598 | Same template as F3, extended to 5 cities |
| **F5** | Dual counting (6 items) | 1,976 | 1,383 | 593 | "Exactly N of the following cities are in [country1] and M in [country2]: ..." |

F0–F2 are sized to match each other (853 true / 853 false, no artificial cap). F3–F5 target Poulis's ~2,000-total convention; exact totals land slightly under 2,000 because of integer-division rounding when splitting evenly across stated-count buckets (see "Balancing" below).

### Arithmetic Family (A1–A3)
Equations with increasing operation depth.

| Task | Complexity | Total | Train | Test | Description |
|------|-----------|-------|-------|------|-------------|
| **A1** | Single operation | 1,000 | 700 | 300 | `a ± b = result` or `a × b = result` |
| **A2** | Two operations | 1,000 | 700 | 300 | `(a ⊕ b) ⊗ c = result` with mixed ±/×/÷ |
| **A3** | Three operations | 1,000 | 700 | 300 | `(a ⊕ b) ⊗ (c ⊘ d) = result` |

---

## Dataset Properties

### Label Distribution
All datasets are **balanced 50% true / 50% false** overall. F3–F5 are additionally balanced **within every stated-count bucket** (see below) — not just balanced globally.

### City Coverage
Cities are drawn from GeoNames `cities1000.txt` (population > 1,000), filtered to population > 500,000, deduplicated by name, and matched against **108 countries** via `CountryInfo.txt`. This yields **853 unique cities**, one per valid country-name match — a substantially larger and more current pool than Poulis's original snapshot (their paper reports "80 cities across 10 countries"). Country display names are aligned to GeoNames' own naming convention (e.g. `North Macedonia`, `Czechia`, `Eswatini`), not the ISO/UN-style names used internally in Poulis's original `countries` dict.

### Arithmetic Details
- Operands: integers [1, 99]
- Operations: +, −, ×, ÷ (division restricted to exact integer results)
- Wrong answers: perturbed by random offset in {−5, ..., −1, 1, ..., 5}
- All results are integers, negative results allowed

### Train/Test Split
All tasks use a stratified ~70% / 30% split (`train_test_split(..., stratify=label)`).

For F3–F5, train/test splitting happens **after** deduplicating on the underlying city-set (see "Leakage prevention" below), so no city combination appears in both splits.

---

## Divergences from Poulis et al.'s Original Methodology

Poulis's own F3–F5 generation scripts (provided directly by the author) were reviewed and found to have properties we chose not to replicate:

1. **Cross-bucket balancing.** Poulis's `balance_by_k`/`balance_by_kpair` balance true/false counts *within* each stated-count value, but never equalize the *number* of examples *across* different stated-count values — bucket sizes are left to whatever the raw generation loop's combinatorics happen to produce. Left uncorrected, this implicitly re-weights the pooled AUROC toward whichever stated-count values were easiest to generate in bulk, contaminating the metric independent of true model performance (see notebook for the full mechanism). Our generators (`data_gen_F2345.py`) instead compute `n_per_k = n_total // num_buckets` and generate exactly that many true/false examples for every stated-count value, for F3, F4, and F5 alike.

2. **Boundary-case inclusion.** Poulis excludes stated-count values of 0 and N ("skip trivial all-or-none cases" / "skip trivial cases: no city from either, or all cities in just one bucket") on the grounds that they are trivially easy. We include them. A per-layer, per-stated-count AUROC breakdown (`check_trivial_cases_all_or_none_correct.ipynb`) shows this justification does not hold: F3 and F4's boundary cases show a transient early-layer inversion (the probe tracks raw match count rather than the stated-vs-actual comparison) that fully resolves by later layers, at which point boundary cases perform as well as or better than interior cases — the opposite of "trivial." F5's boundary cases (including the newly-added `k1=0`/`k2=0`) show the same phenomenon more severely and do not cleanly resolve at current sample sizes, so there we treat the evidence as insufficient grounds for exclusion rather than positive proof of non-triviality.

---

## File Format

Columns differ by task family, reflecting each task's underlying structure:

**F0/F1** — `statement, label, city, country, correct_country`

**F2** — `statement, label, stmt1_id, stmt2_id` (IDs reference the row index of each constituent F0 statement)

**F3/F4** — `statement, label, country, cities, stated_k, actual_k`
- `stated_k`: the count claimed in the statement text
- `actual_k`: the true number of matching cities (equals `stated_k` iff `label=1`)

**F5** — `statement, label, countries, cities, stated_k1, stated_k2, actual_k1, actual_k2`

**A1/A2/A3** — `statement, label`

```csv
statement,label
"The city of Paris is in France.",1
"The city of Berlin is in France.",0
```

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
- False examples' actual counts are drawn from the full range of valid alternatives, not fixed offsets
- Distractor cities (belonging to neither/no named target country) are randomly sampled
- Every stated-count bucket (or `(k1,k2)` pair for F5) is generated to an equal target size and independently balanced 50/50 true/false

### A1–A3
- Integer division constraint eliminates non-exact results
- Wrong answer offset is randomized to avoid trivial patterns
- Negative results are allowed (e.g., "3 - 19 = -16")

### Known Limitations
- **F5 per-bucket sample size** (~20–30 test examples per `(k1,k2)` pair, given 26 valid pairs) is small enough that individual-cell AUROC estimates are noisy; treat per-cell numbers as directional, not precise
- **F0–F2 city pool** depends on live GeoNames data; re-running `dataset_generation/plain_dataset_generation/build_geonames_csv.py` against a future GeoNames snapshot will change exact counts
- **A3 generation** filters for valid integer division; if error rates >30%, consider larger operand ranges

---

## Replication & Extension

To regenerate the dataset from scratch:
1. `dataset_generation/plain_dataset_generation/build_geonames_csv.py` — builds `geonames.csv` from `cities1000.txt` + `CountryInfo.txt`
2. `dataset_generation/plain_dataset_generation/data_gen_F0&F1.py` — generates F0/F1
3. `dataset_generation/plain_dataset_generation/data_gen_F2345.py` — generates F2, F3, F4, F5 (reads F0's output)
4. Arithmetic (A1–A3) generation script — generates the arithmetic family

To extend further:
1. Adjust `F345_TRUE_SAMPLE_AMOUNT`/`F345_FALSE_SAMPLE_AMOUNT` (F3–F5) or `TRUE_SAMPLE_AMOUNT`/`FALSE_SAMPLE_AMOUNT` (F2) in `data_gen_F2345.py` for different dataset sizes — keep them equal to preserve per-bucket balance
2. Widen `K_RANGE`/`N_CITIES` for F3–F5 to test additional list lengths
3. Adjust operand ranges in arithmetic tasks
4. Always re-run `check_f5_leakage.py`-style diagnostics after changing the generation logic, to confirm train/test city-group separation holds

---

## References

- **Marks & Tegmark (2024)**: "The Geometry of Truth" — established baseline truth direction probes
- **Poulis et al. (2024)**: "Testing the Limits of Truth Directions in LLMs" — complexity hierarchy and generalization failures; F3–F5 generation scripts and confirmation of no-chat-template methodology provided directly by the author (personal correspondence)
