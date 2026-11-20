"""
Experiment 3: does the probe read the truth, or the verdict the model is about to give?

The sentence-based CoT prefixes stop right before the model's first verdict word. By then
the model has usually decided, so a probe there could be reading "truth" or "the answer I
am about to write". On most items the two agree (the model is right ~90-99% of the time),
so ordinary AUROC cannot tell them apart. They only come apart on the items where the
model's final answer is WRONG.

On those disagreement items, AUROC against the true label is the test:
  > 0.5  the probe sides with the truth, against the model's own answer
  < 0.5  the probe sides with the model's (wrong) answer
On that subset the verdict is exactly 1 - label, so AUROC vs verdict = 1 - AUROC vs label
and only the label number is reported.

Step 1 (CPU, no model): parse the model's answer out of every generated trace and write
datasets/CoT_datasets/verdicts/{task}.csv. Run with --parse-only on a laptop.

Step 2 (GPU): 5-fold cross-validation over train+test pooled, so every item gets an
out-of-fold score and all disagreement items are usable (the test split alone has ~50 in
F0, and almost none in A1-A3). Grouped by statement so a duplicated statement never sits
on both sides of a fold. Three probes per fold and layer:
  cot_label    trained on CoT activations with the true label      (the paper's probe)
  cot_verdict  trained on CoT activations with the model's answer  (a verdict probe)
  plain_label  trained on plain-statement activations, true label  (the transfer probe),
               scored both on plain activations and on CoT activations of held-out items

Arithmetic has almost no model errors, so this experiment can only settle the factual
tasks. For arithmetic, use exp4 (earlier readout points) instead.

Usage:
  python experiments/review_experiments/exp3_probe_vs_verdict.py --parse-only
  python experiments/review_experiments/exp3_probe_vs_verdict.py --tasks F0 F3 F4 F5 \
      --layers all --cache-dir /content/drive/MyDrive/dsab_cache
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

COT = "sentence-based-CoT"
PLAIN = "no-prompt"
VERDICT_DIR = C.DATA_DIR / "CoT_datasets" / "verdicts"


# ---------------------------------------------------------------------------------------
# Step 1: parse verdicts
# ---------------------------------------------------------------------------------------

def parse_verdict(generated_text):
    """Returns (final_answer, think_verdict) as 1/0/None.

    The prompt itself contains the template lines 'Answer: Yes' / 'Answer: No' and
    'The statement is true/false', so only text after <｜Assistant｜> is searched. The
    final 'Answer:' line is the model's committed answer. The think verdict only counts
    the exact line the prompt asks for ('The statement is true.' on its own line): a
    looser 'statement is true' match picks up hypotheticals mid-reasoning ('if X, the
    statement is true') and produced dozens of false conflicts on F3/F4.
    """
    gen = generated_text.split("<｜Assistant｜>", 1)[-1]
    answers = re.findall(r"Answer:\s*\**\s*(Yes|No)\b", gen, flags=re.IGNORECASE)
    thinks = re.findall(r"^\s*\**the statement is\s+(true|false)\.?\**\s*$", gen,
                        flags=re.IGNORECASE | re.MULTILINE)
    final = None if not answers else int(answers[-1].lower() == "yes")
    think = None if not thinks else int(thinks[-1].lower() == "true")
    return final, think


def build_verdicts(tasks):
    VERDICT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for task in tasks:
        parts = []
        for split in ["train", "test"]:
            df = C.load_split(COT, task, split)
            parsed = df["generated_statement_texts"].map(parse_verdict)
            out = pd.DataFrame({
                "split": split,
                "row": np.arange(len(df)),
                "statement": [C.statement_text(r) for _, r in df.iterrows()],
                "label": df["y"],
                "answer_verdict": [p[0] for p in parsed],
                "think_verdict": [p[1] for p in parsed],
            })
            # Committed answer first; fall back to the in-thinking line if the answer
            # line is missing (e.g. generation cut off before </think>).
            out["model_verdict"] = out["answer_verdict"].fillna(out["think_verdict"])
            out["verdict_source"] = np.where(out["answer_verdict"].notna(), "answer_line",
                                             np.where(out["think_verdict"].notna(), "think_line", "none"))
            parts.append(out)
        out = pd.concat(parts, ignore_index=True)
        out.to_csv(VERDICT_DIR / f"{task}.csv", index=False)

        has = out["model_verdict"].notna()
        both = out["answer_verdict"].notna() & out["think_verdict"].notna()
        summary.append({
            "task": task,
            "items": len(out),
            "parsed": int(has.sum()),
            "model_accuracy": round(float((out.loc[has, "model_verdict"] == out.loc[has, "label"]).mean()), 3),
            "disagree_total": int((has & (out["model_verdict"] != out["label"])).sum()),
            "disagree_test_split": int((has & (out["model_verdict"] != out["label"]) & (out["split"] == "test")).sum()),
            "think_vs_answer_conflicts": int((both & (out["answer_verdict"] != out["think_verdict"])).sum()),
        })
    summary = pd.DataFrame(summary)
    print(summary.to_string(index=False))
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(C.RESULTS_DIR / "exp3_verdict_summary.csv", index=False)
    return summary


# ---------------------------------------------------------------------------------------
# Step 2: probes
# ---------------------------------------------------------------------------------------

def plain_rows_for(statements, task, tokenizer):
    """Plain-statement ids for the same statements, in the same order (exp6 matching)."""
    # Read the full plain files directly: this is a lookup table, never subsampled.
    plain = pd.concat([pd.read_csv(C.CONDITION_DICT[PLAIN] / f"{task}_{s}.csv") for s in ["train", "test"]],
                      ignore_index=True)
    lookup = dict(zip(plain["statement"].str.strip(), plain["statement"]))
    missing = [s for s in statements if s not in lookup]
    if missing:
        raise ValueError(f"{task}: {len(missing)} CoT statements have no plain match, e.g. {missing[0]!r}")
    return C.plain_ids([lookup[s] for s in statements], tokenizer)


def run_probes(tasks, layers_arg, seeds, folds, cache_dir, batch_size):
    model, tokenizer = C.load_model(C.DEEPSEEK_ID)
    n_layers = model.config.num_hidden_layers + 1
    layers = C.parse_layers(layers_arg, n_layers)

    for task in tasks:
        todo = [s for s in seeds if not C.already_done("exp3_probe_vs_verdict", train_task=task, seed=s)]
        if not todo:
            print(f"\n{task}: already done, skipping (delete its rows from the results file to redo)")
            summarize(task)
            continue
        v = pd.read_csv(VERDICT_DIR / f"{task}.csv")
        cot = pd.concat([C.load_split(COT, task, s) for s in ["train", "test"]], ignore_index=True)
        assert len(cot) == len(v), "verdict file out of sync with CoT data; rerun --parse-only"

        keep = v["model_verdict"].notna().to_numpy()
        v, cot = v[keep].reset_index(drop=True), cot[keep].reset_index(drop=True)
        y = v["label"].to_numpy().astype(int)
        verdict = v["model_verdict"].to_numpy().astype(int)
        disagree = verdict != y
        print(f"\n{task}: {len(v)} items, {disagree.sum()} where the model's answer is wrong")

        cot_ids = C.ids_for(cot)
        plain_ids = plain_rows_for(v["statement"].tolist(), task, tokenizer)
        acts_cot = C.cached_activations(
            cache_dir, f"{C.DEEPSEEK_SHORT}__{COT}__{task}__all__final",
            lambda: C.extract_all_layer_activations(cot_ids, model, C.DEEPSEEK_PAD_ID, batch_size, f"{task} CoT"))
        acts_plain = C.cached_activations(
            cache_dir, f"{C.DEEPSEEK_SHORT}__{PLAIN}-matched__{task}__all__final",
            lambda: C.extract_all_layer_activations(plain_ids, model, C.DEEPSEEK_PAD_ID, batch_size, f"{task} plain"))

        groups = pd.factorize(v["statement"])[0]
        rows = []
        for seed in todo:
            # Stratify on label x verdict so each fold gets its share of the rare
            # disagreement items.
            strat = y * 2 + verdict
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
            fold_of = np.empty(len(v), dtype=int)
            for k, (_, te) in enumerate(splitter.split(np.zeros(len(v)), strat, groups)):
                fold_of[te] = k

            for layer in layers:
                Xc, Xp = acts_cot[layer], acts_plain[layer]
                oof = {name: np.zeros(len(v)) for name in
                       ["cot_label", "cot_verdict", "plain_label_on_plain", "plain_label_on_cot"]}
                for k in range(folds):
                    tr, te = fold_of != k, fold_of == k
                    w, m = C.fit_linear_probe(Xc[tr], y[tr], seed)
                    oof["cot_label"][te] = C.probe_scores(w, m, Xc[te])
                    w, m = C.fit_linear_probe(Xc[tr], verdict[tr], seed)
                    oof["cot_verdict"][te] = C.probe_scores(w, m, Xc[te])
                    w, m = C.fit_linear_probe(Xp[tr], y[tr], seed)
                    oof["plain_label_on_plain"][te] = C.probe_scores(w, m, Xp[te])
                    oof["plain_label_on_cot"][te] = C.probe_scores(w, m, Xc[te])

                for name, scores in oof.items():
                    train_cond = PLAIN if name.startswith("plain") else COT
                    test_cond = PLAIN if name.endswith("on_plain") else COT
                    for subset, mask in [("all", np.ones(len(v), bool)),
                                         ("model_wrong", disagree),
                                         ("model_right", ~disagree)]:
                        for target, labels in [("label", y), ("verdict", verdict)]:
                            if subset == "model_wrong" and target == "verdict":
                                continue  # = 1 - (vs label) on this subset
                            auc, lo, hi = C.auroc_with_ci(labels[mask], scores[mask], seed=seed)
                            rows.append(C.result_row(
                                experiment="exp3_probe_vs_verdict", model=C.DEEPSEEK_SHORT,
                                train_task=task, test_task=task,
                                train_condition=f"{train_cond}[{'verdict' if name == 'cot_verdict' else 'label'}]",
                                test_condition=test_cond, readout="final", layer=layer, seed=seed,
                                eval_target=target, subset=subset,
                                n_train=int(round(len(v) * (folds - 1) / folds)), n_test=int(mask.sum()),
                                auroc=auc, ci_low=lo, ci_high=hi))
            C.append_results(rows)
            rows = []
        summarize(task)


def summarize(task):
    """Per probe: choose the layer on ALL items (best AUROC vs the true label), then report
    the model-wrong AUROC at that layer. Choosing the layer on the model-wrong items
    themselves (e.g. the layer furthest from 0.5) would overstate the result.
    Exception: the plain probe applied to CoT activations reuses the layer selected on bare
    statements, so it is the same probe as the bare-statement one."""
    df = pd.read_csv(C.RESULTS_CSV)
    df = df[(df["experiment"] == "exp3_probe_vs_verdict") & (df["train_task"] == task) & (df["eval_target"] == "label")]
    if df.empty:
        return
    print(f"\n{task}: AUROC vs TRUE label on items the model got wrong "
          f"(>0.5 = follows truth, <0.5 = follows the model's answer); layer chosen on all items")
    sel = {}
    for (tr, te), g in df.groupby(["train_condition", "test_condition"]):
        sel[(tr, te)] = g.groupby(["subset", "layer"])[["auroc", "ci_low", "ci_high", "n_test"]].mean()
    bare_layer = sel[(f"{PLAIN}[label]", PLAIN)].loc["all", "auroc"].idxmax()
    for (tr, te), g in sel.items():
        # The plain probe on CoT reuses the bare-statement layer: choosing it on CoT
        # activations would favour verdict-like layers (truth and verdict agree on ~90% there).
        if (tr, te) == (f"{PLAIN}[label]", COT):
            layer = bare_layer
        else:
            layer = g.loc["all", "auroc"].idxmax()
        a, w = g.loc[("all", layer)], g.loc[("model_wrong", layer)]
        print(f"  {tr:>30} -> {te:<20} layer {layer:>2}: all items {a.auroc:.3f} | "
              f"model wrong {w.auroc:.3f} [{w.ci_low:.3f}, {w.ci_high:.3f}] n={int(w.n_test)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=C.TASKS)
    ap.add_argument("--parse-only", action="store_true", help="only build the verdict files (CPU)")
    ap.add_argument("--layers", default="all", help="all | 8,16,24 | 0-32:4")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    build_verdicts(args.tasks)
    if not args.parse_only:
        run_probes(args.tasks, args.layers, args.seeds, args.folds, args.cache_dir, args.batch_size)


if __name__ == "__main__":
    main()
