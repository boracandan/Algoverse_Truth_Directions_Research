"""
Experiment 12: the model's confidence in its own CoT verdict, as a second error-detector
baseline for Table 2 (next to the plain probe and the no-reasoning answer from exp11).

For every item with a verdict from an "Answer:" line, the stored trace is cut right after
its last "Answer:" and fed to the model in one forward pass (no generation). The doubt
score is the probability of the answer the model did NOT give, p(other) / (p(Yes) + p(No)).
It is scored like the other detectors: error-AUROC [95% bootstrap CI] against
error = (CoT verdict != true label), on the same items as exp11. This baseline is free at
deployment, since the logits exist when the model writes its answer.

Usage (local 4090, ~10-20 min for F0-F5):
  python experiments/review_experiments/exp_12_model_confidence_baseline.py --tasks F0 F1 F2 F3 F4 F5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import exp3_probe_vs_verdict as E3  # noqa: E402
import exp10_direct_answer as E10  # noqa: E402


def run(args):
    model, tokenizer = C.load_model(C.DEEPSEEK_ID)
    tasks = args.tasks
    out_dir = C.RESULTS_DIR / "exp12_confidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for task in tasks:
        # Load in the verdicts 
        v = pd.read_csv(E3.VERDICT_DIR / f"{task}.csv")
        v = v[v["model_verdict"].notna()].reset_index(drop=True)
        # drop the item(s) whose verdict came from the think line: no "Answer:" in the generation
        v = v[v["verdict_source"] == "answer_line"].reset_index(drop=True)

        # load full traces
        traces = {s: C.load_split("sentence-based-CoT", task, split=s)["generated_statement_texts"]
                  for s in ["train", "test"]}
        v["trace"] = [traces[s].iloc[r] for s, r in zip(v["split"], v["row"])]

        # Remove the last Yes/No
        v["trace"] = v["trace"].apply(lambda trace: trace[:trace.rfind("Answer:") + len("Answer:")])

        # Tokenize the cut traces (default adds BOS, as at generation time) and read
        # p(" Yes"), p(" No") at the last position: one forward pass per batch, no generation
        yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
        no_id = tokenizer.encode(" No", add_special_tokens=False)[0]
        ids = [[tokenizer.bos_token_id] + x if x[0] != tokenizer.bos_token_id else x
               for x in tokenizer(v["trace"].tolist(), add_special_tokens=False)["input_ids"]]
        p_yes, p_no = E10.direct_answers(ids, model, C.DEEPSEEK_PAD_ID, yes_id, no_id,
                                         batch_size=args.batch_size)

        # Doubt = probability of the answer the model did NOT give (always < 0.5 under greedy);
        # error = was the CoT answer wrong? Scored like the other Table 2 detectors.
        mass = p_yes + p_no
        verdict = v["model_verdict"].to_numpy().astype(int)
        y = v["label"].to_numpy().astype(int)
        score = np.where(verdict == 1, p_no, p_yes) / mass
        error = (verdict != y).astype(int)
        auc, lo, hi = C.auroc_with_ci(error, score, seed=args.seed)
        print(f"{task}: items {len(v)}, CoT errors {error.sum()}, "
              f"error-AUROC {auc:.3f} [{lo:.3f}, {hi:.3f}], median yes/no mass {np.median(mass):.4f}")

        # per-item file, so the numbers can be rechecked or joined with exp11's items later
        items = v[["split", "row", "statement", "label", "model_verdict"]].copy()
        items["p_yes"], items["p_no"], items["yes_no_mass"] = p_yes, p_no, mass
        items["doubt_score"], items["cot_wrong"] = score, error
        items.to_csv(out_dir / f"{task}.csv", index=False)

        summary.append({
            "task": task, "items": len(v), "cot_errors": int(error.sum()),
            "error_AUROC": f"{auc:.3f} [{lo:.3f}, {hi:.3f}]",
            "median_yes_no_mass": round(float(np.median(mass)), 4),
            "median_doubt_correct": float(np.median(score[error == 0])),
            "median_doubt_wrong": float(np.median(score[error == 1])),
        })

    if summary:
        path = C.RESULTS_DIR / "exp12_confidence_summary.csv"
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        new = pd.concat([old, pd.DataFrame(summary)]).drop_duplicates("task", keep="last")
        new.to_csv(path, index=False)
        print(new.to_string(index=False))

        



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=C.FACT_TASKS)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed, same as exp11")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="keep small: logits are computed for every position of ~1k-token traces")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
