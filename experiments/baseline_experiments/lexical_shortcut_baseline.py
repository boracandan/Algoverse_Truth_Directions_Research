"""
Lexical Shortcut Baseline via Logistic Regression on Final CoT Tokens

This script evaluates whether the final token/word (or number) in the extracted
Chain-of-Thought (CoT) reasoning sequence can predict statement truthfulness
on its own.

Dataset: datasets/CoT_datasets/sentence_based_lexically_cleaned/
Tasks: F0, F1, F2, F3, F4, F5, A1, A2, A3
Methodology:
  1. Extract the last word or number from 'extracted_statement_texts' for each sample.
  2. One-hot encode the words using vocabulary fit strictly on the training set.
  3. Train a Logistic Regression model on the training set (target: binarized label).
  4. Evaluate on the test set and report the AUROC score per task.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

# Set stdout encoding for Windows console environments
sys.stdout.reconfigure(encoding="utf-8")


def extract_last_word(text: str, lowercase: bool = True) -> str:
    """
    Extracts the very last word or number from an extracted statement text.
    Handles trailing punctuation, whitespace, and numbers.
    """
    if pd.isna(text) or not str(text).strip():
        return "<EMPTY>"
    tokens = re.findall(r"[A-Za-z']+|\d+", str(text).rstrip())
    if not tokens:
        return "<EMPTY>"
    token = tokens[-1]
    return token.lower() if lowercase else token


def run_evaluation(
    data_dir: Path,
    tasks: list[str],
    lowercase: bool = True,
    max_iter: int = 1000,
    random_state: int = 42,
    output_csv: Path | None = None,
):
    results = []

    print("=" * 70)
    print("LEXICAL SHORTCUT BASELINE: LOGISTIC REGRESSION ON FINAL CoT TOKENS")
    print(f"Data directory: {data_dir}")
    print(f"Lowercase tokens: {lowercase}")
    print("=" * 70)
    print()

    for task in tasks:
        train_path = data_dir / f"{task}_train.csv"
        test_path = data_dir / f"{task}_test.csv"

        if not train_path.exists() or not test_path.exists():
            print(f"Warning: Missing files for task {task} at {train_path} or {test_path}")
            continue

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Extract last word/number
        train_words = train_df["extracted_statement_texts"].apply(
            lambda x: extract_last_word(x, lowercase=lowercase)
        )
        test_words = test_df["extracted_statement_texts"].apply(
            lambda x: extract_last_word(x, lowercase=lowercase)
        )

        y_train = train_df["label"].astype(int).values
        y_test = test_df["label"].astype(int).values

        # Vocab statistics
        train_vocab = set(train_words)
        test_vocab = set(test_words)
        oov_test_samples = (~test_words.isin(train_vocab)).sum()

        # One-hot encoding strictly fit on training set
        encoder = OneHotEncoder(handle_unknown="ignore")
        X_train = encoder.fit_transform(np.array(train_words).reshape(-1, 1))
        X_test = encoder.transform(np.array(test_words).reshape(-1, 1))

        # Train Logistic Regression
        clf = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)

        # Predict probability for the positive class (True = 1)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, y_pred_proba)

        # Also get top positive and negative features
        feature_names = encoder.categories_[0]
        coefs = clf.coef_[0]
        top_pos_idx = np.argsort(coefs)[-3:][::-1]
        top_neg_idx = np.argsort(coefs)[:3]
        top_pos = [f"{feature_names[i]} (+{coefs[i]:.2f})" for i in top_pos_idx]
        top_neg = [f"{feature_names[i]} ({coefs[i]:.2f})" for i in top_neg_idx]

        results.append(
            {
                "Task": task,
                "AUROC": auroc,
                "Train_Samples": len(train_df),
                "Test_Samples": len(test_df),
                "Train_True_Pct": train_df["label"].mean() * 100,
                "Test_True_Pct": test_df["label"].mean() * 100,
                "Train_Vocab_Size": len(train_vocab),
                "Test_Vocab_Size": len(test_vocab),
                "Test_OOV_Count": oov_test_samples,
                "Test_OOV_Pct": (oov_test_samples / len(test_df)) * 100,
                "Top_Predictive_True": ", ".join(top_pos),
                "Top_Predictive_False": ", ".join(top_neg),
            }
        )

        print(f"{task} -> AUROC Score: {auroc:.4f}")

    results_df = pd.DataFrame(results)

    print()
    print("=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)
    summary_cols = ["Task", "AUROC", "Train_Samples", "Test_Samples", "Train_Vocab_Size", "Test_OOV_Pct"]
    print(results_df[summary_cols].to_string(index=False))

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        print(f"\nSaved detailed results to {output_csv}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Lexical Shortcut Baseline via Logistic Regression")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/CoT_datasets/sentence_based_lexically_cleaned",
        help="Path to sentence-based lexically cleaned dataset folder",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="experiments/baseline_experiments/lexical_shortcut_results.csv",
        help="Path to save output CSV summary",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["F0", "F1", "F2", "F3", "F4", "F5", "A1", "A2", "A3"],
        help="Tasks to evaluate",
    )
    parser.add_argument(
        "--no_lowercase",
        action="store_true",
        help="Do not convert extracted words to lowercase",
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for LogisticRegression")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum iterations for LogisticRegression")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = (project_root / args.data_dir) if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    output_path = (
        (project_root / args.output_csv) if not Path(args.output_csv).is_absolute() else Path(args.output_csv)
    )

    run_evaluation(
        data_dir=data_path,
        tasks=args.tasks,
        lowercase=not args.no_lowercase,
        max_iter=args.max_iter,
        random_state=args.random_state,
        output_csv=output_path,
    )
