"""
Lexical and Text Baselines via Logistic Regression:
1. Final CoT Token (One-Hot) - Tests for localized pre-verdict lexical shortcuts.
2. Bag-of-Words + TF-IDF - Tests whether global surface word frequencies across
   the entire reasoning sequence linearly predict statement truthfulness.

Dataset: datasets/CoT_datasets/sentence_based_lexically_cleaned/
Tasks: F0, F1, F2, F3, F4, F5, A1, A2, A3
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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


def evaluate_final_token(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lowercase: bool = True,
    max_iter: int = 1000,
    random_state: int = 42,
) -> dict:
    """Evaluates the single final-token one-hot Logistic Regression baseline."""
    train_words = train_df["extracted_statement_texts"].apply(
        lambda x: extract_last_word(x, lowercase=lowercase)
    )
    test_words = test_df["extracted_statement_texts"].apply(
        lambda x: extract_last_word(x, lowercase=lowercase)
    )

    y_train = train_df["label"].astype(int).values
    y_test = test_df["label"].astype(int).values

    train_vocab = set(train_words)
    test_vocab = set(test_words)
    oov_test_samples = int((~test_words.isin(train_vocab)).sum())

    encoder = OneHotEncoder(handle_unknown="ignore")
    X_train = encoder.fit_transform(np.array(train_words).reshape(-1, 1))
    X_test = encoder.transform(np.array(test_words).reshape(-1, 1))

    clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_pred_proba)

    feature_names = encoder.categories_[0]
    coefs = clf.coef_[0]
    top_pos_idx = np.argsort(coefs)[-3:][::-1]
    top_neg_idx = np.argsort(coefs)[:3]
    top_pos = [f"{feature_names[i]} (+{coefs[i]:.2f})" for i in top_pos_idx]
    top_neg = [f"{feature_names[i]} ({coefs[i]:.2f})" for i in top_neg_idx]

    return {
        "AUROC": auroc,
        "Train_Vocab_Size": len(train_vocab),
        "Test_Vocab_Size": len(test_vocab),
        "Test_OOV_Count": oov_test_samples,
        "Test_OOV_Pct": (oov_test_samples / len(test_df)) * 100,
        "Top_Predictive_True": ", ".join(top_pos),
        "Top_Predictive_False": ", ".join(top_neg),
    }


def evaluate_tfidf_bow(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lowercase: bool = True,
    ngram_range: tuple[int, int] = (1, 1),
    max_features: int | None = None,
    max_iter: int = 1000,
    random_state: int = 42,
) -> dict:
    """Evaluates the Bag-of-Words + TF-IDF index Logistic Regression baseline on full reasoning texts."""
    X_train_raw = train_df["extracted_statement_texts"].fillna("").astype(str)
    X_test_raw = test_df["extracted_statement_texts"].fillna("").astype(str)

    y_train = train_df["label"].astype(int).values
    y_test = test_df["label"].astype(int).values

    vectorizer = TfidfVectorizer(
        lowercase=lowercase,
        ngram_range=ngram_range,
        max_features=max_features,
        token_pattern=r"(?u)\b[A-Za-z0-9']+\b",
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_pred_proba)

    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]
    top_pos_idx = np.argsort(coefs)[-3:][::-1]
    top_neg_idx = np.argsort(coefs)[:3]
    top_pos = [f"{feature_names[i]} (+{coefs[i]:.2f})" for i in top_pos_idx]
    top_neg = [f"{feature_names[i]} ({coefs[i]:.2f})" for i in top_neg_idx]

    return {
        "AUROC": auroc,
        "Train_Vocab_Size": len(feature_names),
        "Top_Predictive_True": ", ".join(top_pos),
        "Top_Predictive_False": ", ".join(top_neg),
    }


def run_evaluation(
    data_dir: Path,
    tasks: list[str],
    methods: list[str] = ["final_token", "tfidf_bow"],
    lowercase: bool = True,
    ngram_range: tuple[int, int] = (1, 1),
    max_features: int | None = None,
    max_iter: int = 1000,
    random_state: int = 42,
    output_dir: Path | None = None,
):
    detailed_rows = []
    comparison_rows = []

    print("=" * 80)
    print("BASELINE EXPERIMENTS: FINAL TOKEN SHORTCUT & FULL-TEXT TF-IDF BAG-OF-WORDS")
    print(f"Data directory: {data_dir}")
    print(f"Methods: {methods}")
    print(f"Lowercase: {lowercase}")
    print(f"TF-IDF n-gram range: {ngram_range}")
    print("=" * 80)
    print()

    for task in tasks:
        train_path = data_dir / f"{task}_train.csv"
        test_path = data_dir / f"{task}_test.csv"

        if not train_path.exists() or not test_path.exists():
            print(f"Warning: Missing files for task {task} at {train_path} or {test_path}")
            continue

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        comp_entry = {
            "Task": task,
            "Category": "Arithmetic" if task.startswith("A") else "Factual",
            "Train_Samples": len(train_df),
            "Test_Samples": len(test_df),
            "Train_True_Pct": round(train_df["label"].mean() * 100, 1),
            "Test_True_Pct": round(test_df["label"].mean() * 100, 1),
        }

        task_print_parts = [f"[{task}]"]

        # 1. Final Token Baseline
        if "final_token" in methods:
            ft_res = evaluate_final_token(
                train_df=train_df,
                test_df=test_df,
                lowercase=lowercase,
                max_iter=max_iter,
                random_state=random_state,
            )
            detailed_rows.append(
                {
                    "Task": task,
                    "Category": comp_entry["Category"],
                    "Method": "Final_Token_OneHot",
                    "AUROC": ft_res["AUROC"],
                    "Train_Samples": len(train_df),
                    "Test_Samples": len(test_df),
                    "Vocab_Size": ft_res["Train_Vocab_Size"],
                    "Test_OOV_Pct": ft_res["Test_OOV_Pct"],
                    "Top_Predictive_True": ft_res["Top_Predictive_True"],
                    "Top_Predictive_False": ft_res["Top_Predictive_False"],
                }
            )
            comp_entry["AUROC_Final_Token"] = ft_res["AUROC"]
            comp_entry["Top_Final_Token_True"] = ft_res["Top_Predictive_True"]
            comp_entry["Top_Final_Token_False"] = ft_res["Top_Predictive_False"]
            task_print_parts.append(f"Final Token AUROC: {ft_res['AUROC']:.4f}")

        # 2. TF-IDF Bag-of-Words Baseline
        if "tfidf_bow" in methods:
            tfidf_res = evaluate_tfidf_bow(
                train_df=train_df,
                test_df=test_df,
                lowercase=lowercase,
                ngram_range=ngram_range,
                max_features=max_features,
                max_iter=max_iter,
                random_state=random_state,
            )
            detailed_rows.append(
                {
                    "Task": task,
                    "Category": comp_entry["Category"],
                    "Method": "TFIDF_BoW",
                    "AUROC": tfidf_res["AUROC"],
                    "Train_Samples": len(train_df),
                    "Test_Samples": len(test_df),
                    "Vocab_Size": tfidf_res["Train_Vocab_Size"],
                    "Test_OOV_Pct": np.nan,
                    "Top_Predictive_True": tfidf_res["Top_Predictive_True"],
                    "Top_Predictive_False": tfidf_res["Top_Predictive_False"],
                }
            )
            comp_entry["AUROC_TFIDF_BoW"] = tfidf_res["AUROC"]
            comp_entry["Vocab_TFIDF"] = tfidf_res["Train_Vocab_Size"]
            comp_entry["Top_TFIDF_True"] = tfidf_res["Top_Predictive_True"]
            comp_entry["Top_TFIDF_False"] = tfidf_res["Top_Predictive_False"]
            task_print_parts.append(f"TF-IDF BoW AUROC: {tfidf_res['AUROC']:.4f}")

        if "final_token" in methods and "tfidf_bow" in methods:
            delta = comp_entry["AUROC_TFIDF_BoW"] - comp_entry["AUROC_Final_Token"]
            comp_entry["Delta_TFIDF_minus_Token"] = delta
            task_print_parts.append(f"Delta: {delta:+.4f}")

        print(" | ".join(task_print_parts))
        comparison_rows.append(comp_entry)

    comp_df = pd.DataFrame(comparison_rows)
    detailed_df = pd.DataFrame(detailed_rows)

    print()
    print("=" * 80)
    print("SIDE-BY-SIDE BASELINE COMPARISON SUMMARY")
    print("=" * 80)
    display_cols = ["Task", "Category"]
    if "AUROC_Final_Token" in comp_df.columns:
        display_cols.append("AUROC_Final_Token")
    if "AUROC_TFIDF_BoW" in comp_df.columns:
        display_cols.append("AUROC_TFIDF_BoW")
    if "Delta_TFIDF_minus_Token" in comp_df.columns:
        display_cols.append("Delta_TFIDF_minus_Token")

    formatters = {
        "AUROC_Final_Token": "{:.4f}".format,
        "AUROC_TFIDF_BoW": "{:.4f}".format,
        "Delta_TFIDF_minus_Token": "{:+.4f}".format,
    }
    print(comp_df[display_cols].to_string(index=False, formatters=formatters))

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_cased" if not lowercase else ""

        comp_csv = output_dir / f"baseline_comparison_results{suffix}.csv"
        comp_df.to_csv(comp_csv, index=False)
        print(f"\nSaved side-by-side comparison to {comp_csv}")

        detailed_csv = output_dir / f"baseline_detailed_results{suffix}.csv"
        detailed_df.to_csv(detailed_csv, index=False)
        print(f"Saved detailed results to {detailed_csv}")

    return comp_df, detailed_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Lexical and BoW/TF-IDF Baselines via Logistic Regression"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/CoT_datasets/sentence_based_lexically_cleaned",
        help="Path to sentence-based lexically cleaned dataset folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/baseline_experiments",
        help="Directory to save output CSV summaries",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["F0", "F1", "F2", "F3", "F4", "F5", "A1", "A2", "A3"],
        help="Tasks to evaluate",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["final_token", "tfidf_bow"],
        choices=["final_token", "tfidf_bow"],
        help="Baselines to run (default: final_token tfidf_bow)",
    )
    parser.add_argument(
        "--ngram_min",
        type=int,
        default=1,
        help="Minimum n-gram size for TF-IDF (default: 1)",
    )
    parser.add_argument(
        "--ngram_max",
        type=int,
        default=1,
        help="Maximum n-gram size for TF-IDF (default: 1)",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=None,
        help="Max vocabulary features for TF-IDF",
    )
    parser.add_argument(
        "--no_lowercase",
        action="store_true",
        help="Do not convert extracted text/words to lowercase",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for LogisticRegression",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Maximum iterations for LogisticRegression",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = (project_root / args.data_dir) if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    out_dir = (
        (project_root / args.output_dir) if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    )

    run_evaluation(
        data_dir=data_path,
        tasks=args.tasks,
        methods=args.methods,
        lowercase=not args.no_lowercase,
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.max_features,
        max_iter=args.max_iter,
        random_state=args.random_state,
        output_dir=out_dir,
    )
