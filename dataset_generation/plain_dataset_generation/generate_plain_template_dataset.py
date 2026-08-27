# Generates the chat-templated no-prompt baseline ("plain_template_dataset"): the plain
# statements, chat-templated with add_generation_prompt=True and nothing else added, so each
# row ends "<｜Assistant｜><think>\n" with zero content after it.
#
# This is NOT the paper-replication "no-prompt" used to validate against Poulis (datasets/
# plain_dataset stays untouched, no chat template, for that comparison). This is the anchor
# the filler-token-only and instructions-only ablations (dataset_generation/
# ablation_dataset_generation/generate_ablation_sets.py) are actually measured against: both
# of those already use add_generation_prompt=True, so this baseline shares that same scaffold,
# and each ablation adds exactly one thing on top of it -- filler tokens, or instructions text.
# "Entering <think> mode" is held fixed across this baseline and both ablations rather than
# treated as a variable, since there's no meaningful way to have reasoning-shaped content
# without it.
#
# Every column already present in datasets/plain_dataset is preserved as-is (these differ by
# task -- e.g. F0/F1 carry city/country/correct_country, F3/F4 carry stated_k/actual_k, etc.);
# only extracted_statement_ids/extracted_statement_texts are added, matching the schema used by
# CoT_datasets and ablation_datasets.

from transformers import AutoTokenizer
from pathlib import Path
import pandas as pd

PLAIN_STATEMENTS_FOLDER = Path(__file__).resolve().parent.parent.parent / "datasets" / "plain_dataset"
OUTPUT_FOLDER = Path(__file__).resolve().parent.parent.parent / "datasets" / "plain_template_dataset"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

TASKS = sorted({p.stem.split("_")[0] for p in PLAIN_STATEMENTS_FOLDER.glob("*_train.csv")})
print(TASKS)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")


def build_plain_template_df(task, split):
    df = pd.read_csv(PLAIN_STATEMENTS_FOLDER / f"{task}_{split}.csv")

    def build_ids(statement):
        messages = [{"role": "user", "content": statement}]
        out = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        return out["input_ids"]

    df["extracted_statement_ids"] = df["statement"].apply(build_ids)
    df["extracted_statement_texts"] = df["extracted_statement_ids"].apply(tokenizer.decode)
    return df


for task in TASKS:
    df_train = build_plain_template_df(task, "train")
    df_test = build_plain_template_df(task, "test")

    df_train.to_csv(OUTPUT_FOLDER / f"{task}_train.csv", index=False)
    df_test.to_csv(OUTPUT_FOLDER / f"{task}_test.csv", index=False)