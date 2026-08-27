# Generating the filler-token only and the instruction-format-only ablation datasets for the 
# Deepseek-R1-LLama-8b-Distilled generated datasets.

from transformers import AutoTokenizer
from pathlib import Path
import pandas as pd
from ast import literal_eval

# Dataset folders -- anchored to this script's own location, not the working directory it
# happens to be run from, so it works regardless of cwd. This script lives at
# dataset_generation/ablation_dataset_generation/, so .parent.parent.parent reaches the repo
# root (one more .parent than before the reorg, since it's now nested one level deeper).
DATASET_FOLDER = Path(__file__).resolve().parent.parent.parent / "datasets" / "CoT_datasets" / "sentence_based_lexically_cleaned"
PLAIN_STATEMENTS_FOLDER = Path(__file__).resolve().parent.parent.parent / "datasets" / "plain_dataset"
TASKS = sorted({p.stem.split("_")[0] for p in DATASET_FOLDER.glob("*_train.csv")})
print(TASKS)

# Verbatim copy of the instructions used for the real generation (generating_CoT_deepseekR1_distill_llama.ipynb)
# -- kept identical so the "instructions" ablation adds exactly this and nothing else.
COT_INSTRUCTIONS = (
    "Work through the statement above before answering. State the facts you need — for "
    "each city named, where it actually is; for each arithmetic sub-expression, its "
    "actual value — then explicitly compute or count what the statement claims, and "
    "compare.\n\n"
    "Do not use the words True, False, correct, incorrect, accurate, or inaccurate while "
    "reasoning. Reserve them for your conclusion only.\n\n"
    "End your reasoning with exactly one line, either:\n"
    "The statement is true.\n"
    "or\n"
    "The statement is false.\n"
    "Then write </think>. After </think>, on its own line, write exactly one of:\n"
    "Answer: Yes\n"
    "or\n"
    "Answer: No"
)

# Anchored to this script's own location (not cwd), same as DATASET_FOLDER above. Each
# ablation type gets its own subfolder -- otherwise both loops below write to the same
# {task}_train.csv filenames and the second one silently overwrites the first.
ABLATION_FOLDER = Path(__file__).resolve().parent.parent.parent / "datasets" / "ablation_datasets"
FILLER_ABLATION_FOLDER = ABLATION_FOLDER / "filler_token_only"
INSTRUCTIONS_ABLATION_FOLDER = ABLATION_FOLDER / "instructions_only"
FILLER_ABLATION_FOLDER.mkdir(parents=True, exist_ok=True)
INSTRUCTIONS_ABLATION_FOLDER.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# CONSTANT IDS
THINK_TOKEN_ID = tokenizer("<think>")["input_ids"][0]
FILLER_TOKEN_ID = tokenizer(".")["input_ids"][0]


def load_dataset(path):
    df = pd.read_csv(path)
    df["extracted_statement_ids"] = df["extracted_statement_ids"].apply(literal_eval)
    return df

def get_statement(generated_text):
    return generated_text.split("<｜User｜>")[1].split("\n")[0].strip()

def build_filler_only_df(task, split):
    plain_df = pd.read_csv(PLAIN_STATEMENTS_FOLDER / f"{task}_{split}.csv")[["statement", "label"]]
    CoT_df = load_dataset(DATASET_FOLDER / f"{task}_{split}.csv")
    CoT_df["statement"] = CoT_df["generated_statement_texts"].apply(get_statement)


    # Check if think token count is 1
    CoT_df["token_count"] = CoT_df["extracted_statement_ids"].apply(lambda el: el.count(THINK_TOKEN_ID))
    assert (CoT_df["token_count"] == 1).all()

    # Find the filler length
    CoT_df["think_index"] = CoT_df["extracted_statement_ids"].apply(lambda el: el.index(THINK_TOKEN_ID))
    CoT_df["filler_len"] = CoT_df.apply(lambda row: len(row["extracted_statement_ids"]) - (row["think_index"] + 1), axis=1)

    plain_df = plain_df.merge(CoT_df[["statement", "filler_len"]], on="statement", how="inner")


    def build_ids(row):
        messages = [{"role": "user", "content": f"{row["statement"]}"}]
        # add_generation_prompt=True already appends "<｜Assistant｜><think>\n" for this
        # model's chat template -- appending another THINK_TOKEN_ID here would double it
        # up ("<think>\n<think>..."), so only the filler goes on top of what the template
        # already gives us.
        out = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        return out["input_ids"] + [FILLER_TOKEN_ID] * row["filler_len"]

    plain_df["extracted_statement_ids"] = plain_df.apply(build_ids, axis=1)
    plain_df["extracted_statement_texts"] = plain_df["extracted_statement_ids"].apply(tokenizer.decode)
    return plain_df.drop(columns=["filler_len", "statement"])

def build_instructions_only_df(task, split):
    """Instructions ablation, anchored to no-prompt (not built by subtracting from the full
    CoT dataset): plain statement + COT_INSTRUCTIONS, chat-templated. add_generation_prompt=True
    appends "<｜Assistant｜><think>\n" (this model's chat template always does this, verified
    byte-identical to the real generation's explicit <think>\n prefill), so the readout lands
    right after entering <think> mode, with zero actual reasoning content -- not literally at
    the last instruction word. Relative to no-prompt, exactly one thing changes: the
    instructions are present (and, as a consequence of following the same template convention
    as every other condition, thinking mode gets entered but immediately reads out)."""
    df = pd.read_csv(PLAIN_STATEMENTS_FOLDER / f"{task}_{split}.csv")[["statement", "label"]]

    def build_ids(statement):
        messages = [{"role": "user", "content": f"{statement}\n\n{COT_INSTRUCTIONS}"}]
        out = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        return out["input_ids"]

    df["extracted_statement_ids"] = df["statement"].apply(build_ids)
    df["extracted_statement_texts"] = df["extracted_statement_ids"].apply(tokenizer.decode)
    return df.drop(columns=["statement"])

# Filler-token-only Ablation

for task in TASKS:
    # Load in the datasets
    df_train, df_test = build_filler_only_df(task, "train"), build_filler_only_df(task, "test")

    # Save the Ablation Datasets
    df_train.to_csv(FILLER_ABLATION_FOLDER / f"{task}_train.csv", index=False)
    df_test.to_csv(FILLER_ABLATION_FOLDER / f"{task}_test.csv", index=False)


# Instructions-only Ablation (anchored to no-prompt: plain statement + instructions, no <think>)

for task in TASKS:
    df_train, df_test = build_instructions_only_df(task, "train"), build_instructions_only_df(task, "test")

    # Save the Ablation Datasets
    df_train.to_csv(INSTRUCTIONS_ABLATION_FOLDER / f"{task}_train.csv", index=False)
    df_test.to_csv(INSTRUCTIONS_ABLATION_FOLDER / f"{task}_test.csv", index=False)
