from collections import deque

from pathlib import Path
import os
import datetime
from openai import OpenAI
from dotenv import load_dotenv
import time
import datetime
import json
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
from transformers import AutoTokenizer

load_dotenv(dotenv_path="dataset_generation/CoT_dataset_generation/azure-openai.env")

import pandas as pd
import re

SYSTEM_PROMPT = """You are given a full reasoning trace and a sentence that was removed from the end of it.
The remaining (kept) text ends right before this removed sentence.

Determine whether the removed sentence introduces information that a reader could NOT
already straightforwardly figure out from the kept text alone.

Answer PASS if the removed sentence only:
- states the final true/false verdict
- aggregates, totals, or restates facts whose individual components were already
  established earlier in the kept text (e.g. if each city's country was stated
  individually earlier, a sentence that sums them into a count is still a PASS,
  even if that exact total was never stated verbatim before)

Answer FAIL if the removed sentence performs a new comparison, computation, or
determination whose result is not straightforwardly derivable from facts already
stated earlier in the kept text.

Reasoning trace (kept portion):
{kept_text}

Removed sentence:
{removed_chunk}

Answer with exactly one word: PASS or FAIL.
"""

AZURE_BATCH_DIR = Path("dataset_generation/CoT_dataset_generation/azure_batches")
AZURE_BATCH_DIR.mkdir(exist_ok=True)

AZURE_DEPLOYMENT_NAME = "o3"

OUTPUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "InformationLossCheckOutputFormat",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "has_new_information": {"type": "boolean"},
            },
            "required": ["has_new_information"],
            "additionalProperties": False,
        },
    },
}

TASKS = ["F0", "F1", "F2", "F3", "F4", "F5", "A1", "A2", "A3"]


def build_batch_request(custom_id: str, sample: pd.Series) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",  # fixed: needs the /v1/ prefix per the docs
        "body": {
            "model": AZURE_DEPLOYMENT_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.format(kept_text=sample["kept_text"], removed_chunk=sample["removed_chunk"])},
            ],
            "response_format": OUTPUT_SCHEMA,
        },
    }


def write_batch_jsonl(dataset: pd.DataFrame, task: str) -> Path:
    out_path = AZURE_BATCH_DIR / f"{task}_batch_input.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, row in dataset.iterrows():
            custom_id = f"{task}_{idx}"
            request = build_batch_request(custom_id, row)
            line = json.dumps(request, ensure_ascii=False)
            f.write(line)
            f.write("\n")   # written as a separate call, so it can't get silently dropped
    print(f"Wrote {len(dataset)} requests to {out_path}")
    return out_path


def remove_last_sentence(row):
    r = r"\."
    last_match = deque(re.finditer(r, row["extracted_statement_texts"]), maxlen=1)
    kept_text = row["extracted_statement_texts"][:last_match[0].start() + 1]
    removed_chunk = row["extracted_statement_texts"][last_match[0].start() + 1:]
    return kept_text, removed_chunk


if __name__ == "__main__":
    validation_data = pd.DataFrame({
        "task": [], "row_idx": [], "kept_text": [], "removed_chunk": [],
    })

    SAMPLES_PER_TASK = 200
    RANDOM_SEED = 42

    # Create the 200 x 9 (Tasks) = 1800 sampled validation set
    for TASK in TASKS:
        # Load train + test data frames
        task_df = pd.concat([
            pd.read_csv(f"datasets/CoT_datasets/filtered/{TASK}_filtered_test.csv"),
            pd.read_csv(f"datasets/CoT_datasets/filtered/{TASK}_filtered_train.csv"),
        ])
        task_df = task_df.sample(n=min(SAMPLES_PER_TASK, len(task_df)), random_state=RANDOM_SEED)
        task_df[["kept_text", "removed_chunk"]] = task_df.apply(remove_last_sentence, axis=1, result_type="expand")
        task_df["task"] = TASK
        task_df["row_idx"] = range(1, len(task_df) + 1)

        validation_data = pd.concat([validation_data, task_df], join="inner", ignore_index=True)

    # Create sample_id column
    validation_data["sample_id"] = range(1, len(validation_data) + 1)

    # Creating the Request File
    a1_train_batch_path = write_batch_jsonl(validation_data, task="validation")

    # Upload batch file
    client = OpenAI(
        base_url="https://student-hub-0122.openai.azure.com/openai/v1",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    # Upload a file with a purpose of "batch"
    file = client.files.create(
        file=open("dataset_generation/CoT_dataset_generation/azure_batches/validation_batch_input.jsonl", "rb"),
        purpose="batch",
        extra_body={"expires_after": {"seconds": 1209600, "anchor": "created_at"}}  # Optional you can set to a number between 1209600-2592000. This is equivalent to 14-30 days
    )

    print(file.model_dump_json(indent=2))
    print(f"File expiration: {datetime.datetime.fromtimestamp(file.expires_at) if file.expires_at is not None else 'Not set'}")

    file_id = file.id
    print(f"Uploaded file: {file_id}")

    # Submit a batch job with the file
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/chat/completions",  # While passing this parameter is required, the system will read your input file to determine if the chat completions or responses API is needed.
        completion_window="24h",
        # extra_body={"output_expires_after":{"seconds": 1209600, "anchor": "created_at"}} # Optional you can set to a number between 1209600-2592000. This is equivalent to 14-30 days
    )

    # Save batch ID for later use
    batch_id = batch_response.id
    print(batch_response.model_dump_json(indent=2))
    print(f"Batch job created: {batch_id}")

    # Tracking progress
    print("Polling for batch completion...")
    status = "validating"
    while status not in ("completed", "failed", "canceled"):
        time.sleep(60)
        batch_response = client.batches.retrieve(batch_id)
        status = batch_response.status
        print(f"{datetime.datetime.now()} Batch Id: {batch_id},  Status: {status}")

    if batch_response.status == "failed":
        for error in batch_response.errors.data:
            print(f"Error code {error.code} Message {error.message}")

    responses = []
    output_file_id = batch_response.output_file_id

    if not output_file_id:
        output_file_id = batch_response.error_file_id

    if output_file_id:
        file_response = client.files.content(output_file_id)
        raw_responses = file_response.text.strip().split('\n')

        for raw_response in raw_responses:
            json_response = json.loads(raw_response)
            responses.append(json_response)

    print(f"Retrieved {len(responses)} responses")

    total = 0
    leakage = 0
    loss_samples = []
    for response in responses:
        total += 1
        response_json = json.loads(response["response"]["body"]["choices"][0]["message"]["content"])
        if response_json["has_new_information"] == True:
            leakage += 1
            loss_samples.append(response["custom_id"])

    print(f"Percentage of samples with information loss: {100 * leakage / total}")

    # Save the flagged samples (with their kept_text/removed_chunk for review) to a .txt file
    loss_log_path = Path("information_loss_samples.txt")
    with open(loss_log_path, "w", encoding="utf-8") as f:
        f.write(f"Samples flagged with information loss: {leakage}/{total} ({100 * leakage / total:.1f}%)\n\n")
        for custom_id in loss_samples:
            row_pos = int(custom_id.split("_")[-1])
            row = validation_data.iloc[row_pos]
            f.write(f"=== custom_id={custom_id}  sample_id={row['sample_id']}  task={row['task']}  row_idx={row['row_idx']} ===\n")
            f.write(f"KEPT TEXT:\n{row['kept_text']}\n\n")
            f.write(f"REMOVED CHUNK:\n{row['removed_chunk']}\n\n")
            f.write("-" * 80 + "\n\n")
    print(f"Wrote {len(loss_samples)} flagged samples to {loss_log_path}")
