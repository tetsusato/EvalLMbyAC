import json
import os

import datasets
import pandas as pd

from src.llmjp.about import Tasks
#from src.display.formatting import has_no_nan_values, make_clickable_model, make_clickable_model_with_shot
from src.llmjp.utils import AutoEvalColumn, EvalQueueColumn

# The values of these columns are in the range of 0-100
# We normalize them to 0-1
COLUMNS_TO_NORMALIZE = [
    "ALT E to J BLEU",
    "ALT J to E BLEU",
    "WikiCorpus E to J BLEU",
    "WikiCorpus J to E BLEU",
    "XL-Sum JA BLEU",
    "XL-Sum ROUGE1",
    "XL-Sum ROUGE2",
    "XL-Sum ROUGE-Lsum",
]

def has_no_nan_values(df, columns):
    return df[columns].notna().all(axis=1)

def get_leaderboard_df(contents_repo: str, cols: list[str], benchmark_cols: list[str]) -> pd.DataFrame:
    df = datasets.load_dataset(contents_repo, split="train").to_pandas()
    # df["Model"] = df["model"].map(make_clickable_model)
    df["Model"] = df["model"]
    #df["Model"] = df.apply(lambda x: make_clickable_model_with_shot(x["model"], x["num_few_shot"]), axis=1)
    df["T"] = df["model_type"].map(lambda x: x.split(":")[0].strip())
    df = df.rename(columns={task.value.metric: task.value.col_name for task in Tasks})
    df = df.rename(
        columns={
            "architecture": "Architecture",
            "weight_type": "Weight type",
            "precision": "Precision",
            "license": "Hub License",
            "params": "#Params (B)",
            "likes": "Hub ❤️",
            "revision": "Revision",
            "num_few_shot": "Few-shot",
            "add_special_tokens": "Add Special Tokens",
            "llm_jp_eval_version": "llm-jp-eval version",
            "vllm_version": "vllm version",
            "model_type": "Type",
            "model": "model_name_for_query",
        }
    )

    # Add a row ID column
    df[AutoEvalColumn.row_id.name] = range(len(df))

    # Normalize the columns
    available_columns_to_normalize = [col for col in COLUMNS_TO_NORMALIZE if col in df.columns]
    df[available_columns_to_normalize] = df[available_columns_to_normalize] / 100

    df = df.sort_values(by=[AutoEvalColumn.AVG.name], ascending=False)
    df = df[cols].round(decimals=4)

    # filter out if any of the benchmarks have not been produced
    df = df[has_no_nan_values(df, benchmark_cols)]

    return df


def get_evaluation_queue_df(save_path: str, cols: list[str]) -> list[pd.DataFrame]:
    """Creates the different dataframes for the evaluation queues requestes"""
    entries = [entry for entry in os.listdir(save_path) if not entry.startswith(".")]
    all_evals = []

    for entry in entries:
        if ".json" in entry:
            file_path = os.path.join(save_path, entry)
            with open(file_path) as fp:
                data = json.load(fp)

            # data[EvalQueueColumn.model.name] = make_clickable_model(data["model"])
            data[EvalQueueColumn.model.name] = make_clickable_model_with_shot(
                data["model"],
                data["num_few_shot"],  # num_few_shotは必ず存在するため、直接参照
            )
            data[EvalQueueColumn.revision.name] = data.get("revision", "main")

            all_evals.append(data)
        elif ".md" not in entry:
            # this is a folder
            sub_entries = [e for e in os.listdir(f"{save_path}/{entry}") if not e.startswith(".")]
            for sub_entry in sub_entries:
                file_path = os.path.join(save_path, entry, sub_entry)
                with open(file_path) as fp:
                    data = json.load(fp)

                data[EvalQueueColumn.model.name] = make_clickable_model(data["model"])
                data[EvalQueueColumn.revision.name] = data.get("revision", "main")
                all_evals.append(data)

    pending_list = [e for e in all_evals if e["status"] in ["PENDING", "RERUN"]]
    running_list = [e for e in all_evals if e["status"] == "RUNNING"]
    finished_list = [e for e in all_evals if e["status"].startswith("FINISHED") or e["status"] == "PENDING_NEW_EVAL"]
    failed_list = [e for e in all_evals if e["status"] == "FAILED"]
    df_pending = pd.DataFrame.from_records(pending_list, columns=cols)
    df_running = pd.DataFrame.from_records(running_list, columns=cols)
    df_finished = pd.DataFrame.from_records(finished_list, columns=cols)
    df_failed = pd.DataFrame.from_records(failed_list, columns=cols)
    return df_finished[cols], df_running[cols], df_pending[cols], df_failed[cols]
