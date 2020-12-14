import os
import glob
from typing import *

import torch
import pandas as pd
import dask.dataframe as dd


def format_content_id(row: pd.Series) -> int:
    if row["content_type_id"]:  # lecture
        return -1 * row["content_id"]
    else:
        return row["content_id"]


def read_data(data_path: str, npartitions: int) -> dd.DataFrame:
    df: dd.DataFrame = dd.read_parquet(data_path).repartition(npartitions)

    df["content_id"] = df.apply(format_content_id, meta=pd.Series([1, 2, 3]), axis=1)
    return df


def read_contents(questions_path: str, lectures_path: str) -> pd.DataFrame:
    dtypes = {
        "question_id": "int16",
        "bundle_id": "int64",
        "correct_answer": "int16",
        "part": "int16",
        "tags": "object"
    }
    questions = pd.read_csv(questions_path, dtype=dtypes)
    questions["type_of"] = "question"
    questions["content_id"] = questions["question_id"]
    questions["tags"] = questions["tags"].apply(lambda x: [int(i) for i in x.split(" ")] if isinstance(x, str) else [])

    dtypes = {
        "lecture_id": "int16",
        "tag": "object",
        "part": "int16",
        "type_of": "object"
    }
    lectures = pd.read_csv(lectures_path, dtype=dtypes)
    lectures["content_id"] = -1 * lectures["lecture_id"]
    lectures["tags"] = lectures["tag"].apply(lambda t: [int(t)])
    lectures["bundle_id"] = -1

    cols = ["content_id", "tags", "part", "type_of", "bundle_id"]
    return pd.concat([questions[cols], lectures[cols]])


def load_state_dict(checkpoint_dir: str):

    if os.path.isdir(checkpoint_dir):
        *_, checkpoint_file = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch=*.ckpt")))
    else:
        checkpoint_file = checkpoint_dir

    print("Load weight from", checkpoint_file)
    if torch.cuda.is_available():
        state_dict = torch.load(checkpoint_file)["state_dict"]
    else:
        state_dict = torch.load(checkpoint_file, map_location=torch.device('cpu'))["state_dict"]

    if "hidden2logit.weight" in state_dict:
        state_dict["hidden2logit.0.weight"] = state_dict["hidden2logit.weight"]
        state_dict["hidden2logit.0.bias"] = state_dict["hidden2logit.bias"]

        del state_dict["hidden2logit.weight"]
        del state_dict["hidden2logit.bias"]
    return state_dict
