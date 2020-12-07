from typing import *

import pandas as pd
import dask.dataframe as dd


def read_data(data_path: str, npartitions: int) -> dd.DataFrame:
    df: dd.DataFrame = dd.read_parquet(data_path).repartition(npartitions)

    def format_content_id(row: pd.Series) -> int:
        if row["content_type_id"]:  # lecture
            return -1 * row["content_id"]
        else:
            return row["content_id"]

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