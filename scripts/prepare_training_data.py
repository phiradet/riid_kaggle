import os
import pickle
import numpy as np
from typing import *
from multiprocessing import cpu_count
from functools import partial

import torch
import fire
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar


NPARTITIONS = cpu_count() - 1


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


def get_mapping(unique_items: List[Any], not_null: bool = True, offset: int = 0) -> Dict[Any, int]:
    if not_null:
        unique_items = [i for i in unique_items if pd.notnull(i)]
    return {item: ind+offset for ind, item in enumerate(sorted(unique_items))}


def get_row(row: pd.DataFrame, key: str, default_val: Optional[Any] = None) -> pd.Series:
    if row[key].isnull().sum() == 0:
        return row[key]
    elif default_val is not None:
        return row[key].fillna(default_val)
    else:
        try:
            row_mode = row[key].mode().iloc[0]
        except IndexError as e:
            if default_val is None:
                print("\nCannot find null in")
                print(row[key])

                raise e
            row_mode = default_val

        return row[key].fillna(row_mode)


def save_data(rows: pd.DataFrame,
              output_dir: str,
              split: str,
              part_idx: Dict[int, int],
              type_idx: Dict[str, int],
              bundle_id_idx: Dict[int, int],
              content_id_idx: Dict[int, int]):

    user_id = rows["_user_id"].iloc[0]

    out_split_dir = os.path.join(output_dir, split)
    if not os.path.exists(out_split_dir):
        os.mkdir(out_split_dir)
    output_path = os.path.join(out_split_dir, f"{split}_{user_id}.pth")

    rows = rows.sort_values("timestamp").reset_index(drop=True)

    y_vec = get_row(rows, "answered_correctly").values
    y_vec = torch.tensor(y_vec, dtype=torch.long)

    content_id_vec = get_row(rows, "content_id").apply(lambda x: content_id_idx[x]).values
    content_id_vec = torch.tensor(content_id_vec, dtype=torch.long)

    bundle_id_vec = get_row(rows, "bundle_id").apply(lambda x: bundle_id_idx[x]).values
    bundle_id_vec = torch.tensor(bundle_id_vec, dtype=torch.long)

    container_id_series = get_row(rows, "task_container_id")
    task_container_vec = (container_id_series == container_id_series.shift(1)).values.astype(np.float)

    part_code_series = get_row(rows, "part").replace(part_idx)
    part_mat = np.eye(len(part_idx), dtype=np.float).take(part_code_series, axis=0)

    tag_code_idx_mat = rows["tags"].explode().dropna().reset_index().values.astype(np.int)
    tag_code_mat = np.zeros((len(rows), 188), dtype=np.float)
    tag_code_mat[tag_code_idx_mat[:, 0], tag_code_idx_mat[:, 1]] = 1

    type_code_series = get_row(rows, "type_of").replace(type_idx)
    type_mat = np.eye(len(type_idx), dtype=np.float).take(type_code_series, axis=0).astype(np.float)

    ms_15days = 86400 * 1000 * 15
    rows["time_diff"] = (get_row(rows, "timestamp") - get_row(rows, "timestamp").shift(1)).fillna(0.0)
    rows["time_diff"].values[rows["time_diff"] > ms_15days] = ms_15days
    time_diff_vec = (rows["time_diff"].values.astype(np.float) - 8265550.253) / 76321386.252

    rows["prior_question_elapsed_time"] = (rows["prior_question_elapsed_time"].fillna(0) - 25423.844) / 19948.146
    elapsed_time_vec = rows["prior_question_elapsed_time"].values.astype(np.float)

    prior_question_had_explanation_vec = get_row(rows,
                                                 "prior_question_had_explanation",
                                                 default_val=False).values.astype(np.float)

    feature_mat = np.concatenate([np.expand_dims(task_container_vec, axis=1),
                                  part_mat,
                                  tag_code_mat,
                                  type_mat,
                                  np.expand_dims(time_diff_vec, axis=1),
                                  np.expand_dims(elapsed_time_vec, axis=1),
                                  np.expand_dims(prior_question_had_explanation_vec, axis=1)], axis=1)
    feature_mat = torch.tensor(feature_mat, dtype=torch.float)

    assert torch.isnan(y_vec).sum() == 0
    assert torch.isnan(content_id_vec).sum() == 0
    assert torch.isnan(bundle_id_vec).sum() == 0
    assert torch.isnan(feature_mat).sum() == 0

    instance = {
        "y": y_vec,
        "user_id": torch.tensor(user_id, dtype=torch.long),
        "content_id": content_id_vec,
        "bundle_id": bundle_id_vec,
        "feature": feature_mat
    }
    torch.save(instance, output_path)


def main(data_path: str, questions_path: str, lectures_path: str, output_dir: str,
         use_existing_idx: bool = False, split: str = "train"):
    content_df = read_contents(questions_path, lectures_path)
    print(content_df.dtypes)

    indexes = ["part_idx", "type_idx", "bundle_id_idx", "content_id_idx"]

    idx_map_dir = os.path.join(output_dir, "indexes")
    if not os.path.exists(idx_map_dir):
        print(f"mkdir -p {idx_map_dir}")
        os.makedirs(idx_map_dir)

    if use_existing_idx:
        part_idx = pickle.load(open(os.path.join(idx_map_dir, f"part_idx.pickle"), "rb"))
        type_idx = pickle.load(open(os.path.join(idx_map_dir, f"type_idx.pickle"), "rb"))
        bundle_id_idx = pickle.load(open(os.path.join(idx_map_dir, f"bundle_id_idx.pickle"), "rb"))
        content_id_idx = pickle.load(open(os.path.join(idx_map_dir, f"content_id_idx.pickle"), "rb"))
    else:
        part_idx = get_mapping(content_df["part"].drop_duplicates().tolist())
        type_idx = get_mapping(content_df["type_of"].drop_duplicates().tolist())
        bundle_id_idx = get_mapping(content_df["bundle_id"].drop_duplicates().tolist(), offset=1)
        content_id_idx = get_mapping(content_df["content_id"].drop_duplicates().tolist(), offset=1)

        for idx in indexes:
            idx_out_path = os.path.join(idx_map_dir, f"{idx}.pickle")
            with open(idx_out_path, "wb") as fp:
                pickle.dump(eval(idx), fp)

    with ProgressBar():
        # data_df = read_data(data_path, npartitions=NPARTITIONS).merge(content_df, how="left", on="content_id")
        data_df = read_data(data_path, npartitions=NPARTITIONS) \
            .merge(content_df, how="left", on="content_id")
        print("data_df", data_df.shape)
        print(data_df.columns)

        _save_data = partial(save_data,
                             output_dir=output_dir,
                             split=split,
                             part_idx=part_idx,
                             type_idx=type_idx,
                             bundle_id_idx=bundle_id_idx,
                             content_id_idx=content_id_idx)
        data_df["_user_id"] = data_df["user_id"]
        data_df \
            .groupby("user_id") \
            .apply(_save_data,
                   meta=pd.Series([None])) \
            .compute()


if __name__ == "__main__":
    fire.Fire(main)
