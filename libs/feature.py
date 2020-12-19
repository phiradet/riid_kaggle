from typing import *

import torch
import numpy as np
import pandas as pd


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


def extract_feature(rows: pd.DataFrame,
                    part_idx: Dict[int, int],
                    type_idx: Dict[str, int],
                    bundle_id_idx: Dict[int, int],
                    content_id_idx: Dict[int, int],
                    seq_len: Optional[int] = 1024,
                    to_sparse: bool = True):

    user_id = rows["user_id"].iloc[0]

    if seq_len is not None:
        # use tail, then we can capture latest state
        rows = rows.tail(seq_len)

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

    timestamp_vec = get_row(rows, "timestamp").values
    timestamp_vec = torch.tensor(timestamp_vec, dtype=torch.long)

    assert torch.isnan(content_id_vec).sum() == 0
    assert torch.isnan(bundle_id_vec).sum() == 0
    assert torch.isnan(feature_mat).sum() == 0
    assert torch.isnan(timestamp_vec).sum() == 0

    if to_sparse:
        feature_mat = feature_mat.to_sparse()

    user_id = torch.tensor(user_id, dtype=torch.long)

    instance = {
        "user_id": user_id,
        "content_id": content_id_vec,
        "bundle_id": bundle_id_vec,
        "feature": feature_mat,
        "timestamp": timestamp_vec,
    }

    if "answered_correctly" in rows:
        y_vec = get_row(rows, "answered_correctly").values
        y_vec = torch.tensor(y_vec, dtype=torch.long)
        assert torch.isnan(y_vec).sum() == 0

        if to_sparse:
            instance["y"] = y_vec.to_sparse()
        else:
            instance["y"] = y_vec

    if not to_sparse:
        is_question_mask = (rows["content_type_id"] == 0).values
        is_question_mask = torch.tensor(is_question_mask, dtype=torch.bool)
        instance["is_question_mask"] = is_question_mask

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for k, v in instance.items():
        instance[k] = v.to(device)

    return instance
