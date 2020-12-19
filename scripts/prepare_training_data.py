import os
import pickle
from typing import *
from multiprocessing import cpu_count
from functools import partial

import torch
import fire
import pandas as pd
import dask.dataframe as dd

from libs.utils.io import read_data, read_contents
from libs.feature import extract_feature


NPARTITIONS = cpu_count()


def get_mapping(unique_items: List[Any], not_null: bool = True, offset: int = 0) -> Dict[Any, int]:
    if not_null:
        unique_items = [i for i in unique_items if pd.notnull(i)]
    return {item: ind+offset for ind, item in enumerate(sorted(unique_items))}


def save_data(rows: pd.DataFrame,
              output_dir: str,
              split: str,
              part_idx: Dict[int, int],
              type_idx: Dict[str, int],
              bundle_id_idx: Dict[int, int],
              content_id_idx: Dict[int, int],
              seq_len: Optional[int] = 512):

    user_id = rows["_user_id"].iloc[0]

    out_split_dir = os.path.join(output_dir, split)
    if not os.path.exists(out_split_dir):
        os.mkdir(out_split_dir)
    output_path = os.path.join(out_split_dir, f"{split}_{user_id}.pth")

    rows = rows.sort_values("timestamp").reset_index(drop=True)

    instance = extract_feature(rows=rows,
                               part_idx=part_idx,
                               type_idx=type_idx,
                               bundle_id_idx=bundle_id_idx,
                               content_id_idx=content_id_idx,
                               seq_len=seq_len)

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

    data_df: dd.DataFrame = read_data(data_path, npartitions=NPARTITIONS) \
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
    data_df["key"] = data_df["user_id"]
    print("Expected file count", len(data_df["key"].drop_duplicates()))
    data_df \
        .groupby("key", sort=False) \
        .apply(_save_data,
               meta=pd.Series([None])) \
        .compute()
    print("==== Done ====")


if __name__ == "__main__":
    fire.Fire(main)
