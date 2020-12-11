from typing import *

import torch
import pandas as pd


def get_content_adjacency_matrix(content_df: pd.DataFrame,
                                 content_id_idx: Dict[int, int],
                                 no_question_lecture_link: bool = True,
                                 no_self_loop: bool = True) -> torch.tensor:
    # edge = overlapping tags
    content_df["idx"] = content_df["content_id"].apply(lambda id: content_id_idx[id])
    content_count = len(content_id_idx)

    # +1 for padding index
    adj_mat = torch.zeros((content_count + 1, content_count + 1),
                          dtype=torch.float,
                          requires_grad=False)

    exploded_content_df = content_df[["idx", "tags"]].explode("tags")
    exploded_content_df = exploded_content_df[exploded_content_df["tags"].notnull()]
    exploded_content_df["tags"] = exploded_content_df["tags"].astype(int)

    edge_indices = exploded_content_df \
        .merge(exploded_content_df, on="tags")

    if no_question_lecture_link:

        def get_sign(num):
            if num >= 0:
                return 1
            else:
                return -1

        same_content_type_mask = edge_indices["idx_x"].apply(get_sign) == edge_indices["idx_y"].apply(get_sign)
        edge_indices = edge_indices[same_content_type_mask]

    if no_self_loop:
        no_self_loop_mask = edge_indices["idx_x"] != edge_indices["idx_y"]
        edge_indices = edge_indices[no_self_loop_mask]

    edge_indices = edge_indices[["idx_x", "idx_y"]].drop_duplicates().values

    adj_mat[edge_indices[:, 0], edge_indices[:, 1]] = 1.
    adj_mat[edge_indices[:, 1], edge_indices[:, 0]] = 1.

    return adj_mat
