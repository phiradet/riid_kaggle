from typing import *

import torch
import pandas as pd


def get_content_adjacency_matrix(content_df: pd.DataFrame,
                                 content_id_idx: Dict[int, int],
                                 no_question_lecture_link: bool = True,
                                 no_self_loop: bool = True,
                                 same_part_only: bool = True) -> torch.tensor:
    # edge = overlapping tags
    content_df["idx"] = content_df["content_id"].apply(lambda id: content_id_idx[id])
    content_count = len(content_id_idx)

    # +1 for padding index
    adj_mat = torch.zeros((content_count + 1, content_count + 1),
                          dtype=torch.float,
                          requires_grad=False)

    key_col = "_key_col"
    if same_part_only:
        content_df[key_col] = content_df \
            .apply(lambda r: [f"{t}_{r['part']}" for t in r["tags"]], axis=1)
    else:
        content_df[key_col] = content_df["tags"]

    exploded_content_df = content_df[["idx", key_col]].explode(key_col)
    exploded_content_df = exploded_content_df[exploded_content_df[key_col].notnull()]
    exploded_content_df[key_col] = exploded_content_df[key_col].astype(str)

    edge_indices = exploded_content_df \
        .merge(exploded_content_df, on=key_col)

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

    edge_weights = edge_indices.groupby(["idx_x", "idx_y"]).size().reset_index(name="weight").values
    weights = torch.tensor(edge_weights[:, 2], dtype=torch.float)

    adj_mat[edge_weights[:, 0], edge_weights[:, 1]] = weights
    adj_mat[edge_weights[:, 1], edge_weights[:, 0]] = weights

    adj_mat -= adj_mat.min(1, keepdim=True)[0]
    adj_mat /= torch.clamp(adj_mat.max(1, keepdim=True)[0], min=1e-8)

    return adj_mat
