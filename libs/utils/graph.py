from typing import *
from math import copysign

import torch
from tqdm.auto import tqdm

from libs.utils.io import read_contents


def get_content_adjacency_matrix(questions_path: str,
                                 lectures_path: str,
                                 content_id_idx: Dict[int, int],
                                 no_question_lecture_link: bool = True) -> torch.tensor:
    # edge = overlapping tags
    content_df = read_contents(questions_path=questions_path,
                               lectures_path=lectures_path)
    content_df["idx"] = content_df["content_id"].apply(lambda id: content_id_idx[id])
    content_count = len(content_id_idx)

    # +1 for padding index
    adj_mat = torch.zeros((content_count + 1, content_count + 1),
                          dtype=torch.float,
                          requires_grad=False)

    exploded_content_df = content_df[["idx", "tags"]].explode("tags")
    edge_indices = exploded_content_df \
        .merge(exploded_content_df, on="tags")[["idx_x", "idx_y"]] \
        .drop_duplicates() \
        .values

    adj_mat[edge_indices[:, 0], edge_indices[:, 1]] = 1.
    adj_mat[edge_indices[:, 1], edge_indices[:, 0]] = 1.

    return adj_mat
