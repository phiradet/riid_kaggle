import torch
import pandas as pd

from libs.utils.graph import get_content_adjacency_matrix


def test_get_content_adjacency():

    content_df = pd.DataFrame({
        "content_id": [0, 1, 2, 3],
        "tags": [
            [1, 2, 3],
            [],
            [3, 4],
            [4, 5]
        ]
    })

    # index 0 is for padding
    content_id_idx = {0: 1, 1: 2, 2: 3, 3: 4}

    output = get_content_adjacency_matrix(content_df=content_df,
                                          content_id_idx=content_id_idx)
    expected = torch.zeros(len(content_id_idx)+1, len(content_id_idx)+1, dtype=torch.float)

    print(output.shape)
    print(output)
    print(expected.shape)

    assert torch.all(expected.eq(output))
