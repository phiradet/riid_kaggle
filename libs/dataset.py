import os
import glob
from typing import *

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def sort_batch_by_length(seq_lens: torch.Tensor):
    sorted_sequence_lengths, permutation_index = seq_lens.sort(0, descending=True)

    index_range = torch.arange(0, len(seq_lens), device=seq_lens.device)
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_sequence_lengths, restoration_indices, permutation_index


def collate_fn(instances: List[Dict[str, torch.tensor]],
               batch_first: bool = True,
               max_len: int = 512) -> Dict[str, torch.tensor]:

    seq_lens = [len(i["y"]) for i in instances]
    seq_max_len = min(max_len, max(seq_lens))

    sorted_seq_lengths, restoration_indices, permutation_index = sort_batch_by_length(torch.tensor(seq_lens,
                                                                                                   dtype=torch.long))

    out = {}
    for k in instances[0].keys():
        if instances[0][k].dim() > 0:
            _tensors = [i[k][:seq_max_len] for i in instances]
            padded_tensor = pad_sequence(_tensors, batch_first=batch_first)
        else:
            _tensors = [i[k] for i in instances]
            padded_tensor = torch.stack(_tensors, dim=0)

        sorted_padded_tensor = padded_tensor.index_select(0, permutation_index)
        out[k] = sorted_padded_tensor

    out["mask"] = (out["content_id"] != 0).to(dtype=torch.uint8)
    out["restoration_indices"] = restoration_indices

    return out


def get_data_loader(**kwargs):
    return DataLoader(collate_fn=collate_fn, **kwargs)


class RiidDataset(Dataset):

    def __init__(self, data_root_dir: str, split: str):
        self.data_root_dir = data_root_dir
        self.split = split

    @property
    def indexes_dir(self):
        return os.path.join(self.data_root_dir, "indexes")

    @property
    def instances_dir(self):
        return os.path.join(self.data_root_dir, self.split)

    @property
    def file_list(self):
        return list(sorted(glob.glob(os.path.join(self.instances_dir, "*.pth"))))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return torch.load(self.file_list[idx])
