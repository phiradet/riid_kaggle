import os
import glob
from typing import *

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def collate_fn(instances: List[Dict[str, torch.tensor]],
               batch_first: bool = True,
               max_len: int = 512) -> Dict[str, torch.tensor]:

    seq_lens = [len(i["y"]) for i in instances]
    seq_max_len = min(max_len, max(seq_lens))

    out = {}
    for k in instances[0].keys():
        if instances[0][k].dim() > 0:
            if k in ["y", "feature"]:
                _tensors = [i[k].to_dense()[:seq_max_len] for i in instances]
            else:
                _tensors = [i[k][:seq_max_len] for i in instances]

            padded_tensor = pad_sequence(_tensors, batch_first=batch_first)
        else:
            _tensors = [i[k] for i in instances]
            padded_tensor = torch.stack(_tensors, dim=0)

        out[k] = padded_tensor

    out["seq_len"] = torch.tensor(seq_lens, dtype=torch.int)
    # out["seq_len_mask"] = (out["content_id"] != 0).to(dtype=torch.uint8)
    # out["question_mask"] = (out["y"] >= 0).to(dtype=torch.uint8)

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
