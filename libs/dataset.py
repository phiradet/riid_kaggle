import os
import glob
from typing import *
from functools import partial
from collections import defaultdict

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
               max_len: int = 512,
               is_sparse_tensor: bool = False) -> Dict[str, torch.tensor]:

    seq_lens = [len(i["y"]) for i in instances]
    seq_max_len = min(max_len, max(seq_lens))

    out: Dict[str, torch.Tensor] = {}
    for k in instances[0].keys():
        if instances[0][k].dim() > 0:
            if is_sparse_tensor and (k == "feature" or k == "y"):
                _tensors = [i[k].to_dense()[:seq_max_len] for i in instances]
            else:
                _tensors = [i[k][:seq_max_len] for i in instances]
            padded_tensor = pad_sequence(_tensors, batch_first=batch_first)
        else:
            _tensors = [i[k] for i in instances]
            padded_tensor = torch.stack(_tensors, dim=0)

        out[k] = padded_tensor

    out["seq_len_mask"] = (out["content_id"] != 0).to(dtype=torch.uint8)
    out["question_mask"] = (out["y"] >= 0).to(dtype=torch.uint8)

    return out


def get_data_loader(**kwargs):
    _collate_fn = partial(collate_fn,
                          max_len=kwargs.get("max_len", 512),
                          is_sparse_tensor=kwargs.get("is_sparse_tensor", False))

    if "max_len" in kwargs:
        del kwargs["max_len"]

    if "is_sparse_tensor" in kwargs:
        del kwargs["is_sparse_tensor"]
        
    return DataLoader(collate_fn=_collate_fn, **kwargs)


class RiidDataset(Dataset):

    def __init__(self, data_root_dir: str, split: str):
        self.data_root_dir = data_root_dir
        self.split = split

        self.indexes_dir = os.path.join(self.data_root_dir, "indexes")
        self.instances_dir = os.path.join(self.data_root_dir, self.split)
        self.file_list = list(sorted(glob.glob(os.path.join(self.instances_dir, "*.pth"))))

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx) -> Dict[str, torch.tensor]:
        return torch.load(self.file_list[idx])


class MultiRiidDataset(Dataset):

    def __init__(self, data_root_dir: str, splits: List[str], verbose: bool = False):
        self.data_root_dir = data_root_dir
        self.splits = splits
        self.verbose = verbose

        self.indexes_dir = os.path.join(self.data_root_dir, "indexes")
        self.instance_dirs = [os.path.join(self.data_root_dir, s) for s in self.splits]

        self.file_list = self.get_file_list()
        self.user_ids = list(sorted(self.file_list.keys()))

        if self.verbose:
            print(self.file_list)
            print(self.user_ids)

        print("Number of users", len(self.user_ids))

    @staticmethod
    def get_user_ids(file_path) -> int:
        filename, _ = os.path.splitext(os.path.basename(file_path))
        split, user_id = filename.split("_")

        return int(user_id)

    def get_file_list(self) -> Dict[str, List[str]]:
        output = defaultdict(list)
        for dir in self.instance_dirs:
            for file_path in glob.glob(os.path.join(dir, "*.pth")):
                user_id = self.__class__.get_user_ids(file_path)
                output[user_id].append(file_path)
        return dict(output)

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx) -> Dict[str, torch.tensor]:
        user_id = self.user_ids[idx]

        instances: List[Dict[str, torch.tensor]] = []
        for file in self.file_list[user_id]:
            instances.append(torch.load(file))

        collated_instance = {}

        if self.verbose:
            print(instances[0]["user_id"])

        for k in instances[0].keys():
            if k == "user_id":
                collated_instance[k] = instances[0]["user_id"]
            else:
                collated_instance[k] = torch.cat([i[k] for i in instances], dim=0)

            if self.verbose:
                print(k, collated_instance[k].shape)

        if self.verbose:
            print()

        return collated_instance
