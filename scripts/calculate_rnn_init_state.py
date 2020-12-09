import os
import json
import pickle
from typing import *
from multiprocessing import cpu_count

import torch
import fire

from libs.dataset import MultiRiidDataset, get_data_loader
from libs.model import Predictor
from libs.utils.io import load_state_dict


def step(batch, model, verbose=False):
    actual = batch["y"]  # (batch, seq)
    seq_len_mask = batch["seq_len_mask"]  # (batch, seq)

    batch_size, seq_len = actual.shape
    actual = actual.view(batch_size * seq_len).float()
    actual[actual < 0] = 0

    content_id = batch["content_id"]
    bundle_id = batch["bundle_id"]
    feature = batch["feature"]
    user_id = batch["user_id"]

    if verbose:
        print(user_id)
        print(feature.shape)
        print()

    with torch.no_grad():
        pred, (h_t, c_t) = model(content_id=content_id,
                                 bundle_id=bundle_id,
                                 feature=feature,
                                 user_id=user_id,
                                 mask=seq_len_mask)

    return h_t, c_t


def main(model_config_file: str,
         checkpoint_dir: str,
         output_dir: str,
         data_root_dir: str,
         batch_size: int = 128,
         verbose: bool = False):
    dataset = MultiRiidDataset(data_root_dir, ["train_dense", "val_dense"], verbose)
    data_loader = get_data_loader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=cpu_count(),
                                  max_len=512)

    model_config = json.load(open(model_config_file))
    model = Predictor(**model_config).eval()

    state_dict = load_state_dict(checkpoint_dir)
    model.load_state_dict(state_dict)

    user_ids = dataset.user_ids
    user_id_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}

    with open(os.path.join(output_dir, "user_id_idx.pickle"), "wb") as fp:
        pickle.dump(user_id_idx, fp)

    h_t_tensors = []
    c_t_tensors = []
    user_ids = []

    for batch in data_loader:
        h_t, c_t = step(batch, model, verbose=verbose)
        h_t_tensors.append(h_t)
        c_t_tensors.append(c_t)
        user_ids.append(batch["user_id"])

    h_t_tensors = torch.cat(h_t_tensors, dim=1)
    c_t_tensors = torch.cat(c_t_tensors, dim=1)

    print("h_t_tensors", h_t_tensors.shape)
    print("c_t_tensors", c_t_tensors.shape)

    torch.save(h_t_tensors, os.path.join(output_dir, "h_t.pth"))
    torch.save(c_t_tensors, os.path.join(output_dir, "c_t.pth"))


if __name__ == "__main__":
    fire.Fire(main)
