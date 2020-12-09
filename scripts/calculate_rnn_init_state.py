import os
import glob
import json
import pickle
from multiprocessing import cpu_count

import torch
import fire
from tqdm.auto import tqdm

from libs.dataset import MultiRiidDataset, get_data_loader
from libs.model import Predictor
from libs.utils.io import load_state_dict


def step(batch, model, verbose=False):
    actual = batch["y"]  # (batch, seq)
    seq_len_mask = batch["seq_len_mask"]  # (batch, seq)

    batch_size, seq_len = actual.shape
    actual = actual.view(batch_size * seq_len).float()
    actual[actual < 0] = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    content_id = batch["content_id"].to(device)
    bundle_id = batch["bundle_id"].to(device)
    feature = batch["feature"].to(device)
    user_id = batch["user_id"].to(device)

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)

    dataset = MultiRiidDataset(data_root_dir, ["train_dense"], verbose)
    data_loader = get_data_loader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=cpu_count(),
                                  max_len=512)

    model_config = json.load(open(model_config_file))
    model = Predictor(**model_config).eval()

    if torch.cuda.is_available():
        model = model.to(device) 

    state_dict = load_state_dict(checkpoint_dir)
    model.load_state_dict(state_dict)

    user_ids = dataset.user_ids
    user_id_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}

    with open(os.path.join(output_dir, "user_id_idx.pickle"), "wb") as fp:
        pickle.dump(user_id_idx, fp)

    user_ids = []

    for idx, batch in tqdm(enumerate(data_loader)):
        h_t, c_t = step(batch, model, verbose=verbose)

        torch.save(h_t, os.path.join(output_dir, f"h_t.part_{idx:08d}.pth"))
        torch.save(h_t, os.path.join(output_dir, f"c_t.part_{idx:08d}.pth"))

        user_ids.append(batch["user_id"])

    h_t_tensors = [torch.load(f) for f in sorted(glob.glob(os.path.join(output_dir, "h_t.part_*.pth")))]
    c_t_tensors = [torch.load(f) for f in sorted(glob.glob(os.path.join(output_dir, "c_t.part_*.pth")))]

    h_t_tensors = torch.cat(h_t_tensors, dim=1)
    c_t_tensors = torch.cat(c_t_tensors, dim=1)
    user_ids = torch.cat(user_ids, dim=0)

    print("h_t_tensors", h_t_tensors.shape)
    print("c_t_tensors", c_t_tensors.shape)

    torch.save(h_t_tensors, os.path.join(output_dir, "h_t.pth"))
    torch.save(c_t_tensors, os.path.join(output_dir, "c_t.pth"))
    torch.save(user_ids, os.path.join(output_dir, "user_ids.pth"))


if __name__ == "__main__":
    fire.Fire(main)
