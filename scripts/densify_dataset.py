import os
import glob

import fire
import torch
from tqdm.auto import tqdm


def main(src_dir: str, dst_dir: str):
    all_files = list(sorted(glob.glob(os.path.join(src_dir, "*pth"))))

    for f in tqdm(all_files):
        data = torch.load(f)

        for k in ["y", "feature"]:
            data[k] = data[k].to_dense()

        basename = os.path.basename(f)
        torch.save(data, os.path.join(dst_dir, basename))


if __name__ == "__main__":
    fire.Fire(main)
