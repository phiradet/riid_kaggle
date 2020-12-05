import os
import glob
from multiprocessing import Pool, cpu_count
from functools import partial

import fire
import torch


def _process_one_file(src_path, dst_dir):
    data = torch.load(src_path)
    for k in ["y", "feature"]:
        data[k] = data[k].to_dense()

    basename = os.path.basename(src_path)
    torch.save(data, os.path.join(dst_dir, basename))


def main(src_dir: str, dst_dir: str):
    all_files = list(sorted(glob.glob(os.path.join(src_dir, "*pth"))))

    with Pool(cpu_count()) as p:
        f = partial(_process_one_file, dst_dir=dst_dir)
        p.map(f, all_files)


if __name__ == "__main__":
    fire.Fire(main)
