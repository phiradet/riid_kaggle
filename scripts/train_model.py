import os
import pickle
from multiprocessing import cpu_count

import torch
import fire
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from libs.dataset import RiidDataset, get_data_loader
from libs.model import Predictor


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def main(data_root_dir: str, batch_size: int, gpus: int = 1, save_top_k: int = 5, max_epochs: int = 10):

    train_dataset = RiidDataset(data_root_dir=data_root_dir, split="train")
    train_loader = get_data_loader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=cpu_count())

    val_dataset = RiidDataset(data_root_dir=data_root_dir, split="val")
    val_loader = get_data_loader(dataset=val_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=cpu_count())

    bundle_id_idx = pickle.load(open(os.path.join(data_root_dir, "indexes/bundle_id_idx.pickle"), "rb"))
    part_idx = pickle.load(open(os.path.join(data_root_dir, "indexes/part_idx.pickle"), "rb"))
    content_id_idx = pickle.load(open(os.path.join(data_root_dir, "indexes/content_id_idx.pickle"), "rb"))
    type_idx = pickle.load(open(os.path.join(data_root_dir, "indexes/type_idx.pickle"), "rb"))

    config = dict(content_id_size=len(content_id_idx) + 1,
                  content_id_dim=256,
                  bundle_id_size=len(bundle_id_idx) + 1,
                  bundle_id_dim=128,
                  feature_dim=204,
                  lstm_hidden_dim=512,
                  lstm_num_layers=2,
                  lstm_dropout=0.3,
                  emb_dropout=0.3,
                  output_dropout=0.3,
                  layer_norm=True,
                  lr=0.05)

    model = Predictor(**config)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total param: {total_params:,}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total learnable param: {total_params:,}")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./logs',
        save_top_k=save_top_k,
        mode='min')

    logger = TensorBoardLogger(save_dir="./tensorboard_logs", name='riid')

    trainer = pl.Trainer(max_epochs=max_epochs,
                         callbacks=[checkpoint_callback],
                         logger=logger,
                         gpus=gpus)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    fire.Fire(main)
