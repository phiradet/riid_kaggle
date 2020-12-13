import os
import pickle
from typing import *
from multiprocessing import cpu_count

import torch
import fire
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from libs.dataset import RiidDataset, get_data_loader
from libs.model import Predictor
from libs.utils.graph import get_content_adjacency_matrix
from libs.utils.io import read_contents


def main(data_root_dir: str,
         batch_size: int = 32,
         gpus: int = 1,
         save_top_k: int = 5,
         max_epochs: int = 10,
         max_len: int = 512,
         lstm_num_layers: int = 2,
         lstm_hidden_dim: int = 512,
         layer_norm: bool = True,
         lr: float = 1e-3,
         smoothness_alpha: float = 0.3,
         lstm_in_dim: int = 460,
         truncated_bptt_steps: Optional[int] = None,
         is_sparse_tensor: bool = False,
         train_split: str = "train_dense",
         val_split: str = "val_dense",
         encoder_type: str = "vanilla_lstm",
         emb_dropout: float = 0.1,
         output_dropout: float = 0.3,
         lstm_dropout: float = 0.1,
         content_id_dim: int = 256,
         optimizer: float = "adam",
         overfit_pct: float = 0.0,
         gradient_clip_val: float = 0.0,
         highway_connection: bool = False,
         track_grad_norm: int = -1,
         val_check_interval: float = 1.0,
         resume_from_checkpoint: Optional[str] = None,
         hidden2logit_num_layers: int = 1):

    print("data_root_dir", data_root_dir, type(data_root_dir))
    print("batch_size", batch_size, type(batch_size))
    print("gpus", gpus, type(gpus))
    print("save_top_k", save_top_k, type(save_top_k))
    print("max_epochs", max_epochs, type(max_epochs))
    print("layer_norm", layer_norm, type(layer_norm))

    train_dataset = RiidDataset(data_root_dir=data_root_dir, split=train_split)
    train_loader = get_data_loader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=cpu_count(),
                                   max_len=max_len,
                                   is_sparse_tensor=is_sparse_tensor)

    val_dataset = RiidDataset(data_root_dir=data_root_dir, split=val_split)
    val_loader = get_data_loader(dataset=val_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=cpu_count(),
                                 max_len=max_len,
                                 is_sparse_tensor=is_sparse_tensor)

    bundle_id_idx = pickle.load(open(os.path.join(data_root_dir, "indexes/bundle_id_idx.pickle"), "rb"))
    content_id_idx = pickle.load(open(os.path.join(data_root_dir, "indexes/content_id_idx.pickle"), "rb"))

    config = dict(content_id_size=len(content_id_idx) + 1,
                  content_id_dim=content_id_dim,
                  bundle_id_size=len(bundle_id_idx) + 1,
                  bundle_id_dim=128,
                  feature_dim=204,
                  lstm_in_dim=lstm_in_dim,
                  lstm_hidden_dim=lstm_hidden_dim,
                  lstm_num_layers=lstm_num_layers,
                  lstm_dropout=lstm_dropout,
                  emb_dropout=emb_dropout,
                  output_dropout=output_dropout,
                  layer_norm=layer_norm,
                  lr=lr,
                  encoder_type=encoder_type,
                  optimizer=optimizer,
                  highway_connection=highway_connection,
                  hidden2logit_num_layers=hidden2logit_num_layers)

    if smoothness_alpha > 0:
        content_df = read_contents(questions_path="./dataset/questions.csv",
                                   lectures_path="./dataset/lectures.csv")
        content_adj_mat = get_content_adjacency_matrix(content_df=content_df,
                                                       content_id_idx=content_id_idx,
                                                       no_question_lecture_link=True,
                                                       no_self_loop=True)
        config["smoothness_alpha"] = smoothness_alpha

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config["content_adj_mat"] = content_adj_mat.to(device)

    print("====== Model config =====")
    print(config)
    print("=========================")

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
                         gpus=gpus,
                         truncated_bptt_steps=truncated_bptt_steps,
                         overfit_batches=overfit_pct,
                         gradient_clip_val=gradient_clip_val,
                         track_grad_norm=track_grad_norm,
                         weights_summary='full',
                         val_check_interval=val_check_interval,
                         resume_from_checkpoint=resume_from_checkpoint)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    fire.Fire(main)
