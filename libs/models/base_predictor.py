import sys
from typing import Optional, Dict

import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from libs.modules.stacked_augmented_lstm import TensorPair


class BasePredictor(pl.LightningModule):

    def __init__(self):
        super().__init__()

    @staticmethod
    def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                         reduce=None, reduction='sum', pos_weight=None):
        loss = F.binary_cross_entropy_with_logits(input, target, weight=weight, size_average=size_average,
                                                  reduce=reduce, reduction=reduction, pos_weight=pos_weight)
        return loss / weight.sum()

    @staticmethod
    def cosine_distance(x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return 1 - (torch.matmul(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps))

    @classmethod
    def content_emb_loss(cls,
                         content_emb_weight: torch.Tensor,
                         content_adjacency: torch.Tensor):
        emb_dist = cls.cosine_distance(content_emb_weight)

        masked_emb_dist = emb_dist * content_adjacency
        normed_emb_dist = masked_emb_dist.sum() / content_adjacency.sum()

        return normed_emb_dist

    def BCE_logit_emb_smooth_loss(self, input, target, weight=None, size_average=None,
                                  reduce=None, reduction='sum', pos_weight=None):
        bce_loss = F.binary_cross_entropy_with_logits(input,
                                                      target,
                                                      weight=weight,
                                                      size_average=size_average,
                                                      reduce=reduce,
                                                      reduction=reduction,
                                                      pos_weight=pos_weight)
        bce_loss = bce_loss / weight.sum()
        smoothness_loss = self.__class__.content_emb_loss(content_emb_weight=self.content_id_emb.weight,
                                                          content_adjacency=self.content_id_adj)

        loss = ((1 - self.smoothness_alpha) * bce_loss) + (self.smoothness_alpha * smoothness_loss)

        return loss

    @staticmethod
    def get_lengths_from_seq_mask(mask: torch.Tensor) -> torch.Tensor:
        return mask.long().sum(-1)

    def spatial_dropout(self, input, p):
        # input: (N, seq, dim)
        # x: (N, dim, seq)
        x = input.permute(0, 2, 1)
        x = F.dropout2d(x, p=p)  # Randomly zero out entire dim
        return x.permute(0, 2, 1)

    @staticmethod
    def sort_batch_by_length(inputs: torch.Tensor, sequence_lengths: torch.LongTensor):
        sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
        sorted_tensor = inputs.index_select(0, permutation_index)

        index_range = torch.arange(0, len(sequence_lengths), device=sequence_lengths.device)
        _, reverse_mapping = permutation_index.sort(0, descending=False)
        restoration_indices = index_range.index_select(0, reverse_mapping)
        return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index

    @staticmethod
    def get_attention_mask(src_seq_len: int,
                           target_seq_len: Optional[int] = None):
        """
        positions with True is not allowed to attend
        """
        if target_seq_len is None:
            target_seq_len = src_seq_len
        mask = torch.triu(torch.ones(target_seq_len, src_seq_len), diagonal=1).bool()
        return mask

    def tbptt_split_batch(self, batch: Dict[str, torch.Tensor], split_size: int) -> list:
        seq_lens = []

        for key, tensor in batch.items():
            if tensor.dim() > 1:
                batch_size, seq_len, *_ = tensor.shape
                seq_lens.append(seq_len)
        seq_lens = set(seq_lens)

        assert len(seq_lens) == 1, f"Ambiguous seq_len {seq_lens}"

        seq_len = next(iter(seq_lens))

        splits = []
        for t in range(0, seq_len, split_size):
            batch_split = {}
            batch_seq_len = sys.maxsize
            for key, tensor in batch.items():

                if tensor.dim() > 1:
                    tensor = tensor[:, t: t + split_size].contiguous()
                    batch_seq_len = min(batch_seq_len, tensor.shape[1])

                batch_split[key] = tensor

            if batch_seq_len > 0:
                splits.append(batch_split)

        return splits

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int,
                      hiddens: Optional[TensorPair] = None):
        loss, hiddens = self._step(batch, hiddens)
        self.logger.experiment.add_scalar("Loss/Train", loss, self.current_epoch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if hiddens is not None:
            hiddens = [h.detach() for h in hiddens]

        outputs = {
            "loss": loss,
            "hiddens": hiddens
        }
        return outputs

    def validation_step(self, batch, batch_idx):
        loss, _ = self._step(batch)

        self.logger.experiment.add_scalar("Loss/Validation", loss, self.current_epoch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        outputs = {
            'loss': loss
        }
        return outputs

    def configure_optimizers(self):
        optimizer = self.hparams.get("optimizer", "adam").lower()

        if optimizer == "adam":
            return torch.optim.Adam(self.parameters(),
                                    lr=self.hparams.get("lr", 1e-3),
                                    weight_decay=self.hparams.get("weight_decay", 0))
        elif optimizer == "sgd":
            return torch.optim.SGD(self.parameters(),
                                   lr=self.hparams.get("lr", 1e-3),
                                   weight_decay=self.hparams.get("weight_decay", 0))
        elif optimizer == "asgd":
            return torch.optim.ASGD(self.parameters(),
                                    lr=self.hparams.get("lr", 1e-2),
                                    weight_decay=self.hparams.get("weight_decay", 0))
        else:
            raise ValueError(f"Unknown optimizer type {optimizer}")
