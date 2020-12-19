from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingWithDropout(nn.Module):
    """
    Apply dropout with probability `embed_p` to an embedding layer `emb`.
    taken from fast.ai library
    """

    def __init__(self, embedding_module, dropout_p):
        super().__init__()

        self.embedding_module = embedding_module
        self.dropout_p = dropout_p

    @staticmethod
    def dropout_mask(x: torch.Tensor, sz: Iterable[int], p: float):
        """
        Return a dropout mask of the same type as `x`, size `sz`,
        with probability `p` to cancel an element.
        """
        return x.new_empty(*sz).bernoulli_(1-p).div_(1-p)

    def forward(self, words, scale=None):
        if self.training and self.dropout_p != 0:
            size = (self.embedding_module.weight.size(0), 1)
            mask = self.__class__.dropout_mask(self.embedding_module.weight.data,
                                               size,
                                               self.dropout_p)
            masked_embed = self.embedding_module.weight * mask
        else:
            masked_embed = self.embedding_module.weight

        if scale:
            masked_embed.mul_(scale)

        padding_idx = self.embedding_module.padding_idx or -1

        return F.embedding(words,
                           masked_embed, padding_idx,
                           self.embedding_module.max_norm,
                           self.embedding_module.norm_type,
                           self.embedding_module.scale_grad_by_freq,
                           self.embedding_module.sparse)
