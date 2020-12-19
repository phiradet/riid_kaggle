from typing import *
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightDropout(nn.Module):
    """
    A module that wraps another layer in which some weights will be replaced by 0 during training.
    taken from fast.ai library
    """

    def __init__(self, module, weight_p, layer_names: List[str]):
        super().__init__()

        self.module = module
        self.weight_p = weight_p
        self.layer_names = layer_names  # ex [weight_hh_l0] for LSTM

        for layer in self.layer_names:
            # makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            delattr(self.module, layer)

            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            setattr(self.module, layer, w.clone())

            if isinstance(self.module, (nn.RNNBase, nn.modules.rnn.RNNBase)):
                self.module.flatten_parameters = self._do_nothing

    def _set_weights(self):
        """
        Apply dropout to the raw weights
        """
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')

            if self.training:
                w = F.dropout(raw_w, p=self.weight_p)
            else:
                w = raw_w.clone()

            setattr(self.module, layer, w)

    def forward(self, *args):
        self._set_weights()

        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore", category=UserWarning)

            return self.module(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            setattr(self.module, layer, raw_w.clone())

        if hasattr(self.module, 'reset'):
            self.module.reset()

    def _do_nothing(self):
        pass