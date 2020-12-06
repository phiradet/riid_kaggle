from typing import *

import torch
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.common.checks import ConfigurationError


TensorPair = Tuple[torch.Tensor, torch.Tensor]


class StackedAugmentedLSTM(torch.nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            recurrent_dropout_probability: float = 0.0,
            layer_dropout_probability: float = 0.0,
            use_highway: bool = True,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True
        self.layer_dropout_probability = layer_dropout_probability

        layers = []
        lstm_input_size = input_size
        for layer_index in range(num_layers):
            forward_layer = AugmentedLstm(
                lstm_input_size,
                hidden_size,
                go_forward=True,
                recurrent_dropout_probability=recurrent_dropout_probability,
                use_highway=use_highway,
                use_input_projection_bias=False,
            )
            lstm_input_size = hidden_size
            self.add_module("forward_layer_{}".format(layer_index), forward_layer)
            layers.append(forward_layer)

        self.lstm_layers = layers
        self.layer_dropout = InputVariationalDropout(layer_dropout_probability)

    def forward(
            self, inputs: PackedSequence, initial_state: Optional[TensorPair] = None
    ) -> Tuple[PackedSequence, TensorPair]:

        if initial_state is None:
            hidden_states: List[Optional[TensorPair]] = [None] * len(self.lstm_layers)
        elif initial_state[0].size()[0] != len(self.lstm_layers):
            raise ConfigurationError(
                "Initial states were passed to forward() but the number of "
                "initial states does not match the number of layers."
            )
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))

        output_sequence = inputs
        final_h = []
        final_c = []

        for i, state in enumerate(hidden_states):
            forward_layer = getattr(self, "forward_layer_{}".format(i))

            # The state is duplicated to mirror the Pytorch API for LSTMs.
            forward_output, final_forward_state = forward_layer(output_sequence, state)

            if self.layer_dropout_probability == 0:
                output_sequence = forward_output
            else:

                forward_output, lengths = pad_packed_sequence(forward_output, batch_first=True)

                output_sequence = forward_output
                # Apply layer wise dropout on each output sequence apart from the
                # first (input) and last
                if i < (self.num_layers - 1):
                    output_sequence = self.layer_dropout(output_sequence)
                output_sequence = pack_padded_sequence(output_sequence, lengths, batch_first=True)

            final_h.append(final_forward_state[0])
            final_c.append(final_forward_state[1])

        final_h = torch.cat(final_h, dim=0)
        final_c = torch.cat(final_c, dim=0)
        final_state_tuple = (final_h, final_c)

        return output_sequence, final_state_tuple
