import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pytorch_lightning as pl
from allennlp.modules.input_variational_dropout import InputVariationalDropout

from libs.modules.stacked_augmented_lstm import StackedAugmentedLSTM


class Predictor(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.hparams = kwargs

        content_id_size = self.hparams["content_id_size"]
        content_id_dim = self.hparams["content_id_dim"]

        self.content_id_emb = nn.Embedding(num_embeddings=content_id_size,
                                           embedding_dim=content_id_dim,
                                           padding_idx=0)

        if self.hparams["emb_dropout"] > 0:
            self.emb_dropout = InputVariationalDropout(p=self.hparams["emb_dropout"])

        lstm_in_dim = self.hparams["feature_dim"] + self.hparams["content_id_dim"]
        lstm_hidden_dim = self.hparams["lstm_hidden_dim"]
        lstm_num_layers = self.hparams["lstm_num_layers"]
        lstm_dropout = self.hparams["lstm_dropout"]

        self.encoder_type = self.hparams.get("encoder_type", "vanilla_lstm")
        if self.encoder_type == "vanilla_lstm":
            self.encoder = nn.LSTM(input_size=lstm_in_dim,
                                   hidden_size=lstm_hidden_dim,
                                   bidirectional=False,
                                   batch_first=True,
                                   num_layers=lstm_num_layers,
                                   dropout=lstm_dropout)
        elif self.encoder_type == "GRU":
            self.encoder = nn.GRU(input_size=lstm_in_dim,
                                  hidden_size=lstm_hidden_dim,
                                  bidirectional=False,
                                  batch_first=True,
                                  num_layers=lstm_num_layers,
                                  dropout=lstm_dropout)
        elif self.encoder_type == "augmented_lstm":
            self.encoder = StackedAugmentedLSTM(input_size=lstm_in_dim,
                                                hidden_size=lstm_hidden_dim,
                                                num_layers=lstm_num_layers,
                                                recurrent_dropout_probability=lstm_dropout,
                                                use_highway=self.hparams.get("lstm_use_highway", True))

        if self.hparams.get("layer_norm", False):
            self.layer_norm = nn.LayerNorm(lstm_hidden_dim)

        if self.hparams["output_dropout"] > 0:
            self.output_dropout = InputVariationalDropout(p=self.hparams["output_dropout"])

        self.hidden2logit = nn.Linear(in_features=lstm_hidden_dim,
                                      out_features=1)

        self.criterion = F.binary_cross_entropy_with_logits

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

    def forward(self,
                content_id: torch.LongTensor,
                bundle_id: torch.LongTensor,
                feature: torch.FloatTensor,
                user_id: torch.FloatTensor,
                mask: torch.Tensor):
        # content_emb: (batch, seq, dim)
        content_emb = self.content_id_emb(content_id)

        if self.hparams["emb_dropout"] > 0:
            content_emb = self.emb_dropout(content_emb)

        feature = torch.cat([content_emb, feature], dim=-1)

        # Apply LSTM
        sequence_lengths = self.__class__.get_lengths_from_seq_mask(mask)

        if self.encoder_type == "augmented_lstm":
            sorted_feature, sorted_sequence_lengths, restoration_indices, sorting_indices = \
                self.__class__.sort_batch_by_length(feature, sequence_lengths)
            packed_sequence_input = pack_padded_sequence(sorted_feature,
                                                         sorted_sequence_lengths.data.tolist(),
                                                         enforce_sorted=False,
                                                         batch_first=True)
            # encoder_out: (batch, seq_len, num_directions * hidden_size):
            # h_t: (num_layers * num_directions, batch, hidden_size)
            #    - this dimension is valid regardless of batch_first=True!!
            packed_lstm_out, (h_t, c_t) = self.encoder(packed_sequence_input)
            # lstm_out: (batch, seq, num_directions * hidden_size)
            lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)

            lstm_out = lstm_out.index_select(0, restoration_indices)
            h_t = h_t.index_select(1, restoration_indices)
            c_t = c_t.index_select(1, restoration_indices)
        else:
            packed_sequence_input = pack_padded_sequence(feature,
                                                         sequence_lengths.data.tolist(),
                                                         enforce_sorted=False,
                                                         batch_first=True)

            # encoder_out: (batch, seq_len, num_directions * hidden_size):
            # h_t: (num_layers * num_directions, batch, hidden_size)
            #    - this dimension is valid regardless of batch_first=True!!
            packed_lstm_out, (h_t, c_t) = self.encoder(packed_sequence_input)

            # lstm_out: (batch, seq, num_directions * hidden_size)
            lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)

        if self.hparams.get("layer_norm", False):
            lstm_out = self.layer_norm(lstm_out)

        lstm_out = F.relu(lstm_out)

        if self.hparams["output_dropout"] > 0:
            lstm_out = self.output_dropout(lstm_out)

        y_pred = torch.squeeze(self.hidden2logit(lstm_out), dim=-1)  # (batch, seq)

        return y_pred, (h_t, c_t)

    def _step(self, batch):
        actual = batch["y"]  # (batch, seq)
        seq_len_mask = batch["seq_len_mask"]  # (batch, seq)
        question_mask = batch["question_mask"]  # (batch, seq)

        batch_size, seq_len = actual.shape
        actual = actual.view(batch_size * seq_len).float()
        actual[actual < 0] = 0

        content_id = batch["content_id"]
        bundle_id = batch["bundle_id"]
        feature = batch["feature"]
        user_id = batch["user_id"]

        # assert torch.isnan(content_id).sum() == 0
        # assert torch.isnan(bundle_id).sum() == 0
        # assert torch.isnan(feature).sum() == 0
        # assert torch.isnan(seq_len_mask).sum() == 0

        content_id[torch.isnan(content_id)] = 0
        bundle_id[torch.isnan(bundle_id)] = 0
        feature[torch.isnan(feature)] = 0
        seq_len_mask[torch.isnan(seq_len_mask)] = 0

        pred, _ = self.forward(content_id=content_id,
                               bundle_id=bundle_id,
                               feature=feature,
                               user_id=user_id,
                               mask=seq_len_mask)
        pred = pred.view(batch_size * seq_len)

        flatten_mask = (seq_len_mask & question_mask).view(batch_size * seq_len)
        loss = self.criterion(input=pred,
                              target=actual,
                              weight=flatten_mask,
                              reduction="sum")
        loss = loss / flatten_mask.sum()

        if self.hparams.get("b_flooding") is not None:
            b = torch.tensor(self.hparams["b_flooding"], dtype=torch.float)
            loss = torch.abs(loss - b) + b

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.logger.experiment.add_scalar("Loss/Train", loss, self.current_epoch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        outputs = {
            'loss': loss
        }
        return outputs

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)

        self.logger.experiment.add_scalar("Loss/Validation", loss, self.current_epoch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        outputs = {
            'loss': loss
        }
        return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
