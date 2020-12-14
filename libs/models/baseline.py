from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.nn.util import add_positional_features

from libs.models.base_predictor import BasePredictor
from libs.modules.stacked_augmented_lstm import StackedAugmentedLSTM, TensorPair


class Predictor(BasePredictor):

    def __init__(self, **kwargs):
        super().__init__()

        content_id_size = kwargs["content_id_size"]
        content_id_dim = kwargs["content_id_dim"]

        self.content_id_emb = nn.Embedding(num_embeddings=content_id_size,
                                           embedding_dim=content_id_dim,
                                           padding_idx=0)

        if kwargs["emb_dropout"] > 0:
            self.emb_dropout = InputVariationalDropout(p=kwargs["emb_dropout"])

        feature_dim = kwargs["feature_dim"] + kwargs["content_id_dim"]
        if not kwargs.get("no_prev_ans", False):
            feature_dim += 1
        if "lstm_in_dim" in kwargs and kwargs["lstm_in_dim"] != feature_dim:
            lstm_in_dim = kwargs["lstm_in_dim"]
            self.lstm_in_proj = nn.Linear(in_features=feature_dim,
                                          out_features=lstm_in_dim,
                                          bias=True)
        else:
            lstm_in_dim = feature_dim

        lstm_hidden_dim = kwargs["lstm_hidden_dim"]
        lstm_num_layers = kwargs["lstm_num_layers"]
        lstm_dropout = kwargs["lstm_dropout"]

        self.encoder_type = kwargs.get("encoder_type", "vanilla_lstm")
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
        elif self.encoder_type == "attention":
            self.encoder = nn.MultiheadAttention(embed_dim=lstm_in_dim,
                                                 num_heads=lstm_num_layers,
                                                 dropout=lstm_dropout)
        elif self.encoder_type == "augmented_lstm":
            self.encoder = StackedAugmentedLSTM(input_size=lstm_in_dim,
                                                hidden_size=lstm_hidden_dim,
                                                num_layers=lstm_num_layers,
                                                recurrent_dropout_probability=lstm_dropout,
                                                use_highway=kwargs.get("lstm_use_highway", True))

        if kwargs.get("layer_norm", False):
            self.layer_norm = nn.LayerNorm(lstm_hidden_dim)

        if kwargs["output_dropout"] > 0:
            self.output_dropout = InputVariationalDropout(p=kwargs["output_dropout"])

        if kwargs.get("highway_connection", False):
            self.highway_H = nn.Linear(in_features=lstm_in_dim, out_features=lstm_hidden_dim)
            self.highway_C = nn.Linear(in_features=lstm_in_dim, out_features=lstm_hidden_dim)

        hidden2logit_num_layers = kwargs.get("hidden2logit_num_layers", 1)
        self.hidden2logit = []

        in_features = lstm_hidden_dim
        for i in range(hidden2logit_num_layers):
            if i == hidden2logit_num_layers - 1:
                self.hidden2logit.append(nn.Linear(in_features=in_features,
                                                   out_features=1))
            else:
                self.hidden2logit.append(nn.Linear(in_features=in_features,
                                                   out_features=in_features // 2))
                self.hidden2logit.append(nn.ReLU())
                self.hidden2logit.append(nn.Dropout(p=kwargs["output_dropout"]))
                in_features = in_features // 2
        self.hidden2logit = nn.Sequential(*self.hidden2logit)

        if "content_adj_mat" in kwargs and "smoothness_alpha" in kwargs:
            # normalize adjacency weight with w_ij/(sqrt(d_i)*sqrt(d_j))
            adjacency_mat = kwargs["content_adj_mat"]
            del kwargs["content_adj_mat"]

            degree_mat = torch.clamp(adjacency_mat.sum(dim=1), 1e-8)
            inv_degree_mat = torch.diag(torch.pow(degree_mat, -0.5))
            self.content_id_adj = inv_degree_mat @ adjacency_mat @ inv_degree_mat

            self.smoothness_alpha = kwargs["smoothness_alpha"]

            self.criterion = self.BCE_logit_emb_smooth_loss
        else:
            self.criterion = self.__class__.binary_cross_entropy_with_logits

        self.hparams = kwargs

    def forward(self,
                content_id: torch.LongTensor,
                bundle_id: torch.LongTensor,
                feature: torch.FloatTensor,
                user_id: torch.FloatTensor,
                mask: torch.Tensor,
                initial_state: Optional[TensorPair] = None,
                ans_prev_correctly: Optional[torch.Tensor] = None):

        batch_size, seq_len = content_id.shape
        # content_emb: (batch, seq, dim)
        content_emb = self.content_id_emb(content_id)

        if self.hparams["emb_dropout"] > 0:
            content_emb = self.emb_dropout(content_emb)

        # content_emb: (batch, seq, dim)
        feature = torch.cat([content_emb, feature], dim=-1)

        if not self.hparams.get("no_prev_ans", False):
            if ans_prev_correctly is None:
                ans_prev_correctly = torch.ones(batch_size, seq_len, 1,
                                                device=self.device, dtype=torch.float)
            feature = torch.cat([ans_prev_correctly, feature], dim=-1)

        if hasattr(self, "lstm_in_proj"):
            feature = self.lstm_in_proj(feature)
            feature = F.relu(feature)

        # Apply LSTM
        sequence_lengths = self.__class__.get_lengths_from_seq_mask(mask)
        clamped_sequence_lengths = sequence_lengths.clamp(min=1)

        if self.encoder_type == "augmented_lstm":
            sorted_feature, sorted_sequence_lengths, restoration_indices, sorting_indices = \
                self.__class__.sort_batch_by_length(feature, clamped_sequence_lengths)
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
            state = (h_t, c_t)
        elif self.encoder_type == "attention":
            if initial_state is None:
                mask_seq_len = seq_len
                attention_mask = self.__class__ \
                    .get_attention_mask(src_seq_len=mask_seq_len) \
                    .to(self.device)
                permuted_feature = add_positional_features(feature, max_timescale=seq_len) \
                    .permute(1, 0, 2)
                query = permuted_feature
            else:
                mask_seq_len = seq_len + 1

                # initial_state: (batch, 1, dim)
                initial_state = initial_state[0]
                feature = torch.cat([feature, initial_state], dim=1)  # previous sequence summary vector

                attention_mask = self.__class__ \
                    .get_attention_mask(src_seq_len=mask_seq_len,
                                        target_seq_len=seq_len) \
                    .to(self.device)

                # (seq, N, dim)
                permuted_feature = add_positional_features(feature, max_timescale=mask_seq_len) \
                    .permute(1, 0, 2)
                query = permuted_feature[1:]

            # att_output: (seq, batch, dim)
            att_output, att_weight = self.encoder(query=query,
                                                  key=permuted_feature,
                                                  value=permuted_feature,
                                                  attn_mask=attention_mask,
                                                  need_weights=False)
            # (batch, seq, dim)
            lstm_out = att_output.permute(1, 0, 2)

            # (batch, 1, dim)
            summary_vec, _ = lstm_out.max(dim=1, keepdim=True)
            state = [summary_vec]
        else:
            packed_sequence_input = pack_padded_sequence(feature,
                                                         clamped_sequence_lengths.data.tolist(),
                                                         enforce_sorted=False,
                                                         batch_first=True)

            # encoder_out: (batch, seq_len, num_directions * hidden_size):
            # h_t: (num_layers * num_directions, batch, hidden_size)
            #    - this dimension is valid regardless of batch_first=True!!
            packed_lstm_out, state = self.encoder(packed_sequence_input,
                                                  initial_state)

            # lstm_out: (batch, seq, num_directions * hidden_size)
            lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)

        if self.hparams.get("layer_norm", False):
            lstm_out = self.layer_norm(lstm_out)

        if self.hparams.get("highway_connection", False):
            c = torch.sigmoid(self.highway_C(lstm_out))
            h = self.highway_H(lstm_out)

            lstm_out = (1 - c) * torch.relu(h) + c * torch.relu(feature)
        else:
            lstm_out = F.relu(lstm_out)

        if self.hparams["output_dropout"] > 0:
            lstm_out = self.output_dropout(lstm_out)

        y_pred = torch.squeeze(self.hidden2logit(lstm_out), dim=-1)  # (batch, seq)

        return y_pred, state

    def _step(self, batch, hiddens=None):
        actual = batch["y"].clone()  # (batch, seq)

        ans_prev_correctly = batch["y"].clone()
        ans_prev_correctly = torch.roll(ans_prev_correctly, 1, dims=1)
        ans_prev_correctly[:, 0] = -1
        ans_prev_correctly = torch.unsqueeze(ans_prev_correctly, dim=2)

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

        pred, hiddens = self.forward(content_id=content_id,
                                     bundle_id=bundle_id,
                                     feature=feature,
                                     user_id=user_id,
                                     mask=seq_len_mask,
                                     initial_state=hiddens,
                                     ans_prev_correctly=ans_prev_correctly)
        pred = pred.view(batch_size * seq_len)

        flatten_mask = (seq_len_mask & question_mask).view(batch_size * seq_len)
        loss = self.criterion(input=pred,
                              target=actual,
                              weight=flatten_mask,
                              reduction="sum")

        if self.hparams.get("b_flooding") is not None:
            b = torch.tensor(self.hparams["b_flooding"], dtype=torch.float)
            loss = torch.abs(loss - b) + b

        return loss, hiddens
