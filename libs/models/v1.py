import sys
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pytorch_lightning as pl
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.nn.util import add_positional_features
from pytorch_lightning.metrics.functional.classification import auroc

from libs.modules.stacked_augmented_lstm import StackedAugmentedLSTM, TensorPair
from libs.models.base_predictor import BasePredictor


class V1Predictor(BasePredictor):

    def __init__(self, **kwargs):
        super().__init__()

        content_id_size = kwargs["content_id_size"]
        content_id_dim = kwargs["content_id_dim"]

        self.content_id_emb = nn.Embedding(num_embeddings=content_id_size,
                                           embedding_dim=content_id_dim,
                                           padding_idx=0)

        if kwargs["emb_dropout"] > 0:
            self.emb_dropout = InputVariationalDropout(p=kwargs["emb_dropout"])

        content_feature_dim = kwargs["feature_dim"] + kwargs["content_id_dim"]
        prev_content_state = 3  # (answer correctly, answer incorrectly, lecture)
        raw_encoder_in_dim = content_feature_dim + prev_content_state

        if "lstm_in_dim" in kwargs and kwargs["lstm_in_dim"] != raw_encoder_in_dim:
            encoder_in_dim = kwargs["lstm_in_dim"]
            self.encoder_in_proj = nn.Linear(in_features=raw_encoder_in_dim,
                                             out_features=encoder_in_dim,
                                             bias=True)
            query_dim = content_feature_dim // 2
            self.query_proj = nn.Linear(in_features=content_feature_dim,
                                        out_features=query_dim,
                                        bias=True)
        else:
            encoder_in_dim = raw_encoder_in_dim
            query_dim = content_feature_dim

        lstm_hidden_dim = kwargs["lstm_hidden_dim"]
        lstm_num_layers = kwargs["lstm_num_layers"]
        lstm_dropout = kwargs["lstm_dropout"]

        self.encoder_type = kwargs.get("encoder_type", "vanilla_lstm")
        if self.encoder_type == "vanilla_lstm":
            self.encoder = nn.LSTM(input_size=encoder_in_dim,
                                   hidden_size=lstm_hidden_dim,
                                   bidirectional=False,
                                   batch_first=True,
                                   num_layers=lstm_num_layers,
                                   dropout=lstm_dropout)
        else:
            raise NotImplementedError

        if kwargs.get("layer_norm", False):
            self.seen_content_layer_norm = nn.LayerNorm(encoder_in_dim)
            self.encoder_layer_norm = nn.LayerNorm(lstm_hidden_dim)

        if kwargs["output_dropout"] > 0:
            self.output_dropout = InputVariationalDropout(p=kwargs["output_dropout"])

        if kwargs.get("highway_connection", False):
            self.highway_H = nn.Linear(in_features=encoder_in_dim,
                                       out_features=lstm_hidden_dim)
            self.highway_C = nn.Linear(in_features=encoder_in_dim,
                                       out_features=lstm_hidden_dim)

        hidden2logit_num_layers = kwargs.get("hidden2logit_num_layers", 1)
        self.hidden2logit = []

        linear_in_dim = lstm_hidden_dim + query_dim
        for i in range(hidden2logit_num_layers):
            if i == hidden2logit_num_layers - 1:
                self.hidden2logit.append(nn.Linear(in_features=linear_in_dim,
                                                   out_features=1))
            else:
                self.hidden2logit.append(nn.Linear(in_features=linear_in_dim,
                                                   out_features=linear_in_dim // 2))
                self.hidden2logit.append(nn.ReLU())
                self.hidden2logit.append(nn.Dropout(p=kwargs["output_dropout"]))
                linear_in_dim = linear_in_dim // 2
        self.hidden2logit = nn.Sequential(*self.hidden2logit)

        # ======== Loss ========
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

    def _combine_content_feature(self,
                                 content_id: torch.LongTensor,
                                 content_feature: torch.FloatTensor,
                                 content_feedback: Optional[torch.FloatTensor] = None) -> torch.Tensor:

        content_emb = self.content_id_emb(content_id)
        if self.hparams["emb_dropout"] > 0:
            content_emb = self.emb_dropout(content_emb)

        features = [content_emb, content_feature]
        if content_feedback is not None:
            content_feedback = F.dropout(content_feedback,
                                         p=0.2,
                                         training=self.training)
            features.append(content_feedback)

        return torch.cat(features, dim=-1)

    def forward(self,
                query_content_id: torch.LongTensor,
                query_content_feature: torch.FloatTensor,
                mask: torch.Tensor,
                seen_content_id: Optional[torch.LongTensor],
                seen_content_feature: Optional[torch.FloatTensor],
                seen_content_feedback: Optional[torch.FloatTensor],
                initial_state: Optional[TensorPair] = None):

        # content_emb: (batch, seq, dim)
        seen_content_tensor = self._combine_content_feature(content_id=seen_content_id,
                                                            content_feature=seen_content_feature,
                                                            content_feedback=seen_content_feedback)

        if hasattr(self, "encoder_in_proj"):
            seen_content_tensor = self.encoder_in_proj(seen_content_tensor)

        if self.hparams.get("layer_norm", False):
            seen_content_tensor = self.seen_content_layer_norm(seen_content_tensor)

        seen_content_tensor = F.relu(seen_content_tensor)

        # Apply LSTM
        sequence_lengths = self.__class__.get_lengths_from_seq_mask(mask)
        clamped_sequence_lengths = sequence_lengths.clamp(min=1)

        if self.encoder_type == "vanilla_lstm":
            packed_sequence_input = pack_padded_sequence(seen_content_tensor,
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
        else:
            raise NotImplementedError

        if self.hparams.get("layer_norm", False):
            lstm_out = self.encoder_layer_norm(lstm_out)

        if self.hparams.get("highway_connection", False):
            c = torch.sigmoid(self.highway_C(lstm_out))
            h = self.highway_H(lstm_out)
            lstm_out = (1 - c) * torch.relu(h) + c * torch.relu(seen_content_tensor)
        else:
            lstm_out = F.relu(lstm_out)

        if self.hparams["output_dropout"] > 0:
            # lstm_out: (batch, seq, dim)
            lstm_out = self.output_dropout(lstm_out)

        # query_content_tensor: (batch, seq, dim)
        query_content_tensor = self._combine_content_feature(content_id=query_content_id,
                                                             content_feature=query_content_feature,
                                                             content_feedback=None)
        if hasattr(self, "encoder_in_proj"):
            query_content_tensor = self.query_proj(query_content_tensor)

        # pred_tensor: (batch, seq, dim)
        pred_tensor = torch.cat([query_content_tensor, lstm_out], dim=-1)

        # y_pred: (batch, seq)
        y_pred = torch.squeeze(self.hidden2logit(pred_tensor), dim=-1)

        return y_pred, state

    @staticmethod
    def _shift_tensor(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.roll(shifts=1, dims=1)
        tensor[:, 0] = 0.

        return tensor

    @classmethod
    def to_seen_content_feedback(cls, actual: torch.Tensor) -> torch.FloatTensor:
        """
        :param actual: (batch, seq)
        :return: (batch, seq, 3)
        """

        # (batch, seq, 1)
        is_answer_correctly = torch.unsqueeze(actual.eq(1.).float(), dim=2)
        is_answer_incorrectly = torch.unsqueeze(actual.eq(0.).float(), dim=2)
        is_lecture = torch.unsqueeze(actual.eq(0.).float(), dim=2)

        # (batch, seq, 2)
        seen_content_feedback = torch.cat([is_answer_correctly,
                                           is_answer_incorrectly,
                                           is_lecture],
                                          dim=2)
        return cls._shift_tensor(seen_content_feedback)

    def _step(self, batch, hiddens=None, calculate_roc=False):
        actual: torch.Tensor = batch["y"].clone()  # (batch, seq)

        seen_content_feedback = self.__class__.to_seen_content_feedback(actual)

        seq_len_mask = batch["seq_len_mask"]  # (batch, seq)
        question_mask = batch["question_mask"]  # (batch, seq)

        batch_size, seq_len = actual.shape
        actual = actual.view(batch_size * seq_len).float()
        actual[actual < 0] = 0

        query_content_id = batch["content_id"]
        query_content_feature = batch["feature"]

        query_content_id[torch.isnan(query_content_id)] = 0
        query_content_feature[torch.isnan(query_content_feature)] = 0
        seq_len_mask[torch.isnan(seq_len_mask)] = 0

        seen_content_id = self.__class__._shift_tensor(query_content_id)
        seen_content_feature = self.__class__._shift_tensor(query_content_feature)

        pred, hiddens = self.forward(query_content_id=query_content_id,
                                     query_content_feature=query_content_feature,
                                     mask=seq_len_mask,
                                     seen_content_id=seen_content_id,
                                     seen_content_feature=seen_content_feature,
                                     seen_content_feedback=seen_content_feedback,
                                     initial_state=hiddens)
        pred = pred.view(batch_size * seq_len)

        flatten_mask = (seq_len_mask & question_mask).view(batch_size * seq_len)
        loss = self.criterion(input=pred,
                              target=actual,
                              weight=flatten_mask,
                              reduction="sum")

        if self.hparams.get("b_flooding") is not None:
            b = torch.tensor(self.hparams["b_flooding"], dtype=torch.float)
            loss = torch.abs(loss - b) + b

        if calculate_roc:
            auc_score = auroc(pred=pred,
                              target=actual,
                              sample_weight=flatten_mask)
            return loss, hiddens, auc_score
        else:
            return loss, hiddens
