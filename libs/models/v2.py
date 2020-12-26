import sys
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from pytorch_lightning.metrics.functional.classification import auroc
from allennlp.nn.util import add_positional_features

from libs.modules.stacked_augmented_lstm import StackedAugmentedLSTM, TensorPair
from libs.models.base_predictor import BasePredictor
from libs.modules.embedding_dropout import EmbeddingWithDropout
from libs.modules.weight_dropout import WeightDropout


class V2Predictor(BasePredictor):

    def __init__(self, **kwargs):
        super().__init__()

        drop_weight_p = kwargs.get("drop_weight_p", 0)

        content_id_size = kwargs["content_id_size"]
        content_id_dim = kwargs["content_id_dim"]

        self.content_id_emb = self.__class__. \
            _get_embedding_layer(num_emb=content_id_size,
                                 emb_dim=content_id_dim,
                                 dropout_p=drop_weight_p)

        bundle_id_size = kwargs["bundle_id_size"]
        bundle_id_dim = kwargs["bundle_id_dim"]
        self.bundle_id_emb = self.__class__. \
            _get_embedding_layer(num_emb=bundle_id_size,
                                 emb_dim=bundle_id_dim,
                                 dropout_p=drop_weight_p)

        if kwargs["emb_dropout"] > 0:
            self.content_emb_dropout = InputVariationalDropout(p=kwargs["emb_dropout"])
            self.bundle_emb_dropout = InputVariationalDropout(p=kwargs["emb_dropout"])

        content_feature_dim = kwargs["feature_dim"] + content_id_dim + bundle_id_dim

        content_feedback_dim = 3  # (answer correctly, answer incorrectly, is_lecture)
        raw_encoder_in_dim = content_feature_dim + content_feedback_dim

        if "lstm_in_dim" in kwargs and kwargs["lstm_in_dim"] != raw_encoder_in_dim:
            encoder_in_dim = kwargs["lstm_in_dim"]
            self.encoder_in_proj = nn.Linear(in_features=raw_encoder_in_dim,
                                             out_features=encoder_in_dim,
                                             bias=True)
            query_dim = encoder_in_dim
            self.query_proj = nn.Linear(in_features=content_feature_dim,
                                        out_features=encoder_in_dim,
                                        bias=True)
        else:
            encoder_in_dim = raw_encoder_in_dim
            query_dim = encoder_in_dim

        lstm_hidden_dim = kwargs["lstm_hidden_dim"]
        lstm_num_layers = kwargs["lstm_num_layers"]
        lstm_dropout = kwargs["lstm_dropout"]

        self.encoder_type = kwargs.get("encoder_type", "vanilla_lstm")
        if self.encoder_type == "vanilla_lstm":
            if drop_weight_p > 0:
                layer_names = [f"weight_hh_l{i}" for i in range(lstm_num_layers)]
                self.encoder = WeightDropout(nn.LSTM(input_size=encoder_in_dim,
                                                     hidden_size=lstm_hidden_dim,
                                                     bidirectional=False,
                                                     batch_first=True,
                                                     num_layers=lstm_num_layers,
                                                     dropout=lstm_dropout),
                                             weight_p=drop_weight_p,
                                             layer_names=layer_names)
            else:
                self.encoder = nn.LSTM(input_size=encoder_in_dim,
                                       hidden_size=lstm_hidden_dim,
                                       bidirectional=False,
                                       batch_first=True,
                                       num_layers=lstm_num_layers,
                                       dropout=lstm_dropout)
        elif self.encoder_type == "augmented_lstm":
            self.encoder = StackedAugmentedLSTM(input_size=encoder_in_dim,
                                                hidden_size=lstm_hidden_dim,
                                                num_layers=lstm_num_layers,
                                                recurrent_dropout_probability=lstm_dropout,
                                                layer_dropout_probability=lstm_dropout,
                                                use_highway=kwargs.get("lstm_use_highway", True))
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

        ################################
        #     Multi-head attention     #
        ################################
        self.attention_range = kwargs["attention_range"]
        self.attention_layer = nn.MultiheadAttention(embed_dim=query_dim,
                                                     num_heads=8,
                                                     dropout=0.3,
                                                     kdim=lstm_hidden_dim,
                                                     vdim=lstm_hidden_dim)

        hidden2logit_num_layers = kwargs.get("hidden2logit_num_layers", 1)
        self.hidden2logit = []

        linear_in_dim = query_dim
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

    @staticmethod
    def _get_embedding_layer(num_emb, emb_dim, dropout_p):
        emb = nn.Embedding(num_embeddings=num_emb,
                           embedding_dim=emb_dim,
                           padding_idx=0)
        if dropout_p > 0:
            return EmbeddingWithDropout(embedding_module=emb,
                                        dropout_p=dropout_p)
        else:
            return emb

    def _combine_content_feature(self,
                                 content_id: torch.LongTensor,
                                 content_feature: torch.FloatTensor,
                                 bundle_id: torch.LongTensor,
                                 content_feedback: Optional[torch.FloatTensor] = None) -> torch.Tensor:

        content_emb = self.content_id_emb(content_id)
        bundle_emb = self.bundle_id_emb(bundle_id)

        if self.hparams["emb_dropout"] > 0:
            content_emb = self.content_emb_dropout(content_emb)
            bundle_emb = self.bundle_emb_dropout(bundle_emb)

        features = [content_emb, bundle_emb, content_feature]
        if content_feedback is not None:
            content_feedback = F.dropout(content_feedback,
                                         p=0.2,
                                         training=self.training)
            features.append(content_feedback)

        return torch.cat(features, dim=-1)

    def _apply_dropout_to_state(self, state: Optional[TensorPair] = None):
        if state is None:
            return state
        else:
            return [F.dropout(s,
                              p=self.hparams["lstm_dropout"],
                              training=self.training) for s in state]

    @staticmethod
    def get_limit_range_attention_mask(seq_len: int,
                                       attention_range: int):
        mask = torch.zeros(seq_len, seq_len)
        for i in range(attention_range):
            mask += torch.diag(torch.ones(seq_len), diagonal=i*-1)[:seq_len, :seq_len]

        return mask == 0

    def _apply_attention(self,
                         query_content: torch.Tensor,
                         lstm_state: torch.Tensor):
        """
        query_content: (batch, seq, dim)
        lstm_state: (batch, seq, dim)
        mask: (batch, seq)
        """

        _, seq_len, _ = query_content.shape

        attn_mask = self.__class__.get_limit_range_attention_mask(seq_len=seq_len,
                                                                  attention_range=self.attention_range)
        attn_mask = attn_mask.to(self.device)

        # (batch, seq, dim) -> (seq, batch, dim)
        query = query_content.permute(1, 0, 2)
        key = lstm_state.permute(1, 0, 2)

        # attention_out: (query_seq_len, batch, dim)
        attention_out, _ = self.attention_layer(query=query,
                                                key=key,
                                                value=key,
                                                attn_mask=attn_mask)

        return attention_out.permute(1, 0, 2)  # (batch, seq, dim)

    def forward(self,
                query_content_id: torch.LongTensor,
                query_content_feature: torch.FloatTensor,
                query_bundle_id: torch.LongTensor,
                mask: torch.Tensor,
                seen_content_id: torch.LongTensor,
                seen_content_feature: torch.FloatTensor,
                seen_content_feedback: torch.FloatTensor,
                seen_bundle_id: torch.LongTensor,
                initial_state: Optional[TensorPair] = None,
                return_feature: bool = False):

        # content_emb: (batch, seq, dim)
        seen_content_tensor = self._combine_content_feature(content_id=seen_content_id,
                                                            content_feature=seen_content_feature,
                                                            content_feedback=seen_content_feedback,
                                                            bundle_id=seen_bundle_id)

        if hasattr(self, "encoder_in_proj"):
            seen_content_tensor = self.encoder_in_proj(seen_content_tensor)

        if self.hparams.get("layer_norm", False):
            seen_content_tensor = self.seen_content_layer_norm(seen_content_tensor)

        seen_content_tensor = F.relu(seen_content_tensor)

        # Apply LSTM
        sequence_lengths = self.__class__.get_lengths_from_seq_mask(mask)
        clamped_sequence_lengths = sequence_lengths.clamp(min=1)

        initial_state = self._apply_dropout_to_state(initial_state)
        if self.encoder_type == "vanilla_lstm":
            lstm_out, state = self._apply_vanilla_lstm(clamped_sequence_lengths,
                                                       initial_state,
                                                       seen_content_tensor)
        elif self.encoder_type == "augmented_lstm":
            lstm_out, state = self._apply_augmented_lstm(clamped_sequence_lengths,
                                                         initial_state,
                                                         seen_content_tensor)
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
                                                             content_feedback=None,
                                                             bundle_id=query_bundle_id)
        if hasattr(self, "encoder_in_proj"):
            query_content_tensor = self.query_proj(query_content_tensor)
            query_content_tensor = F.relu(query_content_tensor)

        # pred_tensor: (batch, seq, dim)
        pred_tensor = self._apply_attention(query_content=query_content_tensor,
                                            lstm_state=lstm_out)

        # y_pred: (batch, seq)
        y_pred = torch.squeeze(self.hidden2logit(pred_tensor), dim=-1)

        if return_feature:
            return y_pred, state, pred_tensor
        else:
            return y_pred, state

    def _apply_vanilla_lstm(self,
                            sequence_lengths: torch.Tensor,
                            initial_state: TensorPair,
                            content_feature: torch.Tensor):
        packed_sequence_input = pack_padded_sequence(content_feature,
                                                     sequence_lengths.data.tolist(),
                                                     enforce_sorted=False,
                                                     batch_first=True)
        # encoder_out: (batch, seq_len, num_directions * hidden_size):
        # h_t: (num_layers * num_directions, batch, hidden_size)
        #    - this dimension is valid regardless of batch_first=True!!
        packed_lstm_out, state = self.encoder(packed_sequence_input,
                                              initial_state)
        # lstm_out: (batch, seq, num_directions * hidden_size)
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)
        return lstm_out, state

    def _apply_augmented_lstm(self,
                              sequence_lengths: torch.Tensor,
                              initial_state: TensorPair,
                              content_feature: torch.Tensor):
        sorted_feature, sorted_sequence_lengths, restoration_indices, sorting_indices = \
            self.__class__.sort_batch_by_length(content_feature, sequence_lengths)
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

        return lstm_out, state

    @staticmethod
    def _shift_tensor(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.roll(shifts=1, dims=1)
        tensor[:, 0] = 0.

        return tensor

    @classmethod
    def to_seen_content_feedback(cls,
                                 actual: torch.Tensor,
                                 do_shift: bool = True) -> torch.FloatTensor:
        """
        :param actual: (batch, seq)
        :param do_shift: bool
        :return: (batch, seq, 3)
        """

        # (batch, seq, 1)
        is_answer_correctly = torch.unsqueeze(actual.eq(1.).float(), dim=2)
        is_answer_incorrectly = torch.unsqueeze(actual.eq(0.).float(), dim=2)
        is_lecture = torch.unsqueeze(actual.eq(-1.).float(), dim=2)

        # (batch, seq, 2)
        seen_content_feedback = torch.cat([is_answer_correctly,
                                           is_answer_incorrectly,
                                           is_lecture],
                                          dim=2)
        if do_shift:
            return cls._shift_tensor(seen_content_feedback)
        else:
            return seen_content_feedback

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
        query_bundle_id = batch["bundle_id"]

        query_content_id[torch.isnan(query_content_id)] = 0
        query_content_feature[torch.isnan(query_content_feature)] = 0
        seq_len_mask[torch.isnan(seq_len_mask)] = 0

        seen_content_id = self.__class__._shift_tensor(query_content_id)
        seen_bundle_id = self.__class__._shift_tensor(query_bundle_id)
        seen_content_feature = self.__class__._shift_tensor(query_content_feature)

        pred, hiddens = self.forward(query_content_id=query_content_id,
                                     query_content_feature=query_content_feature,
                                     query_bundle_id=query_bundle_id,
                                     mask=seq_len_mask,
                                     seen_content_id=seen_content_id,
                                     seen_content_feature=seen_content_feature,
                                     seen_content_feedback=seen_content_feedback,
                                     seen_bundle_id=seen_bundle_id,
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
