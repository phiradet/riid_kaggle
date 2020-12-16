import os
import pickle
from typing import *

import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

from libs.inferencers.rnn_state import RNNState
from libs.modules.stacked_augmented_lstm import TensorPair

from libs.feature import extract_feature
from libs.models.baseline import Predictor
from libs.utils.io import load_state_dict
from libs.inferencers._base import _BaseInference


class BaselineInferencer(_BaseInference):

    def update_state(self, user_ids: List[str], states: TensorPair):
        self.rnn_state.update_state(user_ids, states)

    def update_state_with_prev_ans(self, prior_group_answers_correct: List[int], verbose: bool = False):
        self.prev_group_test_df["answered_correctly"] = prior_group_answers_correct
        df = self.prev_group_test_df.groupby("user_id").apply(self.aggregate)

        user_ids = df.index.to_list()

        # (N, seq)
        content_id_tensor = pad_sequence(df["content_id"].to_list(), batch_first=True)

        # (N, seq, dim)
        feature_tensor = pad_sequence(df["feature"].to_list(), batch_first=True)

        seq_len_mask = (content_id_tensor != 0).to(dtype=torch.uint8)
        initial_state = self.get_state(user_ids)

        with torch.no_grad():
            if verbose:
                print("feature_tensor", feature_tensor.shape)
                print("content_id_tensor", content_id_tensor.shape)

            # ignore state because incorrect prev_ans_correctly val
            _, states = self.model(content_id=content_id_tensor,
                                   bundle_id=None,
                                   feature=feature_tensor,
                                   user_id=None,
                                   mask=seq_len_mask,
                                   initial_state=initial_state)
            self.rnn_state.update_state(user_ids, states)

    def get_state(self, user_ids: List[str]) -> Optional[TensorPair]:
        return self.rnn_state.get_state(user_ids)

    def predict(self, test_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        self.prev_group_test_df = test_df.copy()
        df = test_df.groupby("user_id").apply(self.aggregate)

        user_ids = df.index.to_list()

        # (N, seq)
        content_id_tensor = pad_sequence(df["content_id"].to_list(), batch_first=True)
        row_id_tensor = pad_sequence(df["row_id"].to_list(), batch_first=True)

        # (N, seq, dim)
        feature_tensor = pad_sequence(df["feature"].to_list(), batch_first=True)
        batch_size, seq_len, dim = feature_tensor.shape

        seq_len_mask = (content_id_tensor != 0).to(dtype=torch.uint8)
        is_question_mask = pad_sequence(df["is_question_mask"].to_list(), batch_first=True)

        initial_state = self.get_state(user_ids)
        if "y" in df:
            y = pad_sequence(df["y"].to_list(), batch_first=True)
            y = torch.roll(y, 1, dims=1)
            y[:, 0] = -1
            y = torch.unsqueeze(y, dim=2).float()
        else:
            y = None

        with torch.no_grad():
            if verbose:
                print("feature_tensor", feature_tensor.shape)
                print("content_id_tensor", content_id_tensor.shape)

            pred_logit, states = self.model(content_id=content_id_tensor,
                                            bundle_id=None,
                                            feature=feature_tensor,
                                            user_id=None,
                                            mask=seq_len_mask,
                                            initial_state=initial_state,
                                            ans_prev_correctly=y)
            if y is None:
                self.update_state(user_ids=user_ids, states=states)

        flatten_is_quesion_maks = is_question_mask.view(batch_size * seq_len)
        flatten_seq_mask = seq_len_mask.view(batch_size * seq_len).bool()

        flatten_pred = torch.sigmoid(pred_logit.view(batch_size * seq_len))
        flatten_row_id = row_id_tensor.view(batch_size * seq_len)

        flatten_pred = flatten_pred[flatten_seq_mask & flatten_is_quesion_maks].cpu().data.numpy()
        flatten_row_id = flatten_row_id[flatten_seq_mask & flatten_is_quesion_maks].cpu().data.numpy()

        pred_df = pd.DataFrame({"row_id": flatten_row_id,
                                "answered_correctly": flatten_pred})

        return pred_df
