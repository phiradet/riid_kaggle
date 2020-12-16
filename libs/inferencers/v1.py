import os
import pickle
from typing import *

import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

from libs.inferencers.rnn_state import RNNState
from libs.modules.stacked_augmented_lstm import TensorPair

from libs.feature import extract_feature
from libs.models.v1 import V1Predictor
from libs.utils.io import load_state_dict
from libs.inferencers._base import _BaseInference


class V1Inferencer(_BaseInference):

    def __init__(self,
                 model_config: Dict[str, Any],
                 idx_map_dir: str,
                 checkpoint_dir: str,
                 initial_state_dir: Optional[str] = None,
                 seq_len: Optional[int] = None,
                 verbose: bool = False):
        super(V1Inferencer, self).__init__(model_config,
                                           idx_map_dir,
                                           checkpoint_dir,
                                           initial_state_dir,
                                           seq_len,
                                           verbose)
        self.previous_input = dict()

    def _ans_to_tensor(self, rows: pd.Series):
        return torch.tensor(rows.to_list(),
                            dtype=torch.float, device=self.device)

    def update_rnn_state(self):
        raise NotImplementedError

    def update_last_seen_content_state(self,
                                       last_content_feedback: Optional[torch.Tensor]):
        user_ids = self.previous_input["user_ids"]

    def update_state(self, prior_batch_ans_correct: Optional[List[int]]):
        """
        Update last seen content and RNN state
        """

        # first inference run, do nothing
        if len(self.previous_input) > 0:

            n = len(self.prev_group_test_df)
            if prior_batch_ans_correct is not None and len(prior_batch_ans_correct) == n:
                self.prev_group_test_df["answer"] = prior_batch_ans_correct

                df = self.prev_group_test_df[["user_id", "answer"]] \
                    .groupby("user_id", sort=True, group_keys=True) \
                    .agg({"answer": self._ans_to_tensor})

                user_ids = df.index.to_list()
                no_padding_seq_len = torch.tensor(df["answer"].str.len().values,
                                                  dtype=torch.uint8,
                                                  device=self.device)
                # (batch, seq)
                prev_group_y = pad_sequence(df["answer"].to_list(),
                                            batch_first=True)
                batch_size, seq_len = prev_group_y.shape
                prev_group_feedback = V1Predictor.to_seen_content_feedback(prev_group_y,
                                                                           do_shift=False)
                # (batch_size, 3)
                last_content_feedback = prev_group_y[torch.arange(batch_size),
                                                     no_padding_seq_len-1]
                self.update_last_seen_content_state(last_content_feedback)

                # ===== update prev ans =====
                prev_seen_content_feedback = self.previous_input["seen_content_feedback"]
                prev_batch, prev_seq_len = prev_seen_content_feedback.shape

                # the feedback of the first seen content is either correct or unknown
                if prev_seq_len > 1:
                    prev_seen_content_feedback[:, 1:] = prev_group_feedback[:, :-1]
                    self.previous_input["seen_content_feedback"] = prev_seen_content_feedback
        else:
            self.update_last_seen_content_state(None)

        # at this state, all input are completed (as much as the data allow)
        self.update_rnn_state()

    def predict(self, test_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        prior_batch_answer_correctly = test_df["prior_group_answers_correct"].iloc[0]
        if isinstance(prior_batch_answer_correctly, str):
            prior_batch_answer_correctly = eval(prior_batch_answer_correctly)
        self.update_state(prior_batch_answer_correctly)

        self.prev_group_test_df = test_df.copy()
        df = test_df \
            .groupby("user_id", sort=True, group_keys=True) \
            .apply(self.aggregate)
        # user_ids is always sorted
        user_ids = df.index.to_list()

        # (N, seq)
        content_id_tensor = pad_sequence(df["content_id"].to_list(), batch_first=True)
        row_id_tensor = pad_sequence(df["row_id"].to_list(), batch_first=True)

        # (N, seq, dim)
        feature_tensor = pad_sequence(df["feature"].to_list(), batch_first=True)
        batch_size, seq_len, dim = feature_tensor.shape

        seq_len_mask = (content_id_tensor != 0).to(dtype=torch.uint8)
        is_question_mask = pad_sequence(df["is_question_mask"].to_list(), batch_first=True)

        initial_state = self.rnn_state.get_state(user_ids)
        model: V1Predictor = self.model

        with torch.no_grad():
            if verbose:
                print("feature_tensor", feature_tensor.shape)
                print("content_id_tensor", content_id_tensor.shape)

            model_input = dict(query_content_id=content_id_tensor,
                               query_content_feature=feature_tensor,
                               mask=seq_len_mask,
                               seen_content_id=seen_content_id,
                               seen_content_feature=seen_content_feature,
                               seen_content_feedback=seen_content_feedback,
                               initial_state=initial_state)
            pred_logit, states = model.forward(**model_input)

            model_input["user_ids"] = user_ids
            self.previous_input = model_input

        flatten_is_quesion_maks = is_question_mask.view(batch_size * seq_len)
        flatten_seq_mask = seq_len_mask.view(batch_size * seq_len).bool()

        flatten_pred = torch.sigmoid(pred_logit.view(batch_size * seq_len))
        flatten_row_id = row_id_tensor.view(batch_size * seq_len)

        flatten_pred = flatten_pred[flatten_seq_mask & flatten_is_quesion_maks].cpu().data.numpy()
        flatten_row_id = flatten_row_id[flatten_seq_mask & flatten_is_quesion_maks].cpu().data.numpy()

        pred_df = pd.DataFrame({"row_id": flatten_row_id,
                                "answered_correctly": flatten_pred})

        return pred_df
