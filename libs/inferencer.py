import os
import pickle
from typing import *

import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from libs.modules.stacked_augmented_lstm import TensorPair

from libs.feature import extract_feature
from libs.model import Predictor
from libs.utils.io import load_state_dict


class RNNState(object):

    def __init__(self,
                 known_user_id_idx: Dict[int, int],
                 h_t: torch.tensor,
                 c_t: torch.tensor,
                 verbose: bool = False):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.known_user_id_idx = known_user_id_idx
        self.h_t = h_t.to(self.device)
        self.c_t = c_t.to(self.device)
        self.verbose = verbose

        if verbose:
            print("Known user:", len(self.known_user_id_idx))
            print("h_t:", self.h_t.shape)
            print("c_t:", self.c_t.shape)

    @classmethod
    def from_file(cls,
                  initial_state_dir: Optional[str],
                  lstm_num_layers: int,
                  lstm_hidden_dim: int,
                  verbose: bool):
        if initial_state_dir is None:
            known_user_id_idx = {}
            h_t = torch.zeros(lstm_num_layers, 0, lstm_hidden_dim)
            c_t = torch.zeros(lstm_num_layers, 0, lstm_hidden_dim)
        else:
            known_user_id_idx = pickle.load(open(os.path.join(initial_state_dir, "user_id_idx.pickle"), "rb"))

            if torch.cuda.is_available():
                h_t = torch.load(os.path.join(initial_state_dir, "h_t.pth"))
                c_t = torch.load(os.path.join(initial_state_dir, "c_t.pth"))
            else:
                h_t = torch.load(os.path.join(initial_state_dir, "h_t.pth"),
                                 map_location=torch.device('cpu'))
                c_t = torch.load(os.path.join(initial_state_dir, "c_t.pth"),
                                 map_location=torch.device('cpu'))

        return cls(known_user_id_idx, h_t, c_t, verbose)

    def update_state(self, user_ids: List[int], states: TensorPair):

        selection_ids = []

        for idx, user_id in enumerate(user_ids):
            if user_id in self.known_user_id_idx:
                selection_ids.append(self.known_user_id_idx[user_id])
            else:
                new_user_idx = len(self.known_user_id_idx)
                selection_ids.append(new_user_idx)
                self.known_user_id_idx[user_id] = new_user_idx

        selection_ids = torch.tensor(selection_ids,
                                     dtype=torch.long,
                                     device=self.device)

        num_layers, known_user_count, hidden_size = self.h_t.shape
        unknown_user_count = len(self.known_user_id_idx) - known_user_count

        zero_tensor = torch.zeros(num_layers,
                                  unknown_user_count,
                                  hidden_size,
                                  dtype=self.h_t.dtype,
                                  device=self.device)
        self.h_t = torch.cat([self.h_t, zero_tensor], dim=1)
        self.c_t = torch.cat([self.c_t, zero_tensor], dim=1)

        self.h_t[:, selection_ids, :] = states[0]
        self.c_t[:, selection_ids, :] = states[1]

        assert self.h_t.shape[1] == len(self.known_user_id_idx)
        assert self.c_t.shape[1] == len(self.known_user_id_idx)

    def get_state(self, user_ids: List[int]) -> Optional[TensorPair]:
        if len(self.known_user_id_idx) == 0:
            return None
        else:
            selection_ids = []

            for idx, user_id in enumerate(user_ids):
                selection_ids.append(self.known_user_id_idx.get(user_id, -1))

            selection_ids = torch.tensor(selection_ids,
                                         dtype=torch.long,
                                         device=self.device)

            known_user_mask = selection_ids >= 0

            if self.verbose:
                print("Known user count", known_user_mask.int().sum())
                print("Unknown user count", (selection_ids < 0).int().sum())

            known_user_h_t = self.h_t[:, selection_ids[known_user_mask], :]
            known_user_c_t = self.c_t[:, selection_ids[known_user_mask], :]

            num_layers, _, hidden_size = self.h_t.shape
            output_h_t = torch.zeros(num_layers, len(user_ids), hidden_size,
                                     dtype=self.h_t.dtype, device=self.device)
            output_c_t = torch.zeros(num_layers, len(user_ids), hidden_size,
                                     dtype=self.c_t.dtype, device=self.device)

            output_h_t[:, known_user_mask, :] = known_user_h_t
            output_c_t[:, known_user_mask, :] = known_user_c_t

            if output_h_t.dtype is not torch.float:
                output_h_t = output_h_t.float()

            if output_c_t.dtype is not torch.float:
                output_c_t = output_c_t.float()

            return output_h_t, output_c_t


class Inferencer(object):

    def __init__(self,
                 model_config: Dict[str, Any],
                 idx_map_dir: str,
                 checkpoint_dir: str,
                 initial_state_dir: Optional[str] = None,
                 seq_len: Optional[int] = None,
                 verbose: bool = False):

        self.model_config = model_config
        self.model = Predictor(**model_config)
        self.verbose = verbose

        if idx_map_dir is not None:
            self.part_idx = pickle.load(open(os.path.join(idx_map_dir, f"part_idx.pickle"), "rb"))
            self.type_idx = pickle.load(open(os.path.join(idx_map_dir, f"type_idx.pickle"), "rb"))
            self.bundle_id_idx = pickle.load(open(os.path.join(idx_map_dir, f"bundle_id_idx.pickle"), "rb"))
            self.content_id_idx = pickle.load(open(os.path.join(idx_map_dir, f"content_id_idx.pickle"), "rb"))

        self.seq_len = seq_len
        self.rnn_state = RNNState.from_file(initial_state_dir=initial_state_dir,
                                            lstm_hidden_dim=self.model_config["lstm_hidden_dim"],
                                            lstm_num_layers=self.model_config["lstm_num_layers"],
                                            verbose=verbose)

        state_dict = load_state_dict(checkpoint_dir)
        self.model.load_state_dict(state_dict)
        self.model = self.model.eval()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)

        self.prev_group_test_df = None

    def aggregate(self, rows: pd.DataFrame) -> pd.Series:
        rows = rows.sort_values("timestamp").reset_index(drop=True)
        instances: Dict[str, torch.tensor] = extract_feature(rows=rows,
                                                             part_idx=self.part_idx,
                                                             type_idx=self.type_idx,
                                                             bundle_id_idx=self.bundle_id_idx,
                                                             content_id_idx=self.content_id_idx,
                                                             seq_len=self.seq_len,
                                                             to_sparse=False)
        output = {
            "content_id": instances["content_id"],
            "feature": instances["feature"],
            "row_id": torch.tensor(rows["row_id"].values, dtype=torch.int),
            "is_question_mask": instances["is_question_mask"],
        }

        if "y" in instances:
            output["y"] = instances["y"]

        return pd.Series(output)

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
