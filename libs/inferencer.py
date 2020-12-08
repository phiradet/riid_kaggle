import os
import pickle
import glob
from typing import *

import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

from libs.feature import extract_feature
from libs.model import Predictor


class Inferencer(object):

    def __init__(self,
                 model_config: Dict[str, Any],
                 idx_map_dir: str,
                 checkpoint_dir: str,
                 initial_state_dir: Optional[str] = None,
                 seq_len: Optional[int] = None):

        self.model_config = model_config
        self.model = Predictor(**model_config)

        self.part_idx = pickle.load(open(os.path.join(idx_map_dir, f"part_idx.pickle"), "rb"))
        self.type_idx = pickle.load(open(os.path.join(idx_map_dir, f"type_idx.pickle"), "rb"))
        self.bundle_id_idx = pickle.load(open(os.path.join(idx_map_dir, f"bundle_id_idx.pickle"), "rb"))
        self.content_id_idx = pickle.load(open(os.path.join(idx_map_dir, f"content_id_idx.pickle"), "rb"))
        self.seq_len = seq_len

        if initial_state_dir is None:
            self.training_user_id_idx = {}
            self.h0 = None
            self.c0 = None
        else:
            self.training_user_id_idx = pickle.load(open(os.path.join(initial_state_dir, "user_id_idx.pickle"), "rb"))
            self.h0 = torch.load(os.path.join(initial_state_dir, "h0.pth"))
            self.c0 = torch.load(os.path.join(initial_state_dir, "c0.pth"))

        *_, checkpoint_file = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch=*.ckpt")))
        print("Load weight from", checkpoint_file)
        if torch.cuda.is_available():
            state_dict = torch.load(checkpoint_file)["state_dict"]
        else:
            state_dict = torch.load(checkpoint_file, map_location=torch.device('cpu'))["state_dict"]

        self.model.load_state_dict(state_dict)
        self.model = self.model.eval()

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
            "is_question_mask": instances["is_question_mask"]
        }

        return pd.Series(output)

    def update_state(self, user_ids: List[str], states: torch.tensor):
        pass

    def get_state(self, user_ids: List[str]):
        if len(self.training_user_id_idx) == 0:
            return None
        else:
            raise NotImplementedError

    def predict(self, test_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
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

        with torch.no_grad():
            if verbose:
                print("feature_tensor", feature_tensor.shape)
                print("content_id_tensor", content_id_tensor.shape)
            pred_logit, state = self.model(content_id=content_id_tensor,
                                           bundle_id=None,
                                           feature=feature_tensor,
                                           user_id=None,
                                           mask=seq_len_mask,
                                           initial_state=initial_state)

        self.update_state(user_ids, initial_state)

        flatten_is_quesion_maks = is_question_mask.view(batch_size * seq_len)
        flatten_seq_mask = seq_len_mask.view(batch_size * seq_len).bool()

        flatten_pred = torch.sigmoid(pred_logit.view(batch_size * seq_len))
        flatten_row_id = row_id_tensor.view(batch_size * seq_len)

        flatten_pred = flatten_pred[flatten_seq_mask & flatten_is_quesion_maks].data.numpy()
        flatten_row_id = flatten_row_id[flatten_seq_mask & flatten_is_quesion_maks].data.numpy()

        pred_df = pd.DataFrame({"row_id": flatten_row_id,
                                "answered_correctly": flatten_pred})

        return pred_df
