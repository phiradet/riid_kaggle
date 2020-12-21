from abc import ABC
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


class _BaseInference(ABC):

    def __init__(self,
                 model_config: Dict[str, Any],
                 idx_map_dir: str,
                 checkpoint_dir: str,
                 predictor_class: type,
                 initial_state_dir: Optional[str] = None,
                 seq_len: Optional[int] = None,
                 verbose: bool = False):

        self.model_config = model_config
        self.model = predictor_class(**model_config)
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
        self.model.eval()

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
            "bundle_id": instances["bundle_id"]
        }

        if "y" in instances:
            output["y"] = instances["y"]

        return pd.Series(output)
