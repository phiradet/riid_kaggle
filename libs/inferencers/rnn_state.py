import os
import pickle
from typing import Dict, Optional, List

import torch

from libs.modules.stacked_augmented_lstm import TensorPair


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
            print("Load RNN state from", initial_state_dir)
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