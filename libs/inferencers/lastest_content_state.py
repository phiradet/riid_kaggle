import os
import pickle
from typing import *

import torch


class LatestContentState(object):

    def __init__(self,
                 known_user_id_idx: Dict[int, int],
                 content_id: torch.Tensor,
                 content_feature: torch.Tensor,
                 content_feedback: torch.Tensor,
                 bundle_id: torch.Tensor,
                 verbose: bool = False):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.known_user_id_idx = known_user_id_idx
        self.content_id = content_id.to(self.device)
        self.content_feature = content_feature.to(self.device)
        self.content_feedback = content_feedback.to(self.device)
        self.bundle_id = bundle_id.to(self.device)

        self.verbose = verbose

        _, self.feature_dim = self.content_feature.shape

        if verbose:
            print("Known user:", len(self.known_user_id_idx))
            print("content_id:", self.content_id.shape)
            print("content_feature:", self.content_feature.shape)
            print("content_feedback:", self.content_feedback.shape)
            print("bundle_id:", self.bundle_id.shape)

    @classmethod
    def from_file(cls,
                  data_dir: Optional[str] = None,
                  verbose: bool = False,
                  feature_dim: int = 204,
                  feedback_dim: int = 3):
        if data_dir is None or not cls.found_seen_content_files(data_dir):
            known_user_id_idx = {}
            batch_size = 0
            content_id = torch.zeros(batch_size, 1, dtype=torch.long)
            content_feature = torch.zeros(batch_size, feature_dim, dtype=torch.float)
            content_feedback = torch.zeros(batch_size, feedback_dim, dtype=torch.float)
            bundle_id = torch.zeros(batch_size, 1, dtype=torch.long)
            print("Init empty seen content state")
        else:
            print("Load seen content state from", data_dir)
            known_user_id_idx = pickle.load(open(os.path.join(data_dir, "user_id_idx.pickle"), "rb"))

            if torch.cuda.is_available():
                bundle_id = torch.load(os.path.join(data_dir, "bundle_id.pth"))
                content_id = torch.load(os.path.join(data_dir, "content_id.pth"))
                content_feature = torch.load(os.path.join(data_dir, "content_feature.pth"))
                content_feedback = torch.load(os.path.join(data_dir, "content_feedback.pth"))
            else:
                bundle_id = torch.load(os.path.join(data_dir, "bundle_id.pth"),
                                       map_location=torch.device('cpu'))
                content_id = torch.load(os.path.join(data_dir, "content_id.pth"),
                                        map_location=torch.device('cpu'))
                content_feature = torch.load(os.path.join(data_dir, "content_feature.pth"),
                                             map_location=torch.device('cpu'))
                content_feedback = torch.load(os.path.join(data_dir, "content_feedback.pth"),
                                              map_location=torch.device('cpu'))

        return cls(known_user_id_idx,
                   content_id,
                   content_feature,
                   content_feedback,
                   bundle_id,
                   verbose)

    @staticmethod
    def found_seen_content_files(data_dir: str):
        return os.path.exists(os.path.join(data_dir, "content_id.pth")) and \
               os.path.exists(os.path.join(data_dir, "content_feature.pth")) and \
               os.path.exists(os.path.join(data_dir, "bundle_id.pth")) and \
               os.path.exists(os.path.join(data_dir, "content_feedback.pth"))

    @staticmethod
    def _get_last_element(tensor: torch.Tensor,
                          seq_len_mask: torch.Tensor):
        last_ele_index = torch.sum(seq_len_mask, dim=1) - 1
        batch_size, *_ = tensor.shape
        return tensor[torch.arange(batch_size), last_ele_index]

    def get_state(self, user_ids: List[int]):
        if len(self.known_user_id_idx) == 0:
            batch_size = len(user_ids)
            content_id = torch.zeros(batch_size, 1,
                                     dtype=self.content_id.dtype,
                                     device=self.device)
            content_feature = torch.zeros(batch_size, self.feature_dim,
                                          dtype=self.content_feature.dtype,
                                          device=self.device)
            content_feedback = torch.zeros(batch_size, 3,
                                           dtype=self.content_feedback.dtype,
                                           device=self.device)
            bundle_id = torch.zeros(batch_size, 1,
                                    dtype=self.bundle_id.dtype,
                                    device=self.device)

            output = [content_id, content_feature, content_feedback, bundle_id]
        else:
            selection_ids = []

            for idx, user_id in enumerate(user_ids):
                selection_ids.append(self.known_user_id_idx.get(user_id, -1))

            selection_ids = torch.tensor(selection_ids,
                                         dtype=torch.long,
                                         device=self.device)

            known_user_mask = selection_ids >= 0

            output = []
            src_tensors = [self.content_id,
                           self.content_feature,
                           self.content_feedback,
                           self.bundle_id]

            for src in src_tensors:
                _, dim = src.shape
                o_tensor = torch.zeros(len(user_ids), dim,
                                       dtype=src.dtype,
                                       device=self.device)
                o_tensor[known_user_mask] = src[selection_ids[known_user_mask]]
                output.append(o_tensor)

        return output

    def update_state(self,
                     user_ids: List[int],
                     content_id: torch.Tensor,
                     content_feature: torch.Tensor,
                     bundle_id: torch.Tensor,
                     last_content_feedback: torch.Tensor):
        """
        :param user_ids:
        :param content_id: (batch, seq, 1)
        :param content_feature: (batch, seq, dim)
        :param last_content_feedback: (batch, 3)
        :param bundle_id: (batch, seq, 1)
        :return:
        """
        mask = content_id > 0

        # (batch, 1)
        last_content_id = self.__class__._get_last_element(content_id, mask)
        last_content_id = torch.unsqueeze(last_content_id, dim=1)

        # (batch, 1)
        last_bundle_id = self.__class__._get_last_element(bundle_id, mask)
        last_bundle_id = torch.unsqueeze(last_bundle_id, dim=1)

        # (batch, dim)
        last_content_feature = self.__class__._get_last_element(content_feature, mask)

        selection_ids = []
        unknown_user_count = 0

        for idx, user_id in enumerate(user_ids):
            if user_id in self.known_user_id_idx:
                selection_ids.append(self.known_user_id_idx[user_id])
            else:
                new_user_idx = len(self.known_user_id_idx)
                selection_ids.append(new_user_idx)
                self.known_user_id_idx[user_id] = new_user_idx
                unknown_user_count += 1

        selection_ids = torch.tensor(selection_ids,
                                     dtype=torch.long,
                                     device=self.device)

        self.content_id = torch.cat([self.content_id,
                                     torch.zeros(unknown_user_count, 1,
                                                 dtype=torch.long,
                                                 device=self.device)])
        self.content_id[selection_ids] = last_content_id

        self.bundle_id = torch.cat([self.bundle_id,
                                    torch.zeros(unknown_user_count, 1,
                                                dtype=torch.long,
                                                device=self.device)])
        self.bundle_id[selection_ids] = last_bundle_id

        self.content_feature = torch.cat([self.content_feature,
                                          torch.zeros(unknown_user_count, self.feature_dim,
                                                      dtype=torch.float,
                                                      device=self.device)])
        self.content_feature[selection_ids] = last_content_feature

        self.content_feedback = torch.cat([self.content_feedback,
                                           torch.zeros(unknown_user_count, 3,
                                                       dtype=torch.float,
                                                       device=self.device)])
        self.content_feedback[selection_ids] = last_content_feedback
