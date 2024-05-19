from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from task.observation_model import ObservationModel
from task.transition_model import TransitionModel
from utils.pytorch_utils import convert_8bit_to_float, make_tensor
from utils.sampling_functions import sample_random_walk

# todo: Remove this file


class ObservationDataset(Dataset):
    def __init__(
        self,
        transition_model: TransitionModel,
        observation_model: ObservationModel,
        n: int = 10000,
        train: bool = True,
    ) -> Dataset:
        if train:
            self.observations = sample_random_walk(
                n, transition_model, observation_model
            )
        else:
            # for test, use the uncorrupted dataset
            self.observations = torch.stack(
                [
                    make_tensor(observation_model.embed_state(s))
                    for s in range(transition_model.n_states)
                ]
            )

        self.observations = convert_8bit_to_float(self.observations)
        self.observations = self.observations[:, None, ...]
        self.n = self.observations.shape[0]

    def __getitem__(self, idx):
        return self.observations[idx]

    def __len__(self):
        return self.n
