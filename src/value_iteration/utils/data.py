from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..task.observation_model import ObservationModel
from ..task.transition_model import TransitionModel
from .pytorch_utils import convert_8bit_to_float, make_tensor
from .sampling_functions import sample_random_walk

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


class PomdpDataset(Dataset):
    """Data set contains (o, o') tuples"""

    def __init__(
        self,
        transition_model: TransitionModel,
        observation_model: ObservationModel,
        n: int = 100000,
        n_chains: int = 1,
    ) -> None:
        super().__init__()
        assert n % n_chains == 0

        x, y = [], []
        chain_len = n // n_chains

        for _ in range(n_chains):
            obs = sample_random_walk(chain_len + 1, transition_model, observation_model)
            x.append(obs[:-1])
            y.append(obs[1:])
        self.x = torch.vstack(x)
        self.y = torch.vstack(y)

        self.n = n

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n


class TransitionVaeDataset(Dataset):
    def __init__(
        self,
        observations: torch.tensor,
        actions: torch.tensor,
        successor_obs: torch.tensor,
    ):
        super().__init__()
        self.obs = convert_8bit_to_float(observations)
        self.obsp = convert_8bit_to_float(successor_obs)
        self.actions = actions.float()

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        return self.obs[index], self.actions[index], self.obsp[index]

    def __len__(self):
        return len(self.obs)


class RecurrentVaeDataset(Dataset):
    """Creates a dataset of tuples with the elements
        (obs, actions)

    observations and actions are alligned in time. That is
    the observations and actions are sequences where their indicies
    correspond to the same time. Causally, the observations precede the
    actions.

    where
        -obs is a Batch*Length*Height*Width*Channel tensor
        -actions is a Batch*Length*n_actions tensor
    """

    def __init__(
        self,
        observations: torch.tensor,
        actions: torch.tensor,
        trial_idx: List[int],
        max_sequence_len: int,
    ):
        self.obs = convert_8bit_to_float(observations)
        self.actions = actions.float()
        self.trial_idx = trial_idx
        self.n = max_sequence_len

    def __getitem__(self, index: int) -> Tuple:
        # get the first observation
        start_idx = max(0, index - self.n)

        # filter by trial number
        while self.trial_idx[start_idx] < self.trial_idx[index]:
            start_idx += 1

        return (
            self.obs[start_idx : index + 1],
            self.actions[start_idx : index + 1],
        )

    def __len__(self):
        return len(self.obs)

    @staticmethod
    def collate_fn(batch):
        obs = [x[0] for x in batch]
        actions = [x[1] for x in batch]

        # we want to left-pad the sequences, and the pad_sequence
        # function right pads them.  To do so, we use reverse -> pad -> reverse

        # reverse
        obs = [torch.flip(o, dims=tuple([0])) for o in obs]
        actions = [torch.flip(a, dims=tuple([0])) for a in actions]

        # pad
        obs = pad_sequence(obs, batch_first=True, padding_value=0)
        actions = pad_sequence(actions, batch_first=True, padding_value=0)

        # reverse (dim 1 is sequence after padding)
        obs = torch.flip(obs, dims=tuple([1]))
        actions = torch.flip(actions, dims=tuple([1]))

        # shift the actions by one

        return obs, actions

    @classmethod
    def construct_dataloader(
        cls,
        observations: torch.tensor,
        actions: torch.tensor,
        trial_index: List[int],
        max_sequence_len: int = 10,
        batch_size: int = 64,
    ):
        dataset = cls(observations, actions, trial_index, max_sequence_len)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=cls.collate_fn
        )
