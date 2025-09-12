from typing import Any, Dict

import torch
from torch.utils.data import Dataset


class MdpDataset(Dataset):
    def __init__(self, dataset: Dict[str, Any]):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset["observations"])

    def __getitem__(self, idx):
        return {
            "observations": self.dataset["observations"][idx],
            "next_observations": self.dataset["next_observations"][idx],
            "actions": self.dataset["actions"][idx],
            "rewards": self.dataset["rewards"][idx],
            "dones": self.dataset["terminated"][idx] or self.dataset["timouts"][idx],
        }


class VaeDataset(torch.utils.data.Dataset):
    def __init__(self, obs: torch.Tensor, next_obs: torch.Tensor | None = None):
        self.obs = obs
        self.next_obs = next_obs

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if self.next_obs is None:
            return self.obs[idx]
        return self.obs[idx], self.next_obs[idx]
