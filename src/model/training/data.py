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
    def __init__(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor | None = None,
        rewards: torch.Tensor | None = None,
    ):
        self.obs = obs
        self.actions = actions
        self.next_obs = next_obs
        self.rewards = rewards

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        output = {"obs": self.obs[idx], "actions": self.actions[idx]}
        if self.next_obs is not None:
            output["next_obs"] = self.next_obs[idx]
        if self.rewards is not None:
            output["rewards"] = self.rewards[idx]
        return output

    def collate_fn(self, batch):
        return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}


# from .utils.tabular_agents import ModelFreeAgent


class ActorCriticDataset(torch.utils.data.Dataset):
    def __init__(self, states, next_states, values, rewards, actions):
        self.s = states
        self.sp = next_states
        self.a = actions
        self.r = rewards
        self.v = values

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        return {
            "state": self.s[idx],
            "next_state": self.sp[idx],
            "action": self.a[idx],
            "reward": self.r[idx],
            "values": self.v[idx],
        }
