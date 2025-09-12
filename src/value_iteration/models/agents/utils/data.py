from typing import Dict, List, NamedTuple

import torch
from src.model.agents.stable_baseline_clone.buffers import RolloutBuffer
from torch import FloatTensor


class PpoDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

    def __len__(self):
        return len(self.data["observations"])

    def collate_fn(batch):
        return {
            "observations": torch.stack([item["observations"] for item in batch]),
            "next_observations": torch.stack(
                [item["next_observations"] for item in batch]
            ),
            "rewards_to_go": torch.cat([item["rewards_to_go"] for item in batch]),
            "advantages": torch.cat([item["advantages"] for item in batch]),
            "actions": torch.cat([item["actions"] for item in batch]),
            "log_probs": torch.cat([item["log_probs"] for item in batch]),
        }


class ViPpoRolloutSample(NamedTuple):
    observations: FloatTensor
    next_observations: FloatTensor
    actions: FloatTensor
    old_values: FloatTensor
    old_log_prob: FloatTensor
    advantages: FloatTensor
    returns: FloatTensor
    dones: FloatTensor
    rewards: FloatTensor
    vi_estimates: FloatTensor


class ViPpoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rollout_buffer: RolloutBuffer,
        vi_estimates: torch.Tensor,
        device: torch.device | None = None,
    ):

        self.observations = rollout_buffer.observations
        self.next_observations = rollout_buffer.next_observations
        self.actions = rollout_buffer.actions
        self.values = rollout_buffer.values
        self.log_probs = rollout_buffer.log_probs
        self.advantages = rollout_buffer.advantages
        self.returns = rollout_buffer.returns
        self.dones = rollout_buffer.dones
        self.rewards = rollout_buffer.rewards
        self.vi_estimates = vi_estimates

        self.device = device if device else torch.device("cpu")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        out = {
            "observations": self.observations[index],
            "next_observations": self.next_observations[index],
            "actions": self.actions[index],
            "old_values": self.values[index],
            "old_log_prob": self.log_probs[index],
            "advantages": self.advantages[index],
            "returns": self.returns[index],
            "dones": self.dones[index],
            "rewards": self.rewards[index],
            "vi_estimates": self.vi_estimates[index],
        }
        return {k: torch.as_tensor(v, device=self.device) for k, v in out.items()}

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = {k: torch.stack([d[k] for d in batch]) for k in batch[0].keys()}
        return ViPpoRolloutSample(**batch)
