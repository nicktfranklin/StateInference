from dataclasses import dataclass

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import VecEnv


@dataclass
class ValueIterationConfig(nn.Module):
    pass


class ValueIteration(nn.Module):
    def __init__(self, task: VecEnv | gym.Env, config: ValueIterationConfig):
        super(ValueIteration, self).__init__()
        self.config = config

    def forward(self, x):
        pass
