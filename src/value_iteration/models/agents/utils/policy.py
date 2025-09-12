from typing import Dict

import torch
from stable_baselines3.common.distributions import CategoricalDistribution
from torch import Tensor


class SoftmaxPolicy:
    def __init__(
        self,
        beta: float,
        epsilon: float,
        n_actions: int = 4,
        q_init: float = 1,
    ):
        self.n_actions = n_actions
        self.q_values = dict()
        self.beta = beta
        self.epsilon = epsilon

        self.dist = CategoricalDistribution(action_dim=n_actions)
        self.q_init = {a: q_init for a in range(self.n_actions)}

    def maybe_init_q_values(self, s: int) -> None:
        """
        Initialize a set of q_values if they don't exist, setting them to
        a default value
        """

        if s not in self.q_values:
            if self.q_values:
                q_init = {
                    a: max([max(v.values()) for v in self.q_values.values()])
                    for a in range(self.n_actions)
                }
            else:
                q_init = self.q_init
            self.q_values[s] = q_init

    def get_distribution(self, s: int) -> Tensor:
        def _get_q(s0):
            self.maybe_init_q_values(s0)
            q = torch.tensor(list(self.q_values.get(s0, None).values()))
            return q * self.beta

        q_values = torch.stack([_get_q(s0) for s0 in s])
        return self.dist.proba_distribution(q_values)

    def get_value_function(self) -> Dict[int, float]:
        return {s: max(v.values()) for s, v in self.q_values.items()}
