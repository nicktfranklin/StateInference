from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn

from utils.pytorch_utils import DEVICE

from .mlp import MLP


class GruEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        embedding_dims: int,
        n_actions: int,
        gru_kwargs: Optional[Dict[str, Any]] = None,
        batch_first: bool = True,
    ):
        super().__init__()
        gru_kwargs = gru_kwargs if gru_kwargs is not None else dict()
        self.feature_extracter = MLP(input_size, hidden_sizes, embedding_dims)
        self.gru_cell = nn.GRUCell(embedding_dims, embedding_dims, **gru_kwargs)
        self.hidden_encoder = nn.Linear(embedding_dims + n_actions, embedding_dims)
        self.action_encoder = nn.Linear(n_actions, n_actions)
        self.batch_first = batch_first
        self.hidden_size = embedding_dims
        self.n_actions = n_actions
        self.nin = input_size

    def rnn(self, obs: Tensor, h: Tensor) -> Tensor:
        print(obs.shape, h.shape)
        x = self.feature_extracter(obs)
        return self.gru_cell(x, h)

    def forward(
        self,
        obs: Tensor,
        actions: Tensor,
    ):
        obs = torch.flatten(obs, start_dim=2)
        if self.batch_first:
            obs = torch.permute(obs, (1, 0, 2))
            actions = torch.permute(actions, (1, 0, 2))

        n_batch = obs.shape[1]

        # initialize the hidden state and action
        h = torch.zeros(n_batch, self.hidden_size)
        a_prev = torch.zeros(n_batch, self.n_actions)

        # loop through the sequence of observations
        for o, a in zip(obs, actions):
            # encode the hidden state with the previous actions
            h = self.hidden_encoder(torch.concat([h, a_prev], dim=1).to(DEVICE))

            # encode the action for the next step
            a_prev = self.action_encoder(a)

            # pass the observation through the rnn (+ encoder)
            h = self.rnn(o, h)

        return h
