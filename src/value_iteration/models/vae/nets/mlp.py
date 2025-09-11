from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
    ):
        super().__init__()
        self.nin = input_size
        self.nout = output_size

        # define a simple MLP neural net
        self.net = []

        d_in = self.nin
        for d_out in hidden_sizes:
            self.net.extend(
                [
                    nn.Linear(d_in, d_out),
                    nn.BatchNorm1d(d_out),
                    nn.ELU(),
                ]
            )
            d_in = d_out

        # final linear layer
        self.net.append(nn.Linear(d_out, self.nout))

        self.net = nn.Sequential(*self.net)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MlpEncoder(nn.Module):
    def __init__(
        self,
        input_shape: int | Tuple[int],
        hidden_sizes: List[int],
        embedding_dim: int,
    ):
        super().__init__()
        if isinstance(input_shape, tuple):
            input_shape = torch.tensor(input_shape).prod().item()

        self.net = MLP(input_shape, hidden_sizes, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.view(x.shape[0], -1))


class MlpDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_sizes: List[int],
        output_shape: int | Tuple[int, int, int],
    ):
        super().__init__()

        if isinstance(output_shape, tuple):
            d_out = torch.tensor(output_shape).prod().item()
        else:
            d_out = output_shape

        self.net = MLP(embedding_dim, hidden_sizes, d_out)
        self.output_shape = output_shape

    def forward(self, x):
        output = F.sigmoid(self.net(x))
        return output.view(output.shape[0], *self.output_shape)


class MlpDecoderWithActions(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        n_actions: int,
        hidden_sizes: List[int],
        ouput_size: int,
    ):
        super().__init__()
        self.mlp = MlpDecoder(
            embedding_size,
            hidden_sizes,
            ouput_size,
        )
        self.latent_embedding = nn.Linear(embedding_size, embedding_size)
        self.action_embedding = nn.Linear(n_actions, embedding_size)

    def forward(self, latents, action):
        x = self.latent_embedding(latents) + self.action_embedding(action)
        x = F.relu(x)
        x = self.mlp(x)
        return x

    def loss(self, latents, actions, targets):
        y_hat = self(latents, actions)
        # return y_hat
        return F.mse_loss(y_hat, torch.flatten(targets, start_dim=1))
