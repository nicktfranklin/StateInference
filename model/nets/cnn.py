from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from utils.pytorch_utils import assert_correct_end_shape, maybe_expand_batch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.ELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.act(x)
        return x


class CnnEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        input_shape: Tuple[int, int, int],
        channels: Optional[List[int]] = None,
    ):
        super().__init__()

        if channels is None:
            channels = [32, 64, 128, 256, 512]

        modules = []
        h_in = in_channels
        for h_dim in channels:
            modules.append(ConvBlock(h_in, h_dim, 3, stride=2, padding=1))
            h_in = h_dim
        self.cnn = nn.Sequential(*modules)
        self.mlp = nn.Sequential(
            nn.LayerNorm(channels[-1] * 4),
            nn.Linear(channels[-1] * 4, channels[-1] * 4),
            nn.ELU(),
            nn.Linear(channels[-1] * 4, embedding_dim),
        )

        assert in_channels == input_shape[0], "Input channels do not match shape!"

        self.input_shape = input_shape

    def forward(self, x):
        # Assume NxCxHxW input or CxHxW input
        assert x.ndim == 4 or x.ndim == 3
        assert_correct_end_shape(x, self.input_shape)
        x = maybe_expand_batch(x, self.input_shape)
        x = self.cnn(x)
        x = self.mlp(torch.flatten(x, start_dim=1))
        return x

    def encode_sequence(self, x: Tensor, batch_first: bool = True) -> Tensor:
        assert x.ndim == 5
        if batch_first:
            x = x.permute(1, 0, 2, 3, 4)
        x = torch.stack([self(xt) for xt in x])
        if batch_first:
            x = x.permute(1, 0, 2, 3, 4)
        return x


class ConvTransposeBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        output_padding=0,
    ):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.act = nn.ELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_t(x)
        x = self.act(x)
        return x


class CnnDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        channel_out: int,
        channels: Optional[List[int]] = None,
        output_shape=None,  # not used
    ):
        super().__init__()

        if channels is None:
            channels = [32, 64, 128, 256, 512][::-1]

        self.fc = nn.Linear(embedding_dim, 4 * channels[0])
        self.first_channel_size = channels[0]

        modules = []
        for ii in range(len(channels) - 1):
            modules.append(
                ConvTransposeBlock(
                    channels[ii],
                    channels[ii + 1],
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
        self.deconv = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                channels[-1],
                channels[-1],
                3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(channels[-1]),
            nn.GELU(),
            nn.Conv2d(channels[-1], out_channels=channel_out, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        hidden = self.fc(z)
        hidden = hidden.view(
            -1, self.first_channel_size, 2, 2
        )  # does not effect batching
        hidden = self.deconv(hidden)
        observation = self.final_layer(hidden)
        if observation.shape[0] == 1:
            return observation.squeeze(0)
        return observation
