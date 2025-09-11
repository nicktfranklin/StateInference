from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from value_iteration.utils.pytorch_utils import (
    assert_correct_end_shape,
    maybe_expand_batch,
)


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
        kernel_sizes: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
    ):
        super().__init__()

        if channels is None:
            channels = [32, 64, 64]

        if kernel_sizes is None:
            kernel_sizes = [8, 4, 3]
        if strides is None:
            strides = [4, 2, 1]

        modules = []
        h_in = in_channels
        for h_dim, kernel_size, stride in zip(channels, kernel_sizes, strides):
            modules.append(ConvBlock(h_in, h_dim, kernel_size, stride, padding=1))
            h_in = h_dim

        self.cnn = nn.Sequential(*modules)
        self.embedding_dim = embedding_dim

        with torch.no_grad():
            # Create a dummy input tensor
            dummy_input = torch.zeros(1, *input_shape)
            # Pass the dummy input through the CNN layers
            output = self.cnn(dummy_input)
            # Calculate the output shape
            output_shape = output.shape[1] * output.shape[2] * output.shape[3]

        self.mlp = nn.Sequential(
            nn.Linear(output_shape, embedding_dim),
            nn.ELU(),
        )
        self.input_shape = input_shape

    def _get_cnn_output_shape(self) -> int:
        # Create a dummy input tensor
        dummy_input = torch.zeros(1, *self.input_shape)
        # Pass the dummy input through the CNN layers
        output = self.cnn(dummy_input)
        # Calculate the output shape
        output_shape = output.shape[1] * output.shape[2] * output.shape[3]
        return output_shape

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
        embedding_dim: int = 1024,
        output_channels: int = 1,
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
            nn.Conv2d(
                channels[-1], out_channels=output_channels, kernel_size=3, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        hidden = self.fc(z)
        hidden = hidden.view(-1, self.first_channel_size, 2, 2)
        hidden = self.deconv(hidden)
        observation = self.final_layer(hidden)
        if observation.shape[0] == 1:
            return observation.squeeze(0)
        return observation


class MaskedConv2d(nn.Conv2d):
    """
    Masked 2D convolution where future pixels are not visible.
    This is useful for autoregressive models.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, mask_type="A", stride=1, padding=0
    ):
        super(MaskedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding
        )

        assert mask_type in ["A", "B"], "mask_type must be 'A' or 'B'"
        self.mask_type = mask_type

        # Create the mask
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)

        # Set the center pixel (future pixel) and all future pixels to 0
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, kH // 2 + 1 :] = 0

    def forward(self, x):
        # Apply the mask to the weights
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class AutoregressiveDeconvNet(nn.Module):
    def __init__(
        self,
        embedding_dim=1024,
        hidden_channels=64,
        output_channels=1,
        output_shape=None,  # not used
    ):
        super().__init__()

        # Fully connected layer to transform (batch, 1024) to (batch, hidden_channels * 8 * 8)
        self.fc = nn.Linear(embedding_dim, hidden_channels * 8 * 8)

        # First layer: Reshape and apply Masked convolution (type 'A' mask ensures no dependence on current pixel)
        self.conv1 = MaskedConv2d(
            hidden_channels, hidden_channels, kernel_size=5, padding=2, mask_type="A"
        )

        # Hidden layers: Masked convolutions (type 'B' to allow current pixel but not future ones)
        self.conv2 = MaskedConv2d(
            hidden_channels, hidden_channels, kernel_size=5, padding=2, mask_type="B"
        )
        self.conv3 = MaskedConv2d(
            hidden_channels, hidden_channels, kernel_size=5, padding=2, mask_type="B"
        )

        # Upsampling layers (deconvolutions) to go from 8x8 -> 16x16 -> 32x32 -> 64x64
        self.deconv1 = nn.ConvTranspose2d(
            hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1
        )  # 8x8 -> 16x16
        self.deconv2 = nn.ConvTranspose2d(
            hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1
        )  # 16x16 -> 32x32
        self.deconv3 = nn.ConvTranspose2d(
            hidden_channels // 2, output_channels, kernel_size=4, stride=2, padding=1
        )  # 32x32 -> 64x64

        # Output layer (Optional, depends on your use case)
        self.output_layer = MaskedConv2d(
            output_channels, output_channels, kernel_size=5, padding=2, mask_type="B"
        )

    def forward(self, x):
        # Fully connected layer to reshape the input
        x = self.fc(x)  # (batch, 1024) -> (batch, hidden_channels * 8 * 8)
        x = x.view(-1, 64, 8, 8)  # Reshape to (batch, hidden_channels, 8, 8)

        # Apply convolutional layers with masking
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Perform deconvolution (upsampling)
        x = F.relu(self.deconv1(x))  # (batch, hidden_channels, 16, 16)
        x = F.relu(self.deconv2(x))  # (batch, hidden_channels // 2, 32, 32)
        x = self.deconv3(x)  # (batch, output_channels, 64, 64)

        # Optionally, apply a final masked convolution
        x = self.output_layer(x)

        # # bound outputs to the range 0-1
        # x = torch.sigmoid(x)

        # softclip: bound outputs to the range 0-1 with tanh
        x = torch.tanh(x / 2) * 0.5 + 0.5

        if x.shape[0] == 1:
            return x.squeeze(0)

        return x

    # def forward(self, z):
    #     hidden = self.fc(z)
    #     hidden = hidden.view(
    #         -1, self.first_channel_size, 2, 2
    #     )  # does not effect batching
    #     hidden = self.deconv(hidden)
    #     observation = self.final_layer(hidden)
    #     if observation.shape[0] == 1:
    #         return observation.squeeze(0)
    #     return observation
