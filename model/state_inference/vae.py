import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import trange

from model.state_inference.constants import (
    INPUT_SHAPE,
    OPTIM_KWARGS,
    VAE_BETA,
    VAE_TAU,
    VAE_TAU_ANNEALING_RATE,
)
from utils.pytorch_utils import (
    DEVICE,
    assert_correct_end_shape,
    check_shape_match,
    gumbel_softmax,
    train,
)

# needed to import the Encoder/Decoder from config
from .nets import *


def unit_test_vae_reconstruction(model, input_shape):
    # Unit Test
    model.eval()
    with torch.no_grad():
        x = torch.rand(*input_shape)
        _, z = model.encode(x)
        x_hat = model.decode(z)
        assert check_shape_match(
            x_hat, input_shape
        ), f"Reconstruction Shape {tuple(x_hat.shape)} Doesn't match Input {input_shape}"


class StateVae(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        z_dim: int,
        z_layers: int,
        beta: float = VAE_BETA,
        tau: float = VAE_TAU,
        tau_annealing_rate: float = VAE_TAU_ANNEALING_RATE,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        tau_is_parameter: bool = False,
        *,
        run_unit_test: bool = True,
    ):
        """
        Note: larger values of beta result in more independent state values
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.z_layers = z_layers
        self.z_dim = z_dim
        self.beta = beta
        self.tau = tau
        self.tau_annealing_rate = tau_annealing_rate
        self.input_shape = input_shape

        if tau_is_parameter:
            self.tau = torch.nn.Parameter(torch.tensor([tau]), requires_grad=True)
            assert (
                self.tau_annealing_rate == 1.0
            ), "Must set tau annealing to 1 for learnable tau"

        if run_unit_test:
            unit_test_vae_reconstruction(self, input_shape)

    def configure_optimizers(
        self,
        optim_kwargs: Optional[Dict[str, Any]] = None,
    ):
        optim_kwargs = optim_kwargs if optim_kwargs is not None else OPTIM_KWARGS
        optimizer = torch.optim.AdamW(self.parameters(), **optim_kwargs)

        return optimizer

    @classmethod
    def make_from_configs(cls, vae_config: Dict[str, Any], env_kwargs: Dict[str, Any]):
        h = env_kwargs["map_height"]
        input_shape = (1, h, h)

        vae_kwargs = vae_config["vae_kwargs"]
        vae_kwargs["input_shape"] = input_shape

        print(sys.modules[__name__])
        Encoder = getattr(sys.modules[__name__], vae_config["encoder_class"])
        Decoder = getattr(sys.modules[__name__], vae_config["decoder_class"])

        encoder_kwargs = vae_config["encoder_kwargs"]
        decoder_kwargs = vae_config["decoder_kwargs"]

        embedding_dim = vae_kwargs["z_dim"] * vae_kwargs["z_layers"]
        encoder_kwargs["embedding_dim"] = embedding_dim
        decoder_kwargs["embedding_dim"] = embedding_dim

        encoder_kwargs["input_shape"] = input_shape
        decoder_kwargs["output_shape"] = input_shape

        encoder = Encoder(**encoder_kwargs)
        decoder = Decoder(**decoder_kwargs)

        return cls(encoder, decoder, **vae_kwargs)

    def reparameterize(self, logits):
        """
        Assume input shape of NxLxD
        """
        assert logits.ndim == 3
        assert logits.shape[1] == self.z_layers
        assert logits.shape[2] == self.z_dim

        # either sample the state or take the argmax
        if self.training:
            z = gumbel_softmax(logits=logits, tau=self.tau, hard=False)
        else:
            s = torch.argmax(logits, dim=-1)  # tensor of n_batch * self.z_n_layers
            z = F.one_hot(s, num_classes=self.z_dim)
        return z

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Assume input shape of NxCxHxW
        """
        # reshape encoder output to (n_batch, z_layers, z_dim)
        logits = self.encoder(x).view(-1, self.z_layers, self.z_dim)
        z = self.reparameterize(logits)
        return logits, z

    def decode(self, z):
        return self.decoder(z.view(-1, self.z_layers * self.z_dim).float())

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        logits, z = self.encode(x)
        x_hat = self.decode(z)
        return (logits, z), x_hat.view(x.shape)  # preserve original shape

    def kl_loss(self, logits):
        """
        Logits are shape (B, N, K), where B is the number of batches, N is the number
            of categorical distributions and where K is the number of classes
        # returns kl-divergence, in nats
        """
        assert logits.ndim == 3
        B, N, K = logits.shape
        logits = logits.view(B * N, K)

        q = Categorical(logits=logits)
        p = Categorical(probs=torch.full((B * N, K), 1.0 / K).to(DEVICE))

        # sum loss over dimensions in each example, average over batch
        kl = kl_divergence(q, p).view(B, N).sum(1).mean()

        return kl

    def recontruction_loss(self, x, x_hat):
        mse_loss = F.mse_loss(x_hat, x, reduction="none")
        # sum loss over dimensions in each example, average over batch
        return mse_loss.view(x.shape[0], -1).sum(1).mean()

    def loss(self, x: Tensor, target: Tensor | None = None) -> Tensor:
        x = x.to(DEVICE).float()
        (logits, _), x_hat = self(x)

        # get the two components of the ELBO loss
        kl_loss = self.kl_loss(logits)

        target = target if target is not None else x
        recon_loss = self.recontruction_loss(target, x_hat)

        return recon_loss + kl_loss * self.beta

    def diagnose_loss(self, x):
        with torch.no_grad():
            x = x.to(DEVICE).float()
            (logits, _), x_hat = self(x)

            # get the two components of the ELBO loss
            kl_loss = self.kl_loss(logits)
            recon_loss = self.recontruction_loss(x, x_hat)

            loss = recon_loss - kl_loss * self.beta
            print(
                f"Total Loss: {loss:.4f}, Reconstruction: {recon_loss:.4f}, "
                + f"KL-Loss: {kl_loss:.4f}, beta: {self.beta}"
            )

    def get_state(self, x) -> np.ndarray:
        """
        Assume input shape of NxCxHxW or CxHxW.
        """
        assert x.ndim <= 4
        assert_correct_end_shape(x, self.input_shape)

        self.eval()
        with torch.no_grad():
            _, z = self.encode(x.to(DEVICE))

            state_vars = torch.argmax(z, dim=-1).detach().cpu().numpy()
        return state_vars

    def decode_state(self, s: Tuple[int]):
        self.eval()
        z = (
            F.one_hot(torch.Tensor(s).to(torch.int64).to(DEVICE), self.z_dim)
            .view(-1)
            .unsqueeze(0)
        )
        with torch.no_grad():
            return self.decode(z).detach().cpu().numpy()

    def anneal_tau(self):
        if self.tau_annealing_rate < 1e-4:
            return
        self.tau *= self.tau_annealing_rate

    def prep_next_batch(self):
        self.anneal_tau()

    def train_epochs(
        self,
        n_epochs: int,
        data_loader: DataLoader,
        optim: Optimizer,
        grad_clip: bool = False,
        progress_bar: bool = False,
    ):
        self.train()

        if progress_bar:
            iterator = trange(n_epochs, desc="Vae Epochs")
        else:
            iterator = range(n_epochs)

        for _ in iterator:
            train(self, data_loader, optim, grad_clip)
            self.prep_next_batch()
