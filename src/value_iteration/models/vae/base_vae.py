from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from torch import FloatTensor, Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import trange

from value_iteration.utils.pytorch_utils import DEVICE, assert_correct_end_shape

from .constants import OPTIM_KWARGS


class StateVae(nn.Module, ABC):
    """
    Abstract base class for State VAE models.

    Provides common interface and shared functionality for different
    VAE implementations (Gumbel Softmax, FSQ, etc.).
    """

    def __init__(self):
        super().__init__()
        self.input_shape: Tuple[int, int, int] = None
        self.encoder = None
        self.decoder = None

    @abstractmethod
    def encode(self, x: Tensor) -> Tuple[Tensor, ...]:
        """
        Encode input tensor to latent representation.

        Args:
            x: Input tensor

        Returns:
            Tuple containing encoded representations
        """
        pass

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent representation to reconstruction.

        Args:
            z: Latent tensor

        Returns:
            Reconstructed tensor
        """
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tuple[Tuple[Tensor, ...], Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x: Input tensor

        Returns:
            Tuple of (latent_info, reconstruction)
        """
        pass

    @abstractmethod
    def loss(self, x: FloatTensor, target: FloatTensor = None) -> FloatTensor:
        """
        Compute the total loss for the VAE.

        Args:
            x: Input tensor
            target: Target tensor (defaults to x for autoencoding)

        Returns:
            Total loss
        """
        pass

    def configure_optimizers(
        self,
        optim_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Configure optimizer for training."""
        optim_kwargs = optim_kwargs if optim_kwargs is not None else OPTIM_KWARGS
        optimizer = torch.optim.AdamW(self.parameters(), **optim_kwargs)
        return optimizer

    def reconstruction_loss(self, x: FloatTensor, x_hat: FloatTensor) -> FloatTensor:
        """Compute reconstruction loss (MSE)."""
        mse_loss = torch.nn.functional.mse_loss(x_hat, x, reduction="none")
        return mse_loss.view(x.shape[0], -1).sum(1).mean()

    @abstractmethod
    def get_states(self, x):
        """
        Get discrete state representation from input.

        Args:
            x: Input tensor

        Returns:
            Discrete state representation
        """
        pass

    @abstractmethod
    def sample_state(self, x):
        """
        Sample state representation from input.

        Args:
            x: Input tensor

        Returns:
            Sampled state representation
        """
        pass

    @abstractmethod
    def decode_state(self, state):
        """
        Decode state representation back to observation space.

        Args:
            state: State representation

        Returns:
            Decoded observation
        """
        pass

    @abstractmethod
    def diagnose_loss(self, x):
        """Print diagnostic information about loss components."""
        pass

    def fit(
        self,
        n_epochs: int,
        train_dataloader: DataLoader,
        *,
        test_dataloader: DataLoader | None = None,
        optim: Optimizer | None = None,
        grad_clip: int | None = None,
        progress_bar: bool = False,
        verbose: bool = False,
    ):
        """
        Shared training loop implementation.
        """
        assert not (verbose and progress_bar)

        if progress_bar:
            iterator = trange(n_epochs, desc="VAE Epochs")
        else:
            iterator = range(n_epochs)

        optim = optim if optim is not None else self.configure_optimizers()

        train_losses, test_losses = [], []
        for epoch in iterator:

            # Training
            self.train()
            for batch in train_dataloader:
                optim.zero_grad()
                loss = self.loss(batch)
                loss.backward()

                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

                optim.step()
                train_losses.append(loss.detach().cpu().item())

            # Handle any epoch-end processing
            self._post_epoch_processing()

            # Validation
            if test_dataloader is None:
                continue
            self.eval()
            with torch.no_grad():
                test_loss = 0
                for batch in test_dataloader:
                    loss = self.loss(batch)
                    test_loss += loss * test_dataloader.batch_size
                avg_loss = test_loss / len(test_dataloader.dataset)
                test_losses.append(avg_loss.item())

            if verbose:
                print(f"Epoch {epoch}, Loss (test) {test_losses[-1]:.6f}")

        return train_losses, test_losses

    def _post_epoch_processing(self):
        """
        Hook for subclasses to perform post-epoch processing.
        Default implementation does nothing.
        """
        pass

    @staticmethod
    def _load_configs(
        vae_config_path: str, env_config_path: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load VAE and environment configuration files.

        Args:
            vae_config_path: Absolute path to VAE config file
            env_config_path: Absolute path to environment/task config file

        Returns:
            Tuple of (vae_config_dict, env_kwargs_dict)
        """
        vae_path = Path(vae_config_path)
        env_path = Path(env_config_path)

        # Check files exist
        if not vae_path.exists():
            raise FileNotFoundError(f"VAE config not found: {vae_config_path}")
        if not env_path.exists():
            raise FileNotFoundError(f"Environment config not found: {env_config_path}")

        # Load environment config
        with open(env_path, "r") as f:
            env_yaml = yaml.safe_load(f)
        env_kwargs = env_yaml.get("env_kwargs", {})

        # Load VAE config
        with open(vae_path, "r") as f:
            vae_yaml = yaml.safe_load(f)

        return vae_yaml, env_kwargs
