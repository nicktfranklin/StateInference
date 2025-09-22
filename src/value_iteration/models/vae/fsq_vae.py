"""
FSQ-based VAE implementation using Finite Scalar Quantization.

This module provides a VAE that uses FSQ instead of Gumbel Softmax for quantization.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from pydantic import BaseModel, Field, field_validator
from torch import FloatTensor, Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import trange

from value_iteration.utils.pytorch_utils import DEVICE, assert_correct_end_shape

from .constants import OPTIM_KWARGS
from .fsq import FSQ, calculate_codebook_size
from .nets import (
    AutoregressiveDeconvNet,
    CnnDecoder,
    CnnEncoder,
    MlpDecoder,
    MlpEncoder,
)
from .vae import DecoderConfig, EncoderConfig, unit_test_vae_reconstruction

# Class registries for encoders and decoders
ENCODER_REGISTRY = {
    "CnnEncoder": CnnEncoder,
    "MlpEncoder": MlpEncoder,
}

DECODER_REGISTRY = {
    "CnnDecoder": CnnDecoder,
    "MlpDecoder": MlpDecoder,
    "AutoregressiveDeconvNet": AutoregressiveDeconvNet,
}


class FSQVAEConfig(BaseModel):
    """Configuration for FSQ VAE model parameters."""

    fsq_levels: List[int] = Field(..., description="FSQ levels for each dimension")
    beta: float = Field(
        default=1.0, description="Beta parameter for commitment loss weighting"
    )
    commitment_cost: float = Field(
        default=0.25, description="Commitment cost for encoder training"
    )
    input_shape: Optional[Tuple[int, int, int]] = None
    run_unit_test: bool = Field(
        default=False, description="Run unit test during initialization"
    )

    @field_validator("fsq_levels")
    @classmethod
    def validate_fsq_levels(cls, v):
        if not v or not all(isinstance(level, int) and level > 0 for level in v):
            raise ValueError("fsq_levels must be a non-empty list of positive integers")
        return v

    @property
    def codebook_size(self) -> int:
        """Calculate the total codebook size."""
        return calculate_codebook_size(self.fsq_levels)

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension (number of FSQ dimensions)."""
        return len(self.fsq_levels)


class StateFSQVAEConfig(BaseModel):
    """Complete configuration for StateFSQVAE model."""

    encoder_class: str = Field(..., description="Name of the encoder class")
    decoder_class: str = Field(..., description="Name of the decoder class")
    encoder_kwargs: EncoderConfig = Field(default_factory=EncoderConfig)
    decoder_kwargs: DecoderConfig = Field(default_factory=DecoderConfig)
    vae_kwargs: FSQVAEConfig = Field(..., description="FSQ VAE model parameters")

    def prepare_for_instantiation(self, env_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for model instantiation with environment parameters."""

        # Calculate input shape from environment
        h = env_kwargs["map_height"]
        input_shape = (1, h, h)

        # Update VAE kwargs with input shape
        vae_kwargs = self.vae_kwargs.model_copy()
        vae_kwargs.input_shape = input_shape

        # Get embedding dimension from FSQ levels
        embedding_dim = vae_kwargs.embedding_dim

        # Update encoder kwargs
        encoder_kwargs = self.encoder_kwargs.model_copy()
        encoder_kwargs.embedding_dim = embedding_dim
        encoder_kwargs.input_shape = input_shape

        # Update decoder kwargs
        decoder_kwargs = self.decoder_kwargs.model_copy()
        decoder_kwargs.embedding_dim = embedding_dim
        decoder_kwargs.output_shape = input_shape

        return {
            "encoder_class": self.encoder_class,
            "decoder_class": self.decoder_class,
            "encoder_kwargs": encoder_kwargs.model_dump(exclude_none=True),
            "decoder_kwargs": decoder_kwargs.model_dump(exclude_none=True),
            "vae_kwargs": vae_kwargs.model_dump(exclude_none=True),
        }

    @classmethod
    def from_yaml_dict(cls, config_dict: Dict[str, Any]) -> "StateFSQVAEConfig":
        """Create StateFSQVAEConfig from YAML dictionary format."""
        return cls(
            encoder_class=config_dict["encoder_class"],
            decoder_class=config_dict["decoder_class"],
            encoder_kwargs=EncoderConfig(**config_dict.get("encoder_kwargs", {})),
            decoder_kwargs=DecoderConfig(**config_dict.get("decoder_kwargs", {})),
            vae_kwargs=FSQVAEConfig(**config_dict["vae_kwargs"]),
        )


class StateFSQVAE(nn.Module):
    """
    State VAE using Finite Scalar Quantization (FSQ) instead of Gumbel Softmax.

    This provides a simpler and more stable alternative to VQ-VAE and Gumbel Softmax
    for learning discrete latent representations.
    """

    def __init__(
        self,
        config: StateFSQVAEConfig,
        env_kwargs: Dict[str, Any],
    ):
        """
        Initialize StateFSQVAE from Pydantic config.

        Args:
            config: StateFSQVAEConfig containing all model configuration
            env_kwargs: Environment-specific parameters (e.g., map_height)
        """
        super().__init__()

        # Prepare configuration with environment-specific parameters
        prepared_config = config.prepare_for_instantiation(env_kwargs)

        # Get encoder and decoder classes from registries
        encoder_class = prepared_config["encoder_class"]
        decoder_class = prepared_config["decoder_class"]

        if encoder_class not in ENCODER_REGISTRY:
            raise ValueError(
                f"Unknown encoder class: {encoder_class}. Available: {list(ENCODER_REGISTRY.keys())}"
            )
        if decoder_class not in DECODER_REGISTRY:
            raise ValueError(
                f"Unknown decoder class: {decoder_class}. Available: {list(DECODER_REGISTRY.keys())}"
            )

        Encoder = ENCODER_REGISTRY[encoder_class]
        Decoder = DECODER_REGISTRY[decoder_class]

        # Create encoder and decoder instances
        self.encoder = Encoder(**prepared_config["encoder_kwargs"])
        self.decoder = Decoder(**prepared_config["decoder_kwargs"])

        # Set VAE parameters from config
        vae_kwargs = prepared_config["vae_kwargs"]
        self.fsq_levels = vae_kwargs["fsq_levels"]
        self.beta = vae_kwargs["beta"]
        self.commitment_cost = vae_kwargs["commitment_cost"]
        self.input_shape = tuple(vae_kwargs["input_shape"])

        # Create FSQ quantizer
        self.fsq = FSQ(self.fsq_levels)
        self.embedding_dim = len(self.fsq_levels)
        self.codebook_size = self.fsq.codebook_size

        if vae_kwargs["run_unit_test"]:
            unit_test_vae_reconstruction(self, self.input_shape)

    def configure_optimizers(
        self,
        optim_kwargs: Optional[Dict[str, Any]] = None,
    ):
        optim_kwargs = optim_kwargs if optim_kwargs is not None else OPTIM_KWARGS
        optimizer = torch.optim.AdamW(self.parameters(), **optim_kwargs)
        return optimizer

    @classmethod
    def make_from_configs(cls, vae_config: Dict[str, Any], env_kwargs: Dict[str, Any]):
        """Create from dictionary configs."""
        config = StateFSQVAEConfig.from_yaml_dict(vae_config)
        return cls(config, env_kwargs)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Encode input and quantize using FSQ.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tuple of (continuous_latents, quantized_latents, indices)
        """
        # Get continuous latents from encoder
        z_continuous = self.encoder(x).view(-1, self.embedding_dim)

        # Quantize using FSQ
        z_quantized, indices = self.fsq(z_continuous)

        return z_continuous, z_quantized, indices

    def decode(self, z: Tensor) -> Tensor:
        """Decode quantized latents."""
        return self.decoder(z.float())

    def forward(self, x: Tensor) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]:
        """
        Forward pass of the FSQ VAE.

        Args:
            x: Input tensor

        Returns:
            Tuple of ((continuous_latents, quantized_latents, indices), reconstruction)
        """
        z_continuous, z_quantized, indices = self.encode(x)
        x_hat = self.decode(z_quantized)
        return (z_continuous, z_quantized, indices), x_hat.view(x.shape)

    def commitment_loss(self, z_continuous: Tensor, z_quantized: Tensor) -> Tensor:
        """
        Compute commitment loss to encourage encoder to commit to quantized values.

        Args:
            z_continuous: Continuous latents from encoder
            z_quantized: Quantized latents from FSQ

        Returns:
            Commitment loss scalar
        """
        return F.mse_loss(z_continuous, z_quantized.detach())

    def reconstruction_loss(self, x: FloatTensor, x_hat: FloatTensor) -> FloatTensor:
        """Compute reconstruction loss."""
        mse_loss = F.mse_loss(x_hat, x, reduction="none")
        # Sum loss over dimensions in each example, average over batch
        return mse_loss.view(x.shape[0], -1).sum(1).mean()

    def loss(self, x: FloatTensor, target: FloatTensor = None) -> FloatTensor:
        """
        Compute total loss for FSQ VAE.

        Args:
            x: Input tensor
            target: Target tensor (defaults to x for autoencoding)

        Returns:
            Total loss
        """
        x = x.to(DEVICE).float()
        (z_continuous, z_quantized, _), x_hat = self(x)

        if target is None:
            target = x

        # Reconstruction loss
        recon_loss = self.reconstruction_loss(target, x_hat)

        # Commitment loss (encourages encoder to commit to quantized values)
        commit_loss = self.commitment_loss(z_continuous, z_quantized)

        return recon_loss + self.beta * commit_loss

    def diagnose_loss(self, x):
        """Diagnose loss components."""
        with torch.no_grad():
            x = x.to(DEVICE).float()
            (z_continuous, z_quantized, _), x_hat = self(x)

            recon_loss = self.reconstruction_loss(x, x_hat)
            commit_loss = self.commitment_loss(z_continuous, z_quantized)

            total_loss = recon_loss + self.beta * commit_loss
            print(
                f"Total Loss: {total_loss:.4f}, Reconstruction: {recon_loss:.4f}, "
                + f"Commitment: {commit_loss:.4f}, beta: {self.beta}"
            )

    def get_states(self, x):
        """
        Get discrete state representation from input.

        Args:
            x: Input tensor

        Returns:
            Discrete state indices
        """
        assert x.ndim <= 4
        assert_correct_end_shape(x, self.input_shape)

        self.eval()
        with torch.no_grad():
            _, _, indices = self.encode(x.to(DEVICE))
            return indices.detach().cpu()

    def sample_state(self, x):
        """
        For FSQ, sampling is deterministic, so this is the same as get_states.
        """
        return self.get_states(x)

    def decode_state(self, state_index: int):
        """
        Decode a single state index back to observation space.

        Args:
            state_index: Integer index of the state

        Returns:
            Decoded observation
        """
        self.eval()
        with torch.no_grad():
            # Convert index to quantized values
            indices_tensor = torch.tensor([state_index], device=DEVICE)
            z_quantized = self.fsq.dequantize(indices_tensor)
            return self.decode(z_quantized).detach().cpu()

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
        """Training loop for FSQ VAE."""

        assert not (verbose and progress_bar)

        if progress_bar:
            iterator = trange(n_epochs, desc="FSQ VAE Epochs")
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

    @staticmethod
    def _resolve_fsq_vae_config(
        vae_config_path: str, env_config_path: str
    ) -> tuple[StateFSQVAEConfig, Dict[str, Any]]:
        """
        Resolve FSQ VAE and environment config files to config objects.

        Args:
            vae_config_path: Absolute path to FSQ VAE config file
            env_config_path: Absolute path to environment/task config file

        Returns:
            tuple: (StateFSQVAEConfig, env_kwargs dict)
        """
        vae_path = Path(vae_config_path)
        env_path = Path(env_config_path)

        # Check files exist
        if not vae_path.exists():
            raise FileNotFoundError(f"FSQ VAE config not found: {vae_config_path}")
        if not env_path.exists():
            raise FileNotFoundError(f"Environment config not found: {env_config_path}")

        # Load environment config
        with open(env_path, "r") as f:
            env_yaml = yaml.safe_load(f)
        env_kwargs = env_yaml.get("env_kwargs", {})

        # Load FSQ VAE config
        with open(vae_path, "r") as f:
            vae_yaml = yaml.safe_load(f)

        # Convert to Pydantic config
        config = StateFSQVAEConfig.from_yaml_dict(vae_yaml)
        return config, env_kwargs

    @classmethod
    def load_from_config(
        cls, vae_config_path: str, env_config_path: str
    ) -> "StateFSQVAE":
        """
        Load a StateFSQVAE model from YAML configuration files.

        Args:
            vae_config_path: Absolute path to FSQ VAE config file
            env_config_path: Absolute path to environment/task config file

        Returns:
            StateFSQVAE: Initialized model

        Examples:
            model = load_fsq_vae_from_yaml(
                "/path/to/configs/vae/fsq_cnn.yaml",
                "/path/to/configs/tasks/open_env_config.yaml"
            )
        """
        config, env_kwargs = cls._resolve_fsq_vae_config(
            vae_config_path, env_config_path
        )
        return cls(config, env_kwargs)
