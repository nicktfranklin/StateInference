from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from pydantic import BaseModel, Field, field_validator
from torch import FloatTensor, Tensor, nn
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import trange

from value_iteration.utils.pytorch_utils import (
    DEVICE,
    assert_correct_end_shape,
    check_shape_match,
)

from .constants import OPTIM_KWARGS
from .gumbel_softmax import gumbel_softmax

# Import encoder/decoder classes
from .nets import (
    AutoregressiveDeconvNet,
    CnnDecoder,
    CnnEncoder,
    MlpDecoder,
    MlpEncoder,
)

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


class EncoderConfig(BaseModel):
    """Configuration for encoder networks."""

    hidden_sizes: Optional[List[int]] = None
    in_channels: Optional[int] = None
    channels: Optional[List[int]] = None
    kernel_sizes: Optional[List[int]] = None
    strides: Optional[List[int]] = None
    embedding_dim: Optional[int] = None
    input_shape: Optional[Tuple[int, int, int]] = None


class DecoderConfig(BaseModel):
    """Configuration for decoder networks."""

    hidden_sizes: Optional[List[int]] = None
    output_channels: Optional[int] = None
    channel_out: Optional[int] = None
    channels: Optional[List[int]] = None
    embedding_dim: Optional[int] = None
    output_shape: Optional[Tuple[int, int, int]] = None


class VAEConfig(BaseModel):
    """Configuration for VAE model parameters."""

    z_dim: int = Field(..., description="Dimension of each categorical variable")
    z_layers: int = Field(..., description="Number of categorical variables")
    beta: float = Field(default=1.0, description="Beta parameter for KL loss weighting")
    tau: float = Field(
        default=0.5, description="Temperature parameter for Gumbel softmax"
    )
    tau_annealing_rate: float = Field(
        default=1.0, description="Rate at which tau is annealed during training"
    )
    tau_is_parameter: bool = Field(
        default=False, description="Whether tau is a learnable parameter"
    )
    tau_min: float = Field(
        default=1e-4, description="Minimum value for tau during annealing"
    )
    input_shape: Optional[Tuple[int, int, int]] = None
    run_unit_test: bool = Field(
        default=False, description="Run unit test during initialization"
    )

    @field_validator("tau_annealing_rate")
    @classmethod
    def validate_tau_annealing_rate(cls, v, info):
        if info.data.get("tau_is_parameter", False) and v != 1.0:
            raise ValueError(
                "tau_annealing_rate must be 1.0 when tau_is_parameter is True"
            )
        return v


class StateVAEConfig(BaseModel):
    """Complete configuration for StateVAE model."""

    encoder_class: str = Field(..., description="Name of the encoder class")
    decoder_class: str = Field(..., description="Name of the decoder class")
    encoder_kwargs: EncoderConfig = Field(default_factory=EncoderConfig)
    decoder_kwargs: DecoderConfig = Field(default_factory=DecoderConfig)
    vae_kwargs: VAEConfig = Field(..., description="VAE model parameters")

    def prepare_for_instantiation(self, env_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for model instantiation with environment parameters."""

        # Calculate input shape from environment
        h = env_kwargs["map_height"]
        input_shape = (1, h, h)

        # Update VAE kwargs with input shape
        vae_kwargs = self.vae_kwargs.model_copy()
        vae_kwargs.input_shape = input_shape

        # Calculate embedding dimension
        embedding_dim = vae_kwargs.z_dim * vae_kwargs.z_layers

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
    def from_yaml_dict(cls, config_dict: Dict[str, Any]) -> "StateVAEConfig":
        """Create StateVAEConfig from YAML dictionary format."""
        return cls(
            encoder_class=config_dict["encoder_class"],
            decoder_class=config_dict["decoder_class"],
            encoder_kwargs=EncoderConfig(**config_dict.get("encoder_kwargs", {})),
            decoder_kwargs=DecoderConfig(**config_dict.get("decoder_kwargs", {})),
            vae_kwargs=VAEConfig(**config_dict["vae_kwargs"]),
        )


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
        config: StateVAEConfig,
        env_kwargs: Dict[str, Any],
    ):
        """
        Initialize StateVae from Pydantic config.

        Args:
            config: StateVAEConfig containing all model configuration
            env_kwargs: Environment-specific parameters (e.g., map_height)

        Note: larger values of beta result in more independent state values
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
        self.z_layers = vae_kwargs["z_layers"]
        self.z_dim = vae_kwargs["z_dim"]
        self.beta = vae_kwargs["beta"]
        self.tau = vae_kwargs["tau"]
        self.tau_annealing_rate = vae_kwargs["tau_annealing_rate"]
        self.input_shape = tuple(vae_kwargs["input_shape"])
        self.tau_min = vae_kwargs["tau_min"]

        tau_is_parameter = vae_kwargs["tau_is_parameter"]
        if tau_is_parameter:
            self.tau = torch.nn.Parameter(torch.tensor([self.tau]), requires_grad=True)
            assert (
                self.tau_annealing_rate == 1.0
            ), "Must set tau annealing to 1 for learnable tau"

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
        """Deprecated: Use load_from_yaml_file() or direct __init__ instead."""
        # Convert dict to Pydantic model for validation and processing
        config = StateVAEConfig.from_yaml_dict(vae_config)
        return cls(config, env_kwargs)

    @classmethod
    def make_from_pydantic_config(
        cls, config: StateVAEConfig, env_kwargs: Dict[str, Any]
    ):
        """Deprecated: Use direct __init__ instead."""
        return cls(config, env_kwargs)

    def reparameterize(self, logits):
        """
        Reparameterize the input logits to obtain the latent variable tensor.

        Args:
            logits (Tensor): The input logits tensor of shape (batch_size, z_layers, z_dim).

        Returns:
            Tensor: The latent variable tensor of shape (batch_size, z_layers, z_dim).

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

    def flatten_z(self, z):
        return z.view(-1, self.z_layers * self.z_dim)

    def decode(self, z):
        return self.decoder(self.flatten_z(z).float())

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the Variational Autoencoder (VAE) model.

        Args:
            x (Tensor): The input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing two tensors:
                - logits (Tensor): The output logits tensor of shape (batch_size, z_layers, z_dim).
                - z (Tensor): The latent variable tensor of shape (batch_size, z_layers, z_dim).

        """
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

        return kl * self.beta

    def recontruction_loss(self, x: FloatTensor, x_hat: FloatTensor) -> FloatTensor:
        mse_loss = F.mse_loss(x_hat, x, reduction="none")
        # sum loss over dimensions in each example, average over batch
        return mse_loss.view(x.shape[0], -1).sum(1).mean()

    def loss(self, x: FloatTensor, target: FloatTensor = None) -> FloatTensor:
        x = x.to(DEVICE).float()
        (logits, _), y_hat = self(x)

        if target is None:
            target = x

        # get the two components of the ELBO loss
        kl_loss = self.kl_loss(logits)
        recon_loss = self.recontruction_loss(target, y_hat)

        return recon_loss + kl_loss

    def diagnose_loss(self, x):
        with torch.no_grad():
            x = x.to(DEVICE).float()
            (logits, _), x_hat = self(x)

            # get the two components of the ELBO loss
            kl_loss = self.kl_loss(logits)
            recon_loss = self.recontruction_loss(x, x_hat)

            loss = recon_loss - kl_loss
            print(
                f"Total Loss: {loss:.4f}, Reconstruction: {recon_loss:.4f}, "
                + f"KL-Loss: {kl_loss:.4f}, beta: {self.beta}"
            )

    def get_state(self, x):
        """
        Assume input shape of NxCxHxW or CxHxW.
        """
        assert x.ndim <= 4
        assert_correct_end_shape(x, self.input_shape)

        self.eval()
        with torch.no_grad():
            _, z = self.encode(x.to(DEVICE))

            state_vars = torch.argmax(z, dim=-1).detach().cpu()
        return state_vars

    def sample_state(self, x):
        """
        Assume input shape of NxCxHxW or CxHxW.
        """
        assert x.ndim <= 4
        assert_correct_end_shape(x, self.input_shape)

        self.eval()
        with torch.no_grad():
            logits, _ = self.encode(x.to(DEVICE))

            if logits.ndim == 2:
                logits = logits.unsqueeze(0)
            B, N, K = logits.shape
            logits = logits.view(B * N, K)
            state_vars = Categorical(logits=logits).sample().detach().cpu().numpy()

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
        if self.tau_annealing_rate - 1 < self.tau_min:
            return
        self.tau *= self.tau_annealing_rate

    def prep_next_batch(self):
        self.anneal_tau()

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

        assert not (verbose and progress_bar)

        if progress_bar:
            iterator = trange(n_epochs, desc="Vae Epochs")
        else:
            iterator = range(n_epochs)

        optim = optim if optim is not None else self.configure_optimizers()

        train_losses, test_losses = [], []
        for epoch in iterator:

            # training
            self.train()
            for batch in train_dataloader:

                optim.zero_grad()
                loss = self.loss(batch)
                loss.backward()

                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

                optim.step()

                train_losses.append(loss.detach().cpu().item())

            # validation
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

            self.prep_next_batch()

            if verbose:
                print(f"Epoch {epoch}, ELBO Loss (test) {test_losses[-1]:.6f}")

        return train_losses, test_losses

    @staticmethod
    def _resolve_vae_config(
        vae_config_path: str, env_config_path: str
    ) -> tuple[StateVAEConfig, Dict[str, Any]]:
        """
        Resolve VAE and environment config files to config objects.

        Args:
            vae_config_path: Absolute path to VAE config file
            env_config_path: Absolute path to environment/task config file

        Returns:
            tuple: (StateVAEConfig, env_kwargs dict)
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

        # Convert to Pydantic config
        config = StateVAEConfig.from_yaml_dict(vae_yaml)
        return config, env_kwargs

    @classmethod
    def load_from_yaml(cls, vae_config_path: str, env_config_path: str) -> "StateVae":
        """
        Load a StateVae model from YAML configuration files.

        Args:
            vae_config_path: Absolute path to VAE config file
            env_config_path: Absolute path to environment/task config file

        Returns:
            StateVae: Initialized model

        Examples:
            model = load_vae_from_yaml(
                "/path/to/configs/vae/cnn.yaml",
                "/path/to/configs/tasks/open_env_config.yaml"
            )
        """
        config, env_kwargs = StateVae._resolve_vae_config(
            vae_config_path, env_config_path
        )
        return cls(config, env_kwargs)
