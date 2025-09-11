from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator


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

    @validator("tau_annealing_rate")
    def validate_tau_annealing_rate(cls, v, values):
        if values.get("tau_is_parameter", False) and v != 1.0:
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
