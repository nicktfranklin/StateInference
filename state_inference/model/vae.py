from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.categorical import Categorical

from state_inference.utils.pytorch_utils import (
    DEVICE,
    assert_correct_end_shape,
    check_shape_match,
    gumbel_softmax,
    maybe_expand_batch,
)

OPTIM_KWARGS = dict(lr=3e-4)
VAE_BETA = 1.0
VAE_TAU = 1.0
VAE_GAMMA = 1.0
INPUT_SHAPE = (40, 40, 1)


class ModelBase(nn.Module):
    def __init__(self):
        super().__init__()

    def configure_optimizers(
        self,
        optim_kwargs: Optional[Dict[str, Any]] = None,
    ):
        optim_kwargs = optim_kwargs if optim_kwargs is not None else OPTIM_KWARGS
        optimizer = torch.optim.AdamW(self.parameters(), **optim_kwargs)

        return optimizer

    def forward(self, x):
        raise NotImplementedError

    def prep_next_batch(self):
        pass


class Flatten(nn.Module):
    """
    Progressively flattens the last dimension
    """

    def __init__(self, ndim: int) -> None:
        super().__init__()
        self.ndim = ndim

    def forward(self, x):
        """progressively flatten last dim"""
        assert x.view(-1).shape[0] % self.ndim == 0
        while x.shape[-1] != self.ndim:
            x = torch.flatten(x, start_dim=-2)

        if x.ndim == 1:
            x = x.view(1, self.ndim)
        return x


class MLP(ModelBase):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.nin = input_size
        self.nout = output_size

        # define a simple MLP neural net
        self.net = []
        hidden_size = [self.nin] + hidden_sizes
        for h0, h1 in zip(hidden_size, hidden_size[1:]):
            self.net.extend(
                [
                    nn.Linear(h0, h1),
                    nn.BatchNorm1d(h1),
                    nn.Dropout(p=dropout),
                    nn.ReLU(),
                ]
            )

        # pop the last ReLU and dropout layers for the output
        self.net.append(nn.Linear(h1, output_size))

        self.net = nn.Sequential(*self.net)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.nin
        return self.net(x)


class Encoder(ModelBase):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = MLP(input_size, hidden_sizes, output_size, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.view(x.shape[0], -1))

    def encode_sequence(self, x: Tensor, batch_first: bool = True) -> Tensor:
        x = torch.flatten(x, start_dim=2)
        if batch_first:
            x = x.permute(1, 0, 2)
        x = torch.stack([self.net(xt) for xt in x])
        if batch_first:
            x = x.permute(1, 0, 2)
        return x


class CnnEncoder(ModelBase):
    def __init__(
        self,
        input_channels: int,
        output_size: int,
        height: int = 40,
        width: int = 40,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        # run a random tensor to get the shape of the output dim
        x = torch.rand(1, input_channels, height, width)
        with torch.no_grad():
            x = torch.flatten(self.cnn(x))

        # Flatten and pass through linear layer
        self.linear = nn.Sequential(
            Flatten(x.shape[0]),
            nn.Linear(x.shape[0], output_size),
        )

        # self.linear = nn.Linear(output_size, output_size)
        self.input_shape = (height, width, input_channels)

    def forward(self, x):
        assert_correct_end_shape(x, self.input_shape)
        assert x.ndim <= 4, "Conv Net only accepts 3d or 4d input"

        # permute to (N, C, H, W) or (C, H, W)
        x = x.permute(*range(x.ndim - 3), x.ndim - 1, x.ndim - 3, x.ndim - 2)

        return self.linear(self.cnn(x))


class Decoder(MLP):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float = 0.01,
    ):
        super().__init__(input_size, hidden_sizes, output_size, dropout)
        self.net.append(torch.nn.Sigmoid())

    def loss(self, x, target):
        y_hat = self(x)
        return F.mse_loss(y_hat, torch.flatten(target, start_dim=1))


class Reshape(torch.nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)


class CnnDecoder(ModelBase):
    def __init__(
        self,
        input_size: int,
        channel_out: int,
        h: int = 40,
        w: int = 40,
    ):
        # step up is 4x
        h //= 4
        w //= 4

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64 * channel_out * h * w),
            nn.BatchNorm1d(64 * channel_out * h * w),
            nn.GELU(),
            Reshape((64 * channel_out, h, w)),
            torch.nn.ConvTranspose2d(
                64 * channel_out, 32 * channel_out, 4, 2, padding=1
            ),
            nn.BatchNorm2d(32 * channel_out),
            nn.GELU(),
            torch.nn.ConvTranspose2d(32 * channel_out, channel_out, 4, 2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return torch.flatten(self.net(x))


class StateVae(ModelBase):
    def __init__(
        self,
        encoder: ModelBase,
        decoder: ModelBase,
        z_dim: int,
        z_layers: int,
        beta: float = VAE_BETA,
        tau: float = VAE_TAU,
        gamma: float = VAE_GAMMA,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
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
        self.gamma = gamma
        self.input_shape = input_shape
        self.h_dim = z_layers * z_dim

    def reparameterize(self, logits):
        # either sample the state or take the argmax
        if self.training:
            z = gumbel_softmax(logits=logits, tau=self.tau, hard=False)
        else:
            s = torch.argmax(logits, dim=-1)  # tensor of n_batch * self.z_n_layers
            z = F.one_hot(s, num_classes=self.z_dim)
        return z

    def encode(self, x):
        # reshape encoder output to (n_batch, z_layers, z_dim)
        logits = self.encoder(x).view(-1, self.z_layers, self.z_dim)
        z = self.reparameterize(logits)
        return logits, z

    def decode(self, z):
        return self.decoder(z.view(-1, self.z_layers * self.z_dim).float())

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        logits, z = self.encode(x)
        return (logits, z), self.decode(z).view(x.shape)  # preserve original shape

    def kl_loss(self, logits):
        return Categorical(logits=logits).entropy().mean()

    def loss(self, x: Tensor) -> Tensor:
        x = x.to(DEVICE).float()
        (logits, _), x_hat = self(x)

        # get the two components of the ELBO loss
        kl_loss = self.kl_loss(logits)
        recon_loss = F.mse_loss(x_hat, x)

        return recon_loss + kl_loss * self.beta

    def get_state(self, x):
        self.eval()

        with torch.no_grad():
            # check shape
            assert_correct_end_shape(x, self.input_shape)

            # expand if unbatched
            x = maybe_expand_batch(x, self.input_shape)

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
        self.tau *= self.gamma

    def prep_next_batch(self):
        self.anneal_tau()


class StateVaeLearnedTau(StateVae):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        z_dim: int,
        z_layers: int = 2,
        beta: float = 1,
        tau: float = 1,
        gamma: float = 1,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    ):
        super().__init__(
            encoder, decoder, z_dim, z_layers, beta, tau, gamma, input_shape
        )
        self.tau = torch.nn.Parameter(torch.tensor([tau]), requires_grad=True)

    def anneal_tau(self):
        pass


class DecoderWithActions(ModelBase):
    def __init__(
        self,
        embedding_size: int,
        n_actions: int,
        hidden_sizes: List[int],
        ouput_size: int,
        dropout: float = 0.01,
    ):
        super().__init__()
        self.mlp = Decoder(embedding_size, hidden_sizes, ouput_size, dropout)
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


class TransitionStateVae(StateVae):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        next_obs_decoder: DecoderWithActions,
        z_dim: int,
        z_layers: int = 2,
        beta: float = 1,
        tau: float = 1,
        gamma: float = 1,
        input_shape: Tuple[int, int, int] = (1, 40, 40),
    ):
        super().__init__(
            encoder, decoder, z_dim, z_layers, beta, tau, gamma, input_shape
        )
        self.next_obs_decoder = next_obs_decoder

    def forward(self, x: Tensor):
        raise NotImplementedError

    def loss(self, batch_data: List[Tensor]) -> Tensor:
        obs, actions, obsp = batch_data
        obs = obs.to(DEVICE).float()
        actions = actions.to(DEVICE).float()
        obsp = obsp.to(DEVICE).float()

        logits, z = self.encode(obs)
        z = z.view(-1, self.z_layers * self.z_dim).float()

        # get the two components of the ELBO loss
        kl_loss = self.kl_loss(logits)
        recon_loss = self.decoder.loss(z, obs)
        next_obs_loss = self.next_obs_decoder.loss(z, actions, obsp)

        return recon_loss + next_obs_loss + kl_loss * self.beta


class GruEncoder(ModelBase):
    def __init__(
        self,
        input_size: int,
        embedding_dims: int,
        gru_kwargs: Optional[Dict[str, Any]] = None,
        batch_first: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        gru_kwargs = gru_kwargs if gru_kwargs is not None else dict()
        gru_kwargs["batch_first"] = batch_first
        gru_kwargs["dropout"] = dropout
        self.gru = nn.GRU(embedding_dims, embedding_dims, **gru_kwargs)
        self.batch_first = batch_first
        self.hidden_size = embedding_dims
        self.nin = input_size
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3
        x = torch.flatten(x, start_dim=2)
        assert x.shape[-1] == self.nin

        if not self.batch_first:
            x = torch.permute(x, (1, 0, 2))

        _, h_n = self.gru(x)

        return h_n.squeeze(0)

    def single_update(self, x: Tensor, h: Tensor) -> Tensor:
        """
        Updates the hidden state for a single time-point

        Args:
            -x (NxD) Tensor
            -h (NxH)Tensor
        """
        assert x.ndim == 2
        assert h.ndim == 2
        assert x.shape[1] == self.nin
        assert h.shape[1] == self.hidden_size

        # add time-dimension to input
        x = x[None, ...]
        if self.batch_first:
            x = x.permute(1, 0, 2)

        # reshape to (D*N*H)
        h = h.unsqueeze(1)

        _, h_n = self.gru(x, h)

        return h_n.squeeze(0)


class GruActionEncoder(GruEncoder):
    def __init__(
        self,
        input_size: int,
        embedding_dims: int,
        n_actions: int,
        gru_kwargs: Optional[Dict[str, Any]] = None,
        batch_first: bool = True,
        dropout: float = 0.2,
    ):
        # super().__init__()
        super().__init__(input_size, embedding_dims, gru_kwargs, batch_first, dropout)
        self.gru = nn.GRUCell(embedding_dims, embedding_dims, **gru_kwargs)
        self.hidden_encoder = nn.Linear(embedding_dims + n_actions, embedding_dims)
        self.n_actions = n_actions

    def rnn(self, x: Tensor, h: Tensor) -> Tensor:
        return self.gru(x, h)

    def joint_embed(self, h, a):
        return self.hidden_encoder(torch.cat([h, a], dim=1))

    def forward(
        self,
        embedding: Tensor,
        actions: Tensor,
    ):
        """
        Args:
            obs (Tensor): a NxHxWxC tensor of observations
            action (Tensor): a NxNa vector of actions.  Note, these
                correspond to the action take at the previous time step, prior to
                the observation (obs)
        """
        embedding = torch.flatten(embedding, start_dim=2)
        if self.batch_first:
            embedding = torch.permute(embedding, (1, 0, 2))
            actions = torch.permute(actions, (1, 0, 2))

        n_batch = embedding.shape[1]

        # initialize the hidden state and action
        h = torch.zeros(n_batch, self.hidden_size).to(DEVICE)

        # loop through the sequence of observations
        for ii, (x, a) in enumerate(zip(embedding, actions)):
            # encode the hidden state with the previous actions
            h = self.joint_embed(h, a)

            # pass the observation through the rnn (+ encoder)
            h = self.rnn(x, h)

            # apply dropout except for the last time-step
            if ii < embedding.shape[0] - 1:
                h = F.dropout(h, p=self.recurrent_dropout)

        return h

    def single_update(self, x: Tensor, h: Tensor, a: Tensor):
        """
        only accepts a single observation and a single action
        Args:
            obs (Tensor): a CxHxW tensor
            hidden_state (Tensor) a D tensor of hidden states.  If
                no value is specified, will use a default value of zero
            action (Tensor): a single action value
        """
        x = x.view(1, -1)
        a = F.one_hot(a, num_classes=self.n_actions).float()
        h = h.view(1, -1)

        # use the rnn to take the previous hidden state and current observation
        # to make a new hidden state
        h = self.rnn(x, h)

        # update the hidden state with the action
        h = self.joint_embed(h, a)

        return h.view(-1)


class RecurrentVae(StateVae):
    def __init__(
        self,
        encoder: Encoder,
        rnn: GruEncoder,
        decoder: Decoder,
        z_dim: int,
        z_layers: int = 2,
        beta: float = 1,
        tau: float = 1,
        gamma: float = 1,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    ):
        super().__init__(
            encoder, decoder, z_dim, z_layers, beta, tau, gamma, input_shape
        )
        self.rnn = rnn

    def encode_from_sequence(self, obs: Tensor) -> Tensor:
        # use a loop to encode the sequences
        assert_correct_end_shape(obs, self.input_shape)
        logits = self.encoder.encode_sequence(obs)

        # only use rnn additively on the embeddings (and only the
        # last time step is used in the output)
        logits = logits[:, -1, :] + self.rnn(logits)
        z = self.reparameterize(logits.view(-1, self.z_layers, self.z_dim))
        return logits, z

    def _encode_single_obs_from_state(self, obs: Tensor, h: Tensor) -> Tensor:
        assert check_shape_match(obs, self.input_shape)
        assert check_shape_match(h, self.h_dim)

        logits = self.encoder(obs)
        logits = logits + self.rnn.single_update(logits, h)  # only use rnn additively
        z = self.reparameterize(logits.view(-1, self.z_layers, self.z_dim))
        return logits, z

    def encode_from_state(self, obs: Tensor, h: Tensor) -> Tensor:
        if check_shape_match(obs, self.input_shape):
            return self._encode_single_obs_from_state(obs, h)

        logits, z = zip(
            *[self._encode_single_obs_from_state(o0, h0) for o0, h0 in zip(obs, h)]
        )
        return torch.stack(logits), torch.stack(z)

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
         -obs (NxLxHxWxC) tensor
        """
        assert_correct_end_shape(obs, self.input_shape)
        assert obs.ndim == 5

        logits, z = self.encode_from_sequence(obs)
        return (logits, z), self.decode(z).view(
            obs[:, -1, ...].shape
        )  # preserve original shape

    def update_hidden_state(self, obs: Tensor, h: Tensor) -> Tensor:
        self.eval()

        obs = maybe_expand_batch(obs, self.input_shape)
        h = maybe_expand_batch(h, (self.h_dim))

        with torch.no_grad():
            # will always be a single example
            obs = obs.view(-1, *self.input_shape)
            x = self.encoder(obs.to(DEVICE))
            h = h.to(DEVICE)
            return self.rnn.single_update(x, h).squeeze()

    def get_state(self, obs: Tensor, hidden_state: Tensor):
        r"""
        Takes in observations and returns discrete states. Does not accept sequence data

        Args:
            obs (Tensor): a NxHxWxC tensor
            hidden_state (Tensor, optional) a NxD tensor of hidden states.  If
                no value is specified, will use a default value of zero
        """

        self.eval()
        with torch.no_grad():
            assert_correct_end_shape(obs, self.input_shape)

            # check the dimensions, expand if unbatched
            obs = maybe_expand_batch(obs, self.input_shape)
            hidden_state = maybe_expand_batch(hidden_state, (self.h_dim))

            _, z = self.encode_from_state(obs.to(DEVICE), hidden_state.to(DEVICE))

            z = torch.argmax(z, dim=-1).detach().cpu().view(-1, self.z_layers).numpy()
        return z

    def loss(self, batch_data: List[Tensor]) -> Tensor:
        self.train()
        (obs, _), _ = batch_data
        obs = obs.to(DEVICE).float()

        logits, z = self.encode_from_sequence(obs)

        # flatten the embedding for the input to decoder
        z = z.view(-1, self.z_layers * self.z_dim).float()
        # raise
        # get the two components of the ELBO loss
        kl_loss = self.kl_loss(logits)
        recon_loss = self.decoder.loss(z, obs[:, -1, ...])

        return recon_loss + kl_loss * self.beta
