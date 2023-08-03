from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.categorical import Categorical

from state_inference.utils.pytorch_utils import DEVICE, gumbel_softmax

OPTIM_KWARGS = dict(lr=3e-4)
VAE_BETA = 1.0
VAE_TAU = 1.0
VAE_GAMMA = 1.0


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
        hidden_size = [self.nin] + hidden_sizes + [self.nout]
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
        self.net.pop()
        self.net.pop()

        self.net = nn.Sequential(*self.net)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Encoder(MLP):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.view(x.shape[0], -1))


class Decoder(MLP):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float = 0.01,
    ):
        super().__init__(input_size, hidden_sizes, output_size, dropout)
        self.net.pop(-1)
        self.net.append(torch.nn.Sigmoid())

    def loss(self, x, target):
        y_hat = self(x)
        return F.mse_loss(y_hat, torch.flatten(target, start_dim=1))


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
            # expand if unbatched
            assert x.view(-1).shape[0] % self.encoder.nin == 0
            if x.view(-1).shape[0] == self.encoder.nin:
                x = x[None, ...]

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
        encoder: ModelBase,
        decoder: ModelBase,
        z_dim: int,
        z_layers: int = 2,
        beta: float = 1,
        tau: float = 1,
        gamma: float = 1,
    ):
        super().__init__(encoder, decoder, z_dim, z_layers, beta, tau, gamma)
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
        encoder: ModelBase,
        decoder: ModelBase,
        next_obs_decoder: DecoderWithActions,
        z_dim: int,
        z_layers: int = 2,
        beta: float = 1,
        tau: float = 1,
        gamma: float = 1,
    ):
        super().__init__(encoder, decoder, z_dim, z_layers, beta, tau, gamma)
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
        hidden_sizes: List[int],
        embedding_dims: int,
        n_actions: int,
        gru_kwargs: Optional[Dict[str, Any]] = None,
        batch_first: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        gru_kwargs = gru_kwargs if gru_kwargs is not None else dict()
        self.feature_extracter = MLP(input_size, hidden_sizes, embedding_dims, dropout)
        self.gru_cell = nn.GRUCell(embedding_dims, embedding_dims, **gru_kwargs)
        self.hidden_encoder = nn.Linear(embedding_dims, embedding_dims)
        self.action_encoder = nn.Linear(n_actions, embedding_dims)
        self.batch_first = batch_first
        self.hidden_size = embedding_dims
        self.n_actions = n_actions
        self.nin = input_size

    def rnn(self, obs: Tensor, h: Tensor) -> Tensor:
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
        h = torch.zeros(n_batch, self.hidden_size).to(DEVICE)
        a_prev = torch.zeros(n_batch, self.hidden_size).to(DEVICE)

        # loop through the sequence of observations
        for o, a in zip(obs, actions):
            # encode the hidden state with the previous actions
            h = self.hidden_encoder(h + a_prev)

            # encode the action for the next step
            a_prev = self.action_encoder(a)

            # pass the observation through the rnn (+ encoder)
            h = self.rnn(o, h)

        return h


class RecurrentVae(StateVae):
    def __init__(
        self,
        encoder: GruEncoder,
        decoder: ModelBase,
        z_dim: int,
        z_layers: int = 2,
        beta: float = 1,
        tau: float = 1,
        gamma: float = 1,
    ):
        super().__init__(encoder, decoder, z_dim, z_layers, beta, tau, gamma)

    def encode(self, x):
        raise NotImplementedError

    def _encode_from_sequence(self, obs: Tensor, actions: Tensor) -> Tensor:
        logits = self.encoder(obs, actions).view(-1, self.z_layers, self.z_dim)
        z = self.reparameterize(logits)
        return logits, z

    def _encode_from_state(self, obs: Tensor, h: Tensor) -> Tensor:
        logits = self.encoder.rnn(obs, h).view(-1, self.z_layers, self.z_dim)
        z = self.reparameterize(logits)
        return logits, z

    def get_state(self, obs: Tensor, hidden_state: Optional[Tensor] = None):
        r"""
        Takes in observations and returns discrete states

        Args:
            obs (Tensor): a NxCxHxW tensor
            hidden_state (Tensor, optional) a NxD tensor of hidden states.  If
                no value is specified, will use a default value of zero
        """
        # hidden_state = (
        #     hidden_state
        #     if isinstance(hidden_state, Tensor)
        #     else torch.zeros(ob).to(DEVICE)
        # )

        h_dim = self.z_dim * self.z_layers

        self.eval()
        with torch.no_grad():
            # check the dimensions, expand if unbatch

            # expand if unbatched
            assert obs.view(-1).shape[0] % self.encoder.nin == 0
            if obs.view(-1).shape[0] == self.encoder.nin:
                obs = obs[None, ...]
                if hidden_state is not None:
                    hidden_state[None, ...]

            if hidden_state is None:
                hidden_state = torch.zeros(obs.shape[0], h_dim)

            state_vars = []
            for o, h in zip(obs, hidden_state):
                o = o.view(-1)[None, ...].to(DEVICE)
                h = h[None, ...].to(DEVICE)
                assert h.ndim == 2
                assert h.shape[1] == h_dim
                _, z = self._encode_from_state(o, h)
                state_vars.append(torch.argmax(z, dim=-1).detach().cpu().view(-1))

        return torch.stack(state_vars).numpy()

    def loss(self, batch_data: List[Tensor]) -> Tensor:
        (obs, actions), _ = batch_data
        obs = obs.to(DEVICE).float()
        actions = actions.to(DEVICE).float()

        logits, z = self._encode_from_sequence(obs, actions)

        # flatten the embedding for the input to decoder
        z = z.view(-1, self.z_layers * self.z_dim).float()

        # get the two components of the ELBO loss
        kl_loss = self.kl_loss(logits)
        recon_loss = self.decoder.loss(z, obs[:, -1, ...])

        return recon_loss + kl_loss * self.beta
