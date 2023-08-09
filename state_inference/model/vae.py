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
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
        )
        # run a random tensor to get the shape of the output dim
        x = torch.rand(1, input_channels, height, width)
        with torch.no_grad():
            print(self.cnn(x).shape)
            x = torch.flatten(self.cnn(x))

        # Flatten and pass through linear layer
        self.linear = nn.Sequential(
            Flatten(x.shape[0]),
            nn.Linear(x.shape[0], output_size),
        )

        # self.linear = nn.Linear(output_size, output_size)

    def forward(self, x):
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
        embedding_dims: int,
        n_actions: int,
        gru_kwargs: Optional[Dict[str, Any]] = None,
        batch_first: bool = True,
        recurrent_dropout: float = 0.2,
    ):
        super().__init__()
        gru_kwargs = gru_kwargs if gru_kwargs is not None else dict()
        self.gru_cell = nn.GRUCell(embedding_dims, embedding_dims, **gru_kwargs)
        self.hidden_encoder = nn.Linear(embedding_dims + n_actions, embedding_dims)
        self.batch_first = batch_first
        self.hidden_size = embedding_dims
        self.n_actions = n_actions
        self.nin = input_size
        self.recurrent_dropout = recurrent_dropout

    def rnn(self, x: Tensor, h: Tensor) -> Tensor:
        return self.gru_cell(x, h) + x

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

    def update_hidden_state(self, x: Tensor, h: Tensor, a: Tensor):
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
        decoder: ModelBase,
        z_dim: int,
        z_layers: int = 2,
        beta: float = 1,
        tau: float = 1,
        gamma: float = 1,
    ):
        super().__init__(encoder, decoder, z_dim, z_layers, beta, tau, gamma)
        self.rnn = rnn

    def encode_from_sequence(self, obs: Tensor, actions: Tensor) -> Tensor:
        x = self.encoder(obs)
        logits = self.rnn(x, actions).view(-1, self.z_layers, self.z_dim)
        z = self.reparameterize(logits)
        return logits, z

    def encode_from_state(self, obs: Tensor, h: Tensor) -> Tensor:
        x = self.encoder(obs)
        logits = self.rnn.rnn(x, h).view(-1, self.z_layers, self.z_dim)
        z = self.reparameterize(logits)
        return logits, z

    def forward(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.encode(obs)
        time.sleep(0.2)
        logits, z = self.encode_from_sequence(x, actions)
        return (logits, z), self.decode(z).view(
            obs[:, -1, ...].shape
        )  # preserve original shape

    def update_hidden_state(self, obs: Tensor, h: Tensor, action: Tensor) -> Tensor:
        self.eval()
        with torch.no_grad():
            x = self.encoder(obs.to(DEVICE))
            h = h.to(DEVICE)
            action = action.to(DEVICE)
            return self.rnn.update_hidden_state(x, h, action)

    def get_state(self, obs: Tensor, hidden_state: Optional[Tensor] = None):
        r"""
        Takes in observations and returns discrete states

        Args:
            obs (Tensor): a NxCxHxW tensor
            hidden_state (Tensor, optional) a NxD tensor of hidden states.  If
                no value is specified, will use a default value of zero
        """

        h_dim = self.z_dim * self.z_layers

        self.eval()
        with torch.no_grad():
            assert obs.view(-1).shape[0] % self.encoder.nin == 0

            # check the dimensions, expand if unbatched
            if obs.view(-1).shape[0] == self.encoder.nin:
                obs = obs[None, ...]
                if hidden_state is not None:
                    hidden_state = hidden_state[None, ...]

            if hidden_state is None:
                hidden_state = torch.zeros(obs.shape[0], h_dim)

            state_vars = []
            for o, h in zip(obs, hidden_state):
                o = o.view(-1)[None, ...].to(DEVICE)
                h = h[None, ...].to(DEVICE)
                assert h.ndim == 2
                assert h.shape[1] == h_dim
                _, z = self.encode_from_state(o, h)
                state_vars.append(torch.argmax(z, dim=-1).detach().cpu().view(-1))

        return torch.stack(state_vars).numpy()

    def loss(self, batch_data: List[Tensor]) -> Tensor:
        (obs, actions), _ = batch_data
        obs = obs.to(DEVICE).float()
        actions = actions.to(DEVICE).float()

        logits, z = self.encode_from_sequence(obs, actions)

        # flatten the embedding for the input to decoder
        z = z.view(-1, self.z_layers * self.z_dim).float()

        # get the two components of the ELBO loss
        kl_loss = self.kl_loss(logits)
        recon_loss = self.decoder.loss(z, obs[:, -1, ...])

        return recon_loss + kl_loss * self.beta
