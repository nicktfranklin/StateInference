from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
from torch import FloatTensor

from .utils import ObsType, StateType


def normalize(
    x: Union[np.ndarray, FloatTensor], min_val: int = 0, max_val: int = 1
) -> Union[np.ndarray, FloatTensor]:
    return (x - x.min()) / (x.max() - x.min()) * (max_val - min_val) + min_val


# The state-location is smoothed by convoling an RBF kernel with the
# one-hot representation of the location in the nXn Grid
class RbfKernelEmbedding:
    def __init__(self, kernel_size: int = 51, len_scale: float = 0.15) -> None:
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.kernel = np.empty([kernel_size, kernel_size], dtype=float)
        center = np.ones([1, 1], dtype=float) * kernel_size // 2

        def _rbf(x: np.ndarray) -> float:
            dist = ((x - center) ** 2).sum() ** 0.5
            return np.exp(-len_scale * dist)

        for r in range(kernel_size):
            for c in range(kernel_size):
                self.kernel[r, c] = _rbf(np.array([r, c]))

    def __call__(self, input_space: np.ndarray) -> np.ndarray:
        assert len(input_space.shape) == 2
        return fftconvolve(input_space, self.kernel, mode="same")


class ObservationModel:
    def __init__(
        self,
        h: int = 20,
        w: int = 20,
        map_height: int = 64,
        rbf_kernel_size: int = 51,
        rbf_kernel_scale: float = 0.15,
        location_noise_scale=1.0,
        noise_log_mean: float = -2,
        noise_log_scale: float = 0.05,
        noise_corruption_prob: float = 0.01,
        discrete_range: Optional[tuple[int, int]] = None,
    ) -> None:
        # evenly space the observations
        multiplier = map_height / (h + 1)
        self.h = h
        self.w = w
        self.map_height = map_height

        x = [ii for ii in range(h) for _ in range(h)]
        y = [ii for _ in range(w) for ii in range(w)]
        self.coordinates = np.array([x, y]).T * multiplier + multiplier
        self.states = {ii: c for ii, c in enumerate(self.coordinates)}

        self.kernel = RbfKernelEmbedding(rbf_kernel_size, rbf_kernel_scale)

        self.loc_noise_scale = location_noise_scale
        self.noise_log_mean = noise_log_mean
        self.noise_log_scale = noise_log_scale
        self.noise_corruption_prob = noise_corruption_prob
        self.discrete_range = (
            discrete_range if discrete_range is not None else tuple([0, 255])
        )

    def get_obs_coords(self, s: StateType) -> tuple[int, int]:
        x, y = self.states[s]
        x, y = int(round(x)), int(round(y))  # assign to nearest gridcell
        return x, y

    def get_grid_coords(self, s: StateType) -> tuple[int, int]:
        r = s // self.w
        c = s % self.w
        return r, c

    def get_grid_location(self, s: StateType) -> ObsType:
        r, c = self.get_grid_coords(s)
        raw_state = np.zeros((self.h, self.w))
        raw_state[r, c] = 1
        return raw_state

    def _embed_one_hot(self, x: int, y: int) -> np.ndarray:
        grid = np.zeros((self.map_height, self.map_height))
        x, y = int(round(x)), int(round(y))  # assign to nearest gridcell
        grid[x, y] = 1.0
        return grid

    def _make_embedding_from_grid(self, grid: np.ndarray) -> np.ndarray:
        embedding = self.kernel(grid)

        # convert from float to 8bit
        embedding = normalize(embedding)
        return (embedding * 255).astype(int)

    def embed_state(self, s: StateType) -> np.ndarray:
        x, y = self.get_obs_coords(s)
        grid = self._embed_one_hot(x, y)
        return self._make_embedding_from_grid(grid)

    def decode_obs(self, obs: np.ndarray) -> StateType:
        """decodes the state based on the maximum, marginally, of each dimension"""

        x_hat, y_hat = np.argmax(np.mean(obs, axis=1)), np.argmax(np.mean(obs, axis=0))
        # get the closest state
        f = lambda xypair: np.sqrt((xypair[0] - x_hat) ** 2 + (xypair[1] - y_hat) ** 2)
        d = {s: f(self.get_obs_coords(s)) for s in self.states}
        return min(d, key=d.get)

    def _location_corruption(self, x: int, y: int) -> tuple[int, int]:
        x += int(round(np.random.normal(loc=0, scale=self.loc_noise_scale)))
        y += int(round(np.random.normal(loc=0, scale=self.loc_noise_scale)))

        # truncate
        if x < 0:
            x = 0
        if x >= self.map_height:
            x = self.map_height - 1
        if y < 0:
            y = 0
        if y >= self.map_height:
            y = self.map_height - 1

        return x, y

    def _random_embedding_noise(self) -> np.ndarray:
        corrupted_mask = np.exp(
            np.random.randn(self.map_height, self.map_height) * self.noise_log_scale
            + self.noise_log_mean
        )
        corrupted_mask *= (
            np.random.uniform(size=(self.map_height, self.map_height))
            < self.noise_corruption_prob
        )
        return corrupted_mask

    def embed_state_corrupted(self, s: StateType) -> np.ndarray:
        x, y = self.get_obs_coords(s)
        x, y = self._location_corruption(x, y)
        grid = self._embed_one_hot(x, y)
        grid += self._random_embedding_noise()

        return self._make_embedding_from_grid(grid)

    def _discretize_observation(self, obs: np.ndarray):
        # normalize to the discrete range
        obs = (obs - obs.min()) / (obs.max() - obs.min()) * (
            self.discrete_range[1] - self.discrete_range[0]
        ) + self.discrete_range[0]

        return obs.astype(int)

    def embed_state_discrete(self, s: StateType) -> np.ndarray:
        return self._discretize_observation(self.embed_state_corrupted(s))

    def __call__(self, s: Union[int, List[int]]) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(s, list):
            return [self.embed_state_corrupted(s0) for s0 in s]
        return self.embed_state_corrupted(s)

    def display_state(self, s: StateType) -> None:
        # x, y = self.get_obs_coords(s)
        raw_state = self.get_grid_location(s)
        _, axes = plt.subplots(1, 3, figsize=(10, 20))
        axes[0].imshow(1 - raw_state, cmap="gray")

        def plt_lines(h, w, ax):
            y_lim = ax.get_ylim()
            x_lim = ax.get_xlim()
            for h0 in range(h):
                ax.plot([h0 + 0.5, h0 + 0.5], [-0.5, w + 0.5], c="gray")
            for w0 in range(w):
                ax.plot([-0.5, h + 0.5], [w0 + 0.5, w0 + 0.5], c="gray")
                ax.set_ylim(y_lim)
                ax.set_xlim(x_lim)

        plt_lines(self.h, self.w, axes[0])
        axes[1].imshow(self.embed_state(s))
        axes[2].imshow(self.embed_state_corrupted(s))
        axes[0].set_title("Grid-States")
        axes[1].set_title("Noise-free Observations")
        axes[2].set_title("Noisy Observation")
        plt.show()
