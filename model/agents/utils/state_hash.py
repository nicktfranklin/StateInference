import numpy as np


class StateHash:
    def __init__(self, z_dim: int, z_layers: int) -> None:
        self.hash_vector = np.array([z_dim**ii for ii in range(z_layers)])
        self.z_dim = z_dim
        self.z_layers = z_layers
        self.inverse = dict()

    def _maybe_add_to_hash(self, v: int | np.ndarray, z: np.ndarray) -> None:
        if isinstance(v, np.ndarray):
            for v0 in v:
                if v0 not in self.inverse:
                    self.inverse[v0] = z
        elif v not in self.inverse:
            self.inverse[v] = z

    def __call__(self, z: np.ndarray) -> np.array:
        v = z.dot(self.hash_vector)
        self._maybe_add_to_hash(v, z)
        return v

    def get_inverse(self, v: int) -> np.ndarray:
        return self.inverse[v]

    def reset(self):
        self.inverse = dict()
