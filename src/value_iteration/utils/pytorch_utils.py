from typing import List, Tuple, Union

import numpy as np
import torch
from torch import FloatTensor, LongTensor

QUIET = False

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")


# Note: there is an issue F.gumbel_softmax that appears to causes an error w
# where a valide distribution will return nans, preventing training.  Fix from
# https://gist.github.com/GongXinyuu/3536da55639bd9bfdd5a905ebf3ab88e
def make_tensor(x: np.ndarray) -> FloatTensor:
    return torch.tensor(x, dtype=float)


def normalize(
    x: Union[np.ndarray, FloatTensor], min_val: int = 0, max_val: int = 1
) -> Union[np.ndarray, FloatTensor]:
    return (x - x.min()) / (x.max() - x.min()) * (max_val - min_val) + min_val


def convert_float_to_8bit(x: Union[List[FloatTensor], FloatTensor]) -> LongTensor:
    if isinstance(x, List):
        return [convert_float_to_8bit(x0) for x0 in x]
    assert x.max() <= 1.0
    assert x.min() >= 0.0

    return (x * 255).type(torch.int)


def convert_8bit_to_float(
    x: Union[List[Union[LongTensor, np.ndarray]], Union[LongTensor, np.ndarray]],
) -> FloatTensor:
    if isinstance(x, List):
        return [convert_8bit_to_float(x0) for x0 in x]

    x = maybe_convert_to_long_tensor(x)
    assert x.max() <= 255
    assert x.min() >= 0

    return (x / 255).type(torch.float)


def convert_8bit_array_to_float_tensor(
    x: Union[List[np.ndarray], np.ndarray],
) -> FloatTensor:
    if isinstance(x, list):
        return torch.stack([convert_8bit_array_to_float_tensor(x0) for x0 in x])

    return convert_8bit_to_float(make_tensor(x))


def maybe_convert_to_long_tensor(x: Union[LongTensor, np.ndarray]) -> LongTensor:
    if isinstance(x, np.ndarray) and x.dtype == np.int_:
        return torch.tensor(x, dtype=torch.long)
    elif torch.is_tensor(x) and torch.is_floating_point(x):
        return x.to(torch.long)
    elif torch.is_tensor(x) and not torch.is_floating_point(x):
        return x
    raise ValueError(f"Cannot convert {x} to LongTensor")


def maybe_convert_to_tensor(x: Union[FloatTensor, np.ndarray]) -> FloatTensor:
    return x if isinstance(x, FloatTensor) else torch.tensor(x)


def check_shape_match(x: FloatTensor, shape: Tuple[int]):
    return (x.ndim == len(shape)) and (
        x.view(-1).shape[0] == torch.prod(torch.tensor(shape))
    )


def assert_correct_end_shape(
    x: torch.tensor, shape: Tuple[int, int] | Tuple[int, int, int]
) -> bool:
    if len(shape) == 2:
        assert (
            x.shape[-2:] == shape
        ), f"Tensor Shape {x.shape} does not match target {shape}"
    elif len(shape) == 3:
        assert (
            x.shape[-3:] == shape
        ), f"Tensor Shape {x.shape} does not match target {shape}"
    else:
        raise Exception(f"Shape {shape} is unsupported")


def maybe_expand_batch(
    x: torch.tensor, target_shape: Tuple[int, int] | Tuple[int, int, int]
) -> torch.tensor:
    # expand if unbatched
    if check_shape_match(x, target_shape):
        return x[None, ...]
    return x
