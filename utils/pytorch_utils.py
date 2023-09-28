from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

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
def gumbel_softmax(
    logits: Tensor,
    tau: float = 1,
    hard: bool = False,
    dim: int = -1,
) -> Tensor:
    r"""
    Samples from the `Gumbel-Softmax distribution`_ and optionally discretizes.
    You can use this function to replace "F.gumbel_softmax".
    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.
    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`
      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)
    .. _Gumbel-Softmax distribution:
        https://arxiv.org/abs/1611.00712
        https://arxiv.org/abs/1611.01144
    """

    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    clip_grad: bool = None,
    preprocess: Optional[callable] = None,
) -> List[Tensor]:
    model.train()

    train_losses = []
    for x in train_loader:
        if preprocess:
            x = preprocess(x)

        optimizer.zero_grad()
        loss = model.loss(x)
        loss.backward()

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()
        train_losses.append(loss.item())

    return train_losses


def eval_loss(
    model: nn.Module,
    data_loader: DataLoader,
    preprocess: Optional[callable] = None,
) -> Tensor:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in data_loader:
            if preprocess:
                x = preprocess(x)
            loss = model.loss(x)
            total_loss += loss * data_loader.batch_size
        avg_loss = total_loss / len(data_loader.dataset)

    return avg_loss.item()


def train_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    train_args: Dict[str, Any],
    preprocess: Optional[callable] = None,
    scheduler: Optional[str] = None,
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
):
    epochs, lr = train_args["epochs"], train_args["lr"]
    lr_decay = train_args.get("lr_decay", None)
    grad_clip = train_args.get("grad_clip", None)
    if hasattr(model, "configure_optimizers"):
        optimizer = model.configure_optimizers(dict(lr=lr))
    else:
        assert lr_decay is None, "LR Decay only supported for Model Base"
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    if scheduler is not None:
        assert scheduler_kwargs is not None, "must specify kwargs for scheduler!"
        SchedularClass = getattr(torch.optim.lr_scheduler, scheduler)
        scheduler = SchedularClass(optimizer, **scheduler_kwargs)
        assert hasattr(scheduler, "step")

    train_losses = []
    test_losses = [eval_loss(model, test_loader, preprocess=preprocess)]
    for epoch in range(epochs):
        model.train()
        train_losses.extend(
            train(model, train_loader, optimizer, grad_clip, preprocess=preprocess)
        )
        test_loss = eval_loss(model, test_loader, preprocess=preprocess)
        test_losses.append(test_loss)

        model.prep_next_batch()

        if not QUIET:
            print(f"Epoch {epoch}, ELBO Loss (test) {test_loss:.6f}")

        if scheduler is not None:
            scheduler.step()

    return train_losses, test_losses


def make_tensor(x: np.ndarray) -> Tensor:
    return torch.tensor(x, dtype=float)


def normalize(
    x: Union[np.ndarray, Tensor], min_val: int = 0, max_val: int = 1
) -> Union[np.ndarray, Tensor]:
    return (x - x.min()) / (x.max() - x.min()) * (max_val - min_val) + min_val


def convert_float_to_8bit(x: Union[List[Tensor], Tensor]) -> torch.IntTensor:
    if isinstance(x, List):
        return [convert_float_to_8bit(x0) for x0 in x]
    assert x.max() <= 1.0
    assert x.min() >= 0.0

    return (x * 255).type(torch.int)


def convert_8bit_to_float(x: Union[List[Tensor], Tensor]) -> Tensor:
    if isinstance(x, List):
        return [convert_8bit_to_float(x0) for x0 in x]

    assert isinstance(x, Tensor)
    assert x.max() <= 255
    assert x.min() >= 0

    return (x / 255).type(torch.float)


def convert_8bit_array_to_float_tensor(
    x: Union[List[np.ndarray], np.ndarray]
) -> Tensor:
    if isinstance(x, list):
        return torch.stack([convert_8bit_array_to_float_tensor(x0) for x0 in x])

    return convert_8bit_to_float(make_tensor(x))


def maybe_convert_to_tensor(x: Union[Tensor, np.ndarray]) -> Tensor:
    return x if isinstance(x, Tensor) else torch.tensor(x)


def check_shape_match(x: Tensor, shape: Tuple[int]):
    return x.view(-1).shape[0] == torch.prod(torch.tensor(shape))


def assert_correct_end_shape(
    x: Tensor, shape: Tuple[int, int] | Tuple[int, int, int]
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
    x: Tensor, target_shape: Tuple[int, int] | Tuple[int, int, int]
) -> Tensor:
    # expand if unbatched
    if check_shape_match(x, target_shape):
        return x[None, ...]
    return x


def split_into_batches(x: Tensor, batch_size: int, dim: int = 0) -> List[Tensor]:
    """
    Function that takes in an tensor and splits it into batches of size
    b along specified dimension (default dim=0).  If the tensor isn't evenly
    divisable, the last batch will be smaller than batch_size.

    Returns a list of tensors
    """
    n = x.shape[dim]
    if n <= batch_size:
        return x

    full_batches = n // batch_size
    last_batch = n % batch_size

    if last_batch > 0:
        last_tensor = tuple([x[-last_batch:]])
    else:
        last_tensor = tuple([])

    # split the batches
    tensors = x[:-last_batch].reshape(full_batches, batch_size, -1).split(1, dim=0)

    # drop the extra dim
    tensors = tuple([x0.squeeze(0) for x0 in tensors])

    return tensors + last_tensor
