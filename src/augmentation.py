import torch
from torch import Tensor


@torch.no_grad()
def invert_phase(*tensors: Tensor) -> None:
    for tensor in tensors:
        tensor.imag *= -1
