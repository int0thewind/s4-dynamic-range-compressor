import torch
from torch import Tensor


@torch.no_grad()
def invert_phase(*tensors: Tensor):
    return tuple(-t for t in tensors)
