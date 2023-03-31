"""Data augmentation module.

This module contains functions that can be used to augment data for training.

So far, there is only one possible data augmentation method: phase inversion.
"""

import torch
from torch import Tensor

__all__ = ['invert_phase']


@torch.no_grad()
def invert_phase(*tensors: Tensor):
    """Invert tensor(s) phase.

    This function can take one or more tensors as input.
    If there is only one tensor, the function will return the single tensor.
    Otherwise, a tuple of tensors will be returned.
    """
    if len(tensors) == 1:
        return -tensors[0]
    return tuple(-t for t in tensors)
