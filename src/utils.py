"""The utility package.

This package exposes utility functions to set random seed, clean memeory, and so on. 
"""

import gc
import random
from datetime import datetime, timezone

import numpy as np
import torch


def set_random_seed_to(seed: int = 0):
    """Manually set random seeds in Python standard library, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def clear_memory():
    """Clear unused CPU or GPU memory."""
    # TODO: add apple silicon support if necessary
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_tensor_device(apple_silicon: bool = True) -> torch.device:
    """A function to detect and return
    the most efficient device available on the current machine.
    CUDA is the most preferred device.
    If Apple Silicon is available, MPS would be selected.
    """

    if torch.cuda.is_available():
        return torch.device('cuda')

    if apple_silicon:
        try:
            if torch.backends.mps.is_available():
                return torch.device('mps')
        except AttributeError:
            ...

    return torch.device('cpu')


def current_utc_time() -> str:
    """Return current time in UTC timezone as a string."""
    dtn = datetime.now(timezone.utc)
    return '-'.join(list(map(str, [
        dtn.year, dtn.month, dtn.day, dtn.hour, dtn.minute, dtn.second
    ])))
