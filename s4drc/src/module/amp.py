import torch.nn as nn
from torch import Tensor


class Amplitude(nn.Module):
    """Converts decibels to amplitudes."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor):
        return 10 ** (x / 20)