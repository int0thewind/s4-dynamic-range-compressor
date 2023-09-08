import torch
import torch.nn as nn
from torch import Tensor


def convert_to_decibel(t: Tensor, minimum_decibel_value: float = -80.0):
    """The functional version of a decibel converter."""
    minimum_amplitude = 10 ** (minimum_decibel_value / 20)
    return 20 * torch.log10(torch.clamp(t.abs(), minimum_amplitude))


class Decibel(nn.Module):
    """Converts amplitudes to decibels.

    The layer constructor accepts a minimum decibel value. Defaults to -80.0.
    Any decibel values below this value will be clamped to this value.
    """
    minimum_decibel_value: float

    def __init__(self, minimum_decibel_value: float = -80.0) -> None:
        super().__init__()
        self.minimum_decibel_value = minimum_decibel_value

    def forward(self, x: Tensor):
        return convert_to_decibel(x, self.minimum_decibel_value)
