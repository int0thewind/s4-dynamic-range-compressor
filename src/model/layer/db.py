import torch
import torch.nn as nn
from torch import Tensor


class Decibel(nn.Module):
    """Converts amplitudes to decibels.

    The layer constructor accepts a minimum decibel value. Defaults to -80.0.
    Any decibel values below this value will be clamped to this value.
    """
    minimum_amplitude: float

    def __init__(self, minimum_decibel_value: float = -80.0) -> None:
        super().__init__()
        self.minimum_amplitude = 10 ** (minimum_decibel_value / 20)

    def forward(self, x: Tensor):
        return 20 * torch.log10(torch.clamp(x.abs(), self.minimum_amplitude))
