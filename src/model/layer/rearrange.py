import torch.nn as nn
from einops import rearrange
from torch import Tensor


class Rearrange(nn.Module):
    pattern: str
    axes_lengths: dict[str, int]

    def __init__(self, pattern: str, **axes_lengths: int) -> None:
        """Einops rearrange operation layer.

        This layer wraps the `einops.rearrange` method as an neural network layer
        to streamline tensor reshape operations.
        Useful for `nn.Sequential` layers.

        See `einops.rearrange` method for help.
        """
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths

    def forward(self, x: Tensor) -> Tensor:
        return rearrange(x, self.pattern, **self.axes_lengths)
