import torch.nn as nn
from einops import rearrange
from torch import Tensor


class Rearrange(nn.Module):
    def __init__(self, ops: str, **kwargs: int) -> None:
        """Einops rearrange operation layer.

        This layer wraps the `einops.rearrange` method as an neural network layer
        to streamline tensor reshape operations.

        See `einops.rearrange` method for help.
        """
        super().__init__()
        self.ops = ops
        self.kwargs = kwargs

    def forward(self, x: Tensor) -> Tensor:
        return rearrange(x, self.ops, **self.kwargs)
