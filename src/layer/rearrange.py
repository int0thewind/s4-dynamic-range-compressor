import torch.nn as nn
from einops import rearrange
from torch import Tensor


class Rearrange(nn.Module):
    def __init__(self, r: str, **kwargs: int) -> None:
        # TODO: should I save arguments as
        super().__init__()
        self.r = r
        self.kwargs = kwargs

    def forward(self, x: Tensor) -> Tensor:
        return rearrange(x, self.r, **self.kwargs)
