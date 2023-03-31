import torch
import torch.nn as nn
from torch import Tensor


class Absolute(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.abs()
