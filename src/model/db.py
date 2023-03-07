import torch
import torch.nn as nn
from torch import Tensor


class Decibel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor):
        return 20 * torch.log10(torch.clamp(torch.abs(x), 1e-4))
