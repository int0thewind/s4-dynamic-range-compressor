from typing import Literal, get_args

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ['Activation', 'get_activation_type_from']

Activation = Literal['tanh', 'sigmoid', 'PTanh', 'GELU', 'ReLU', 'Identity', 'PReLU']


class PTanh(nn.Module):
    def __init__(self, a: float = 1.0, b: float = 1.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))

    def forward(self, x: Tensor) -> Tensor:
        return self.a * torch.tanh(self.b * x)


def get_activation_type_from(activation: Activation) -> type[nn.Module]:
    if not activation in get_args(Activation):
        raise ValueError(
            f'Unsupported non-linear activation `{activation}`.'
        )

    if activation == 'tanh':
        return nn.Tanh
    if activation == 'sigmoid':
        return nn.Sigmoid
    if activation == 'PTanh':
        return PTanh
    if activation == 'ReLU':
        return nn.ReLU
    if activation == 'GELU':
        return nn.GELU
    if activation == 'PReLU':
        return nn.PReLU
    return nn.Identity
