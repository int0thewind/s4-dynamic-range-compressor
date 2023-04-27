from typing import Literal, get_args

import torch.nn as nn

__all__ = ['Activation', 'get_activation_type_from']

Activation = Literal['tanh', 'sigmoid', 'GELU', 'ReLU', 'Identity', 'PReLU']


def get_activation_type_from(activation: Activation) -> type[nn.Module]:
    if not activation in get_args(Activation):
        raise ValueError(
            f'Unsupported non-linear activation `{activation}`.'
        )

    if activation == 'tanh':
        return nn.Tanh
    if activation == 'sigmoid':
        return nn.Sigmoid
    if activation == 'ReLU':
        return nn.ReLU
    if activation == 'GELU':
        return nn.GELU
    if activation == 'PReLU':
        return nn.PReLU
    return nn.Identity
