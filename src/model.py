"""Definitions of actual training module.
"""

from typing import Literal, get_args

import torch.nn as nn
from torch import Tensor

from .layer import DSSM, Absolute, Amplitude, Decibel, Rearrange

ActivationType = Literal['tanh', 'sigmoid', 'gelu']


class DRCModel(nn.Module):
    side_chain: nn.Sequential

    def __init__(
        self, num_channel: int, s4_hidden_size: int,
        s4_learning_rate: float | None,
        model_depth: int, activation: ActivationType,
        db: bool, abs: bool, amp: bool,
    ):
        if num_channel < 1:
            raise ValueError()
        if s4_hidden_size < 1:
            raise ValueError()
        if model_depth < 0:
            raise ValueError()
        if not activation in get_args(ActivationType):
            raise ValueError(
                f'Unsupported non-linear activation `{activation}`.'
            )

        super().__init__()
        if activation == 'tanh':
            Act = nn.Tanh
        elif activation == 'sigmoid':
            Act = nn.Sigmoid
        else:
            Act = nn.GELU

        layers: list[nn.Module] = []

        if abs:
            layers.append(Absolute())
        if db:
            layers.append(Decibel())

        layers.extend([
            Rearrange('B L  -> B L 1'),
            nn.Linear(1, num_channel),
            Act(),
            Rearrange('B L H -> B H L'),
            DSSM(num_channel, s4_hidden_size, lr=s4_learning_rate),
            Rearrange('B H L -> B L H'),
            Act(),
        ])
        for _ in range(model_depth):
            layers.extend([
                nn.Linear(num_channel, num_channel),
                Act(),
                Rearrange('B L H -> B H L'),
                DSSM(num_channel, s4_hidden_size, lr=s4_learning_rate),
                Rearrange('B H L -> B L H'),
                Act(),
            ])

        layers.extend([
            nn.Linear(num_channel, 1),
            Rearrange('B L 1 -> B L')
        ])

        if amp:
            layers.append(Amplitude())

        self.side_chain = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return x * self.side_chain(x)
