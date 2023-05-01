from typing import Literal, get_args

import torch.nn as nn
from torch import Tensor

from .activation import Activation, get_activation_type_from
from .layer import DSSM, Amplitude, Decibel, Rearrange


class S4ConiditionalModel(nn.Module):
    side_chain: nn.Module

    def __init__(
        self,
        inner_audio_channel: int,
        s4_hidden_size: int,
        s4_learning_rate: float | None,
        model_depth: int,
        activation: Activation,
    ):
        if inner_audio_channel < 1:
            raise ValueError()
        if s4_hidden_size < 1:
            raise ValueError()
        if model_depth < 0:
            raise ValueError()

        super().__init__()
        Act = get_activation_type_from(activation)

        layers: list[nn.Module] = []

        layers.extend([
            Rearrange('B L -> B L 1'),
            nn.Linear(1, inner_audio_channel),
        ])

        for _ in range(model_depth):
            pass

        layers.extend([
            nn.Linear(inner_audio_channel, 1),
            Rearrange('B L 1 -> B L')
        ])

        self.side_chain = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.side_chain(x)
