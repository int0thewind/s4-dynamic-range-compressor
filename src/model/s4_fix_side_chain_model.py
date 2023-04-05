"""Definitions of actual training module.
"""
from typing import Literal, get_args

import torch.nn as nn
from torch import Tensor

from .activation import Activation, get_activation_type_from
from .layer import DSSM, Amplitude, Decibel, Rearrange

ModelVersion = Literal[0, 1, 2, 3, 4]


class S4FixSideChainModel(nn.Module):
    side_chain: nn.Module

    def __init__(
        self, model_version: ModelVersion,
        inner_audio_channel: int,
        s4_hidden_size: int,
        s4_learning_rate: float | None,
        model_depth: int,
        activation: Activation,
        convert_to_decibels: bool,
    ):
        if not model_version in get_args(ModelVersion):
            raise ValueError(
                f'Unsupported model version. '
                f'Expect one of {get_args(ModelVersion)}, but got {model_version}.'
            )
        if inner_audio_channel < 1:
            raise ValueError()
        if s4_hidden_size < 1:
            raise ValueError()
        if model_depth < 0:
            raise ValueError()

        super().__init__()

        layers: list[nn.Module] = []
        if convert_to_decibels:
            layers.append(Decibel())

        Act = get_activation_type_from(activation)

        layers.extend([
            Rearrange('B L -> B L 1'),
            nn.Linear(1, inner_audio_channel),
        ])

        if model_version == 1:
            layers.extend([
                Act(),
                Rearrange('B L H -> B H L'),
                DSSM(inner_audio_channel, s4_hidden_size, lr=s4_learning_rate),
                Rearrange('B H L -> B L H'),
            ])
            for _ in range(model_depth):
                layers.extend([
                    nn.Linear(inner_audio_channel, inner_audio_channel),
                    Act(),
                    Rearrange('B L H -> B H L'),
                    DSSM(inner_audio_channel, s4_hidden_size,
                         lr=s4_learning_rate),
                    Rearrange('B H L -> B L H'),
                ])
        elif model_version == 2:
            layers.extend([
                Rearrange('B L H -> B H L'),
                DSSM(inner_audio_channel, s4_hidden_size, lr=s4_learning_rate),
                Rearrange('B H L -> B L H'),
                Act(),
            ])
            for _ in range(model_depth):
                layers.extend([
                    nn.Linear(inner_audio_channel, inner_audio_channel),
                    Rearrange('B L H -> B H L'),
                    DSSM(inner_audio_channel, s4_hidden_size,
                         lr=s4_learning_rate),
                    Rearrange('B H L -> B L H'),
                    Act(),
                ])
        elif model_version == 3:
            layers.extend([
                Act(),
                Rearrange('B L H -> B H L'),
                DSSM(inner_audio_channel, s4_hidden_size, lr=s4_learning_rate),
                Rearrange('B H L -> B L H'),
                Act(),
            ])
            for _ in range(model_depth):
                layers.extend([
                    nn.Linear(inner_audio_channel, inner_audio_channel),
                    Act(),
                    Rearrange('B L H -> B H L'),
                    DSSM(inner_audio_channel, s4_hidden_size,
                         lr=s4_learning_rate),
                    Rearrange('B H L -> B L H'),
                    Act(),
                ])
        elif model_version == 4:
            layers.extend([
                Act(),
                Rearrange('B L H -> B H L'),
                DSSM(inner_audio_channel, s4_hidden_size, lr=s4_learning_rate),
                Rearrange('B H L -> B L H'),
                Act(),
            ])
            for _ in range(model_depth):
                layers.extend([
                    Rearrange('B L H -> B H L'),
                    DSSM(inner_audio_channel, s4_hidden_size,
                         lr=s4_learning_rate),
                    Rearrange('B H L -> B L H'),
                    Act(),
                ])
        else:
            layers.extend([
                Rearrange('B L  -> B L 1'),
                nn.Linear(1, inner_audio_channel),
                Act(),
            ])
            for _ in range(model_depth):
                layers.extend([
                    nn.Linear(inner_audio_channel, inner_audio_channel),
                    Act(),
                ])

        layers.extend([
            nn.Linear(inner_audio_channel, 1),
            Rearrange('B L 1 -> B L')
        ])

        if convert_to_decibels:
            layers.append(Amplitude())

        self.side_chain = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.side_chain(x)
