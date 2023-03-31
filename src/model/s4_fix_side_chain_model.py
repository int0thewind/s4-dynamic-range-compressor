"""Definitions of actual training module.
"""
from abc import ABC, abstractmethod
from typing import Literal, Type, get_args

import torch.nn as nn
from torch import Tensor

from .layer import DSSM, Amplitude, Decibel, Rearrange

Activation = Literal['tanh', 'sigmoid', 'GELU', 'ReLU', 'Identity']
DRCSideChainModelVersion = Literal[0, 1, 2, 3]


def get_activation_type_from(activation: Activation) -> Type[nn.Module]:
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
    return nn.Identity


class AbstractS4FixSideChainModel(ABC, nn.Module):
    side_chain: nn.Module

    @staticmethod
    @abstractmethod
    def forge_layer_sequence(
        inner_audio_channel: int, s4_hidden_size: int,
        s4_learning_rate: float | None,
        model_depth: int, activation: Activation,
    ) -> list[nn.Module]:
        pass

    def __init__(
        self, num_channel: int, s4_hidden_size: int,
        s4_learning_rate: float | None,
        model_depth: int, activation: Activation,
        take_db: bool, take_amp: bool,
    ):
        super().__init__()
        if num_channel < 1:
            raise ValueError()
        if s4_hidden_size < 1:
            raise ValueError()
        if model_depth < 0:
            raise ValueError()

        layers: list[nn.Module] = []
        if take_db:
            layers.append(Decibel())

        layers.extend(self.forge_layer_sequence(
            num_channel, s4_hidden_size,
            s4_learning_rate,
            model_depth, activation,
        ))

        if take_amp:
            layers.append(Amplitude())

        self.side_chain = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.side_chain(x)


class S4FixSideChainModelV0(AbstractS4FixSideChainModel):
    @staticmethod
    def forge_layer_sequence(
        inner_audio_channel: int, s4_hidden_size: int,
        s4_learning_rate: float | None,
        model_depth: int, activation: Activation
    ) -> list[nn.Module]:
        layers: list[nn.Module] = []

        Act = get_activation_type_from(activation)

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

        return layers


class S4FixSideChainModelV1(AbstractS4FixSideChainModel):
    @staticmethod
    def forge_layer_sequence(
        inner_audio_channel: int, s4_hidden_size: int,
        s4_learning_rate: float | None,
        model_depth: int, activation: Activation
    ) -> list[nn.Module]:
        layers: list[nn.Module] = []

        Act = get_activation_type_from(activation)

        layers.extend([
            Rearrange('B L  -> B L 1'),
            nn.Linear(1, inner_audio_channel),
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
                DSSM(inner_audio_channel, s4_hidden_size, lr=s4_learning_rate),
                Rearrange('B H L -> B L H'),
            ])
        layers.extend([
            nn.Linear(inner_audio_channel, 1),
            Rearrange('B L 1 -> B L')
        ])

        return layers


class S4FixSideChainModelV2(AbstractS4FixSideChainModel):
    @staticmethod
    def forge_layer_sequence(
        inner_audio_channel: int, s4_hidden_size: int,
        s4_learning_rate: float | None,
        model_depth: int, activation: Activation
    ) -> list[nn.Module]:
        layers: list[nn.Module] = []

        Act = get_activation_type_from(activation)

        layers.extend([
            Rearrange('B L  -> B L 1'),
            nn.Linear(1, inner_audio_channel),
            Rearrange('B L H -> B H L'),
            DSSM(inner_audio_channel, s4_hidden_size, lr=s4_learning_rate),
            Rearrange('B H L -> B L H'),
            Act(),
        ])
        for _ in range(model_depth):
            layers.extend([
                nn.Linear(inner_audio_channel, inner_audio_channel),
                Rearrange('B L H -> B H L'),
                DSSM(inner_audio_channel, s4_hidden_size, lr=s4_learning_rate),
                Rearrange('B H L -> B L H'),
                Act(),
            ])
        layers.extend([
            nn.Linear(inner_audio_channel, 1),
            Rearrange('B L 1 -> B L')
        ])

        return layers


class S4FixSideChainModelV3(AbstractS4FixSideChainModel):
    @staticmethod
    def forge_layer_sequence(
        inner_audio_channel: int, s4_hidden_size: int,
        s4_learning_rate: float | None,
        model_depth: int, activation: Activation
    ) -> list[nn.Module]:
        layers: list[nn.Module] = []

        Act = get_activation_type_from(activation)

        layers.extend([
            Rearrange('B L  -> B L 1'),
            nn.Linear(1, inner_audio_channel),
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
                DSSM(inner_audio_channel, s4_hidden_size, lr=s4_learning_rate),
                Rearrange('B H L -> B L H'),
                Act(),
            ])
        layers.extend([
            nn.Linear(inner_audio_channel, 1),
            Rearrange('B L 1 -> B L')
        ])

        return layers


def forge_s4_fix_side_chain_model_by(
    model_version: DRCSideChainModelVersion,
    inner_audio_channel: int, s4_hidden_size: int,
    s4_learning_rate: float | None,
    model_depth: int, activation: Activation,
    take_db: bool, take_amp: bool,
) -> AbstractS4FixSideChainModel:
    if not model_version in get_args(DRCSideChainModelVersion):
        raise ValueError(
            f'Unsupported model version. Expect one of {get_args(DRCSideChainModelVersion)}.')
    if model_version == 1:
        TrainingModel = S4FixSideChainModelV1
    elif model_version == 2:
        TrainingModel = S4FixSideChainModelV2
    elif model_version == 3:
        TrainingModel = S4FixSideChainModelV3
    else:
        TrainingModel = S4FixSideChainModelV0

    return TrainingModel(
        inner_audio_channel, s4_hidden_size,
        s4_learning_rate,
        model_depth, activation,
        take_db, take_amp,
    )
