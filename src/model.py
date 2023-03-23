"""Definitions of actual training module.
"""
from abc import ABC, abstractmethod
from typing import Literal, get_args

import torch.nn as nn
from torch import Tensor

from .layer import DSSM, Absolute, Amplitude, Decibel, Rearrange

__all__ = ['Activation', 'S4LinearModelV1']

Activation = Literal['tanh', 'sigmoid', 'GELU', 'ReLU', 'Identity']


class DRCModel(ABC, nn.Module):
    @property
    @abstractmethod
    def side_chain(self) -> nn.Module:
        pass

    def forward(self, x: Tensor) -> Tensor:
        return x * self.side_chain(x)


# class ElmanModel(DRCModel):
#     # TODO: finish Elman Network Model
#     def side_chain(self) -> nn.Module:
#         return self._side_chain

#     def __init__(self, ) -> None:
#         super().__init__()


class S4LinearModelV1(DRCModel):
    _side_chain: nn.Sequential

    def side_chain(self) -> nn.Module:
        return self._side_chain

    def __init__(
        self, num_channel: int, s4_hidden_size: int,
        s4_learning_rate: float | None,
        model_depth: int, activation: Activation,
        take_db: bool, take_abs: bool, take_amp: bool,
        take_s4: bool = True,
    ):
        if num_channel < 1:
            raise ValueError()
        if s4_hidden_size < 1:
            raise ValueError()
        if model_depth < 0:
            raise ValueError()
        if not activation in get_args(Activation):
            raise ValueError(
                f'Unsupported non-linear activation `{activation}`.'
            )

        super().__init__()

        if activation == 'tanh':
            Act = nn.Tanh
        elif activation == 'sigmoid':
            Act = nn.Sigmoid
        elif activation == 'ReLU':
            Act = nn.ReLU
        elif activation == 'GELU':
            Act = nn.GELU
        else:
            Act = nn.Identity

        layers: list[nn.Module] = []

        if take_abs:
            layers.append(Absolute())
        if take_db:
            layers.append(Decibel())

        layers.extend([
            Rearrange('B L  -> B L 1'),
            nn.Linear(1, num_channel),
            Act(),
        ])
        if take_s4:
            layers.extend([
                Rearrange('B L H -> B H L'),
                DSSM(num_channel, s4_hidden_size, lr=s4_learning_rate),
                Rearrange('B H L -> B L H'),
                Act(),
            ])
        for _ in range(model_depth):
            layers.extend([
                nn.Linear(num_channel, num_channel),
                Act(),
            ])
            if take_s4:
                layers.extend([
                    Rearrange('B L H -> B H L'),
                    DSSM(num_channel, s4_hidden_size, lr=s4_learning_rate),
                    Rearrange('B H L -> B L H'),
                    Act(),
                ])

        layers.extend([
            nn.Linear(num_channel, 1),
            Rearrange('B L 1 -> B L')
        ])

        if take_amp:
            layers.append(Amplitude())

        self._side_chain = nn.Sequential(*layers)
