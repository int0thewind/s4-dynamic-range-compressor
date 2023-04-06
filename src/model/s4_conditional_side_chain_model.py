from typing import Literal, get_args

import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .activation import Activation, get_activation_type_from
from .layer import DSSM, Amplitude, Decibel, FiLM

ModelVersion = Literal[1, 2]


class BlockV1(nn.Module):
    def __init__(
        self, conditional_information_dimension: int,
        film_take_batch_normalization: bool,
        inner_audio_channel: int, s4_hidden_size: int,
        s4_learning_rate: float | None,
        activation: Activation,
    ):
        super().__init__()
        Act = get_activation_type_from(activation)
        self.linear = nn.Linear(inner_audio_channel, inner_audio_channel)
        self.act1 = Act()
        self.s4 = DSSM(
            inner_audio_channel,
            s4_hidden_size,
            lr=s4_learning_rate
        )
        self.film = FiLM(
            inner_audio_channel,
            conditional_information_dimension,
            film_take_batch_normalization
        )
        self.act2 = Act()

    def forward(self, x: Tensor, conditional_information: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.act1(x)

        x = rearrange(x, 'B L H -> B H L')
        x = self.s4(x)
        x = rearrange(x, 'B H L -> B L H')
        x = self.film(x, conditional_information)
        x = self.act2(x)

        return x


class BlockV2(nn.Module):
    def __init__(
        self, conditional_information_dimension: int,
        film_take_batch_normalization: bool,
        inner_audio_channel: int, s4_hidden_size: int,
        s4_learning_rate: float | None,
        activation: Activation,

    ):
        super().__init__()
        Act = get_activation_type_from(activation)

        self.s4 = DSSM(
            inner_audio_channel,
            s4_hidden_size,
            lr=s4_learning_rate
        )
        self.film = FiLM(
            inner_audio_channel,
            conditional_information_dimension,
            film_take_batch_normalization
        )
        self.act = Act()

    def forward(self, x: Tensor, conditional_information: Tensor) -> Tensor:
        x = rearrange(x, 'B L H -> B H L')
        x = self.s4(x)
        x = rearrange(x, 'B H L -> B L H')
        x = self.film(x, conditional_information)
        x = self.act(x)
        return x


class S4ConditionalSideChainModel(nn.Module):
    control_parameter_mlp: nn.Sequential
    decibel: Decibel | None
    expansion: nn.Linear
    side_chain_blocks: nn.ModuleList
    contraction: nn.Linear
    amplitude: Amplitude | None

    def __init__(
        self,
        model_version: ModelVersion,
        control_parameter_mlp_depth: int,
        control_parameter_mlp_hidden_size: int,
        film_take_batch_normalization: bool,
        inner_audio_channel: int,
        s4_hidden_size: int,
        s4_learning_rate: float | None,
        side_chain_depth: int,
        activation: Activation,
        convert_to_decibels: bool
    ):
        if not model_version in get_args(ModelVersion):
            raise ValueError(
                f'Unsupported side chain version. '
                f'Expect one of {get_args(ModelVersion)}, but got {model_version}.'
            )
        super().__init__()

        control_parameter_mlp_layers: list[nn.Module] = [
            nn.Linear(2, control_parameter_mlp_hidden_size),
            nn.GELU(),
        ]
        for _ in range(control_parameter_mlp_depth):
            control_parameter_mlp_layers.extend([
                nn.Linear(control_parameter_mlp_hidden_size,
                          control_parameter_mlp_hidden_size),
                nn.GELU(),
            ])
        self.control_parameter_mlp = nn.Sequential(
            *control_parameter_mlp_layers,
        )

        self.decibel = Decibel() if convert_to_decibels else None
        self.expansion = nn.Linear(1, inner_audio_channel)

        if model_version == 1:
            Block = BlockV1
        else:
            Block = BlockV2

        self.side_chain_blocks = nn.ModuleList([
            Block(
                control_parameter_mlp_hidden_size,
                film_take_batch_normalization,
                inner_audio_channel,
                s4_hidden_size,
                s4_learning_rate,
                activation,
            ) for _ in range(side_chain_depth)
        ])

        self.contraction = nn.Linear(inner_audio_channel, 1)
        self.amplitude = Amplitude() if convert_to_decibels else None

    def forward(self, x: Tensor, control_parameters: Tensor) -> Tensor:
        cond = self.control_parameter_mlp(control_parameters)
        if self.decibel is not None:
            x = self.decibel(x)
        x = x.unsqueeze_(-1)
        x = self.expansion(x)
        for block in self.side_chain_blocks:
            x = block(x, cond)
        x = self.contraction(x)
        x = x.squeeze_(-1)
        if self.amplitude is not None:
            x = self.amplitude(x)
        return x
