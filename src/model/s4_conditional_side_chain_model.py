from typing import Literal, get_args

import torch.nn as nn
from torch import Tensor

from .activation import Activation, get_activation_type_from
from .layer import DSSM, FiLM, Rearrange

SideChainVersion = Literal[1, 2]


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
        self.rearrange1 = Rearrange('B L H -> B H L')
        self.s4 = DSSM(
            inner_audio_channel,
            s4_hidden_size,
            lr=s4_learning_rate
        )
        self.rearrange2 = Rearrange('B H L -> B L H')
        self.film = FiLM(
            inner_audio_channel,
            conditional_information_dimension,
            film_take_batch_normalization
        )
        self.act2 = Act()

    def forward(self, x: Tensor, conditional_information: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.act1(x)
        x = self.rearrange1(x)
        x = self.s4(x)
        x = self.rearrange2(x)
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

        self.rearrange1 = Rearrange('B L H -> B H L')
        self.s4 = DSSM(
            inner_audio_channel,
            s4_hidden_size,
            lr=s4_learning_rate
        )
        self.rearrange2 = Rearrange('B H L -> B L H')
        self.film = FiLM(
            inner_audio_channel,
            conditional_information_dimension,
            film_take_batch_normalization
        )
        self.act = Act()

    def forward(self, x: Tensor, conditional_information: Tensor) -> Tensor:
        x = self.s4(x)
        x = self.film(x, conditional_information)
        x = self.act(x)
        return x


class S4ConditionalSideChainModel(nn.Module):
    side_chain: nn.Module
    control_parameter_mlp: nn.Sequential

    def __init__(
        self,
        control_parameter_mlp_depth: int,
        control_parameter_mlp_hidden_size: int,
        side_chain_version: SideChainVersion,
        film_take_batch_normalization: bool,
        inner_audio_channel: int, s4_hidden_size: int,
        s4_learning_rate: float | None,
        model_depth: int, activation: Activation,
        convert_to_decibels: bool
    ):
        if not side_chain_version in get_args(SideChainVersion):
            raise ValueError(
                f'Unsupported side chain version. '
                f'Expect one of {get_args(SideChainVersion)}, but got {side_chain_version}.'
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

        Act = get_activation_type_from(activation)

        if side_chain_version == 1:
            Model = BlockV1
        else:
            Model = BlockV2

        self.side_chain = Model(
            film_take_batch_normalization,
            inner_audio_channel, s4_hidden_size,
            s4_learning_rate,
            model_depth, activation,
            convert_to_decibels,
        )

    def forward(self, x: Tensor, control_parameters: Tensor) -> Tensor:
        cond = self.control_parameter_mlp(control_parameters)
        return x * self.side_chain(x, cond)
