from typing import Literal, get_args

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .activation import Activation, get_activation_type_from
from .layer import DSSM, FiLM, Rearrange


class Blocks(nn.Module):
    def __init__(
        self,
        inner_audio_channel: int,
        s4_hidden_size: int,
        s4_learning_rate: float | None,
        conditional_information_dimension: int,
        film_take_batchnorm: bool,
        activation: Activation
    ):
        super().__init__()

        Act = get_activation_type_from(activation)

        self.linear = nn.Linear(inner_audio_channel, inner_audio_channel)
        self.activation1 = Act()
        self.s4 = DSSM(inner_audio_channel,
                       s4_hidden_size, lr=s4_learning_rate)
        self.film = FiLM(inner_audio_channel,
                         conditional_information_dimension,
                         film_take_batchnorm)
        self.activation2 = Act()
        self.res = torch.nn.Conv1d(inner_audio_channel,
                                   inner_audio_channel,
                                   kernel_size=1,
                                   groups=inner_audio_channel,
                                   bias=False)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x_res = x

        x = self.linear(x)
        x = self.activation1(x)
        x = rearrange(x, 'B L H -> B H L')
        x = self.s4(x)
        x = rearrange(x, 'B H L -> B L H')
        x = self.film(x, cond)
        x = self.activation2(x)

        x_res = rearrange(x_res, 'B L H -> B H L')
        x_res = self.res(x_res)
        x_res = rearrange(x_res, 'B H L -> B L H')

        return x + x_res


class S4ConditionalModel(nn.Module):
    blocks: nn.ModuleList
    control_parameter_mlp_layers: nn.Sequential

    def __init__(
        self,
        inner_audio_channel: int,
        s4_hidden_size: int,
        s4_learning_rate: float | None,
        model_depth: int,
        take_batchnorm: bool,
        activation: Activation,
    ):
        if inner_audio_channel < 1:
            raise ValueError()
        if s4_hidden_size < 1:
            raise ValueError()
        if model_depth < 0:
            raise ValueError()

        super().__init__()

        self.control_parameter_mlp_layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.expand = nn.Linear(1, inner_audio_channel)

        self.blocks = nn.ModuleList()
        for _ in range(model_depth):
            self.blocks.append(
                Blocks(
                    inner_audio_channel,
                    s4_hidden_size,
                    s4_learning_rate,
                    32,
                    take_batchnorm,
                    activation,
                )
            )

        self.contract = nn.Linear(inner_audio_channel, 1)

    def forward(self, x: Tensor, param: Tensor) -> Tensor:
        cond = self.control_parameter_mlp_layers(param)

        x = rearrange(x, 'B H -> B H 1')
        x = self.expand(x)

        for block in self.blocks:
            x = block(x, cond)

        x = self.contract(x)
        x = rearrange(x, 'B H 1 -> B H')

        x = torch.tanh(x)
        return x
