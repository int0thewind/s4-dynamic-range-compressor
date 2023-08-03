from typing import Literal, get_args

import pytorch_lightning as pl
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .activation import Activation, PTanh, get_activation_type_from
from .layer import DSSM, Amplitude, Decibel, FiLM


class Block(nn.Module):
    def __init__(
        self,
        conditional_information_dimension: int,
        inner_audio_channel: int,
        s4_hidden_size: int,
        take_batchnorm: bool,
        take_residual_connection: bool,
        activation: Activation
    ):
        super().__init__()

        Act = get_activation_type_from(activation)

        self.linear = nn.Linear(inner_audio_channel, inner_audio_channel)
        self.activation1 = Act()
        self.s4 = DSSM(inner_audio_channel,s4_hidden_size)
        self.batchnorm = nn.BatchNorm1d(inner_audio_channel, affine=False) if take_batchnorm else None
        self.film = FiLM(inner_audio_channel,
                         conditional_information_dimension)
        self.activation2 = Act()
        self.residual_connection = nn.Conv1d(
            inner_audio_channel,
            inner_audio_channel,
            kernel_size=1,
            groups=inner_audio_channel,
            bias=False
        ) if take_residual_connection else None

    def forward(self, x: Tensor, conditional_information: Tensor) -> Tensor:
        out = rearrange(x, 'B H L -> B L H')
        out = self.linear(out)
        out = rearrange(out, 'B L H -> B H L')

        out = self.activation1(out)
        out = self.s4(out)
        if self.batchnorm:
            out = self.batchnorm(out)
        out = self.film(out, conditional_information)
        out = self.activation2(out)

        if self.residual_connection:
            out += self.residual_connection(x)

        return out


class S4ConditionalModel(pl.LightningModule):
    def __init__(
        self,
        take_side_chain: bool,
        inner_audio_channel: int,
        s4_hidden_size: int,
        model_depth: int,
        take_batchnorm: bool,
        take_residual_connection: bool,
        convert_to_decibels: bool,
        convert_to_amplitude: bool,
        tanh: Literal['none', 'tanh', 'ptanh'],
        activation: Activation,
    ):
        if inner_audio_channel < 1:
            raise ValueError(
                f'The inner audio channel is expected to be one or greater, but got {inner_audio_channel}.')
        if s4_hidden_size < 1:
            raise ValueError(
                f'The S4 hidden size is expected to be one or greater, but got {s4_hidden_size}.')
        if model_depth < 0:
            raise ValueError(
                f'The model depth is expected to be zero or greater, but got {model_depth}.')
        
        self.save_hyperparameters()

        super().__init__()

        self.control_parameter_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.decibel = Decibel() if convert_to_decibels else None

        self.expand = nn.Linear(1, inner_audio_channel)
        self.blocks = nn.ModuleList([Block(
            32,
            inner_audio_channel,
            s4_hidden_size,
            take_batchnorm,
            take_residual_connection,
            activation,
        ) for _ in range(model_depth)])
        self.contract = nn.Linear(inner_audio_channel, 1)

        self.amplitude = Amplitude() if convert_to_amplitude else None

        if tanh == 'ptanh':
            self.tanh = PTanh()
        elif tanh == 'tanh':
            self.tanh = nn.Tanh()    
        else:
            self.tanh = None

        self.take_side_chain = take_side_chain

    def forward(self, x: Tensor, parameters: Tensor) -> Tensor:
        conditional_information = self.control_parameter_mlp(parameters)
        out = rearrange(x, 'B L -> B L 1')
        if self.decibel:
            out = self.decibel(out)

        out = self.expand(out)

        out = rearrange(out, 'B L H -> B H L')
        for block in self.blocks:
            out = block(out, conditional_information)
        out = rearrange(out, 'B H L -> B L H')

        out = self.contract(out)

        if self.amplitude:
            out = self.amplitude(out)
        out = rearrange(out, 'B H 1 -> B H')

        if self.tanh:
            out = self.tanh(out)

        if self.take_side_chain:
            out *= x

        return out
