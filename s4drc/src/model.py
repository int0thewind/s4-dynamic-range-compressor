from typing import Literal, TypedDict

import pytorch_lightning as pl
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.optim import AdamW

from .loss import (LossType, forge_loss_criterion_by,
                   forge_validation_criterions_by)
from .module import DSSM, Amplitude, Decibel, FiLM
from .module.activation import Activation, PTanh, get_activation_type_from

TanhType = Literal['none', 'tanh', 'ptanh']


class S4Block(nn.Module):
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
    

class S4ModelParam(TypedDict):
    learning_rate: float

    loss: LossType
    loss_filter_coef: float

    inner_audio_channel: int
    s4_hidden_size: int
    depth: int
    take_side_chain: bool
    take_batchnorm: bool
    take_residual_connection: bool
    convert_to_decibels: bool
    convert_to_amplitude: bool
    tanh: TanhType
    activation: Activation


class S4Model(pl.LightningModule):
    hparams: S4ModelParam

    def __init__(
        self, 
        learning_rate: float = 1e-3,  # saved in self.hparams

        loss: LossType = 'MAE+Multi-STFT',
        loss_filter_coef: float = 0.85,

        inner_audio_channel: int = 32,
        s4_hidden_size: int = 4,
        depth: int = 4,
        take_side_chain: bool = False,  # saved in self.hparams
        tanh: TanhType = 'tanh',
        activation: Activation = 'PReLU',
        take_batchnorm: bool = True,
        take_residual_connection: bool = True,
        convert_to_decibels: bool = False,
        convert_to_amplitude: bool = False,
    ):
        if inner_audio_channel < 1:
            raise ValueError(
                f'The inner audio channel is expected to be one or greater, but got {inner_audio_channel}.')
        if s4_hidden_size < 1:
            raise ValueError(
                f'The S4 hidden size is expected to be one or greater, but got {s4_hidden_size}.')
        if depth < 0:
            raise ValueError(
                f'The model depth is expected to be zero or greater, but got {depth}.')
        
        super().__init__()

        self.save_hyperparameters()

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
        self.blocks = nn.ModuleList([S4Block(
            32,
            inner_audio_channel,
            s4_hidden_size,
            take_batchnorm,
            take_residual_connection,
            activation,
        ) for _ in range(depth)])
        self.contract = nn.Linear(inner_audio_channel, 1)

        self.amplitude = Amplitude() if convert_to_amplitude else None

        if tanh == 'ptanh':
            self.tanh = PTanh()
        elif tanh == 'tanh':
            self.tanh = nn.Tanh()    
        else:
            self.tanh = None

        self.training_criterion = forge_loss_criterion_by(loss, loss_filter_coef)
        self.validation_criterions = forge_validation_criterions_by(loss_filter_coef, loss)

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

        if self.hparams['take_side_chain']:
            out *= x

        return out
    
    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        x, y, cond = batch

        y_hat = self(x, cond)
        loss = self.training_criterion(y_hat.unsqueeze(1), y.unsqueeze(1))

        self.log(f'Training Loss: {self.hparams["loss"]}', loss)

        return loss
    
    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        x, y, cond = batch

        y_hat = self(x, cond)
        for criterion_name, criterion in self.validation_criterions.items():
            loss = criterion(y_hat.unsqueeze(1), y.unsqueeze(1))
            self.log(f'Validation Loss: {criterion_name}', loss)

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        x, y, cond = batch

        y_hat = self(x, cond)
        for criterion_name, criterion in self.validation_criterions.items():
            if '+' in criterion_name:
                continue
            loss = criterion(y_hat.unsqueeze(1), y.unsqueeze(1))
            self.log(f'Validation Loss: {criterion_name}', loss)
    
    def configure_optimizers(self):
        s4_layers: list[nn.Parameter] = []
        other_layers: list[nn.Parameter] = []
        for name, parameter in self.named_parameters():
            (s4_layers if 's4' in name else other_layers).append(parameter)

        return AdamW(
            [
                {'params': s4_layers, 'weight_decay': 0.0},
                {'params': other_layers}
            ],
            lr=self.hparams['learning_rate']
        )
