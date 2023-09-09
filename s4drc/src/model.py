from typing import Literal, TypedDict

import lightning.pytorch as pl
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .loss import (LossType, forge_loss_criterion_by,
                   forge_validation_criterions_by)
from .module.film import FiLM
from .module.s4 import FFTConv as S4


class S4Block(nn.Module):
    def __init__(
        self,
        conditional_information_dimension: int,
        inner_audio_channel: int,
        s4_hidden_size: int,
    ):
        super().__init__()

        self.linear = nn.Linear(inner_audio_channel, inner_audio_channel)
        self.activation1 = nn.PReLU()
        self.s4 = S4(inner_audio_channel, activation='id', mode='diag', d_state=s4_hidden_size)
        self.batchnorm = nn.BatchNorm1d(inner_audio_channel, affine=False)
        self.film = FiLM(inner_audio_channel,
                         conditional_information_dimension)
        self.activation2 = nn.PReLU()
        self.residual_connection = nn.Conv1d(
            inner_audio_channel,
            inner_audio_channel,
            kernel_size=1,
            groups=inner_audio_channel,
            bias=False
        )

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
    loss_filter_coef: float
    inner_audio_channel: int
    s4_hidden_size: int
    depth: int


class S4Model(pl.LightningModule):
    loss = 'MAE+Multi-STFT'

    hparams: S4ModelParam

    def __init__(
        self, 
        learning_rate: float = 1e-3,  # saved in self.hparams
        loss_filter_coef: float = 0.85,
        inner_audio_channel: int = 32,
        s4_hidden_size: int = 4,
        depth: int = 4,
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

        self.expand = nn.Linear(1, inner_audio_channel)
        self.blocks = nn.ModuleList([S4Block(
            32,
            inner_audio_channel,
            s4_hidden_size,
        ) for _ in range(depth)])
        self.contract = nn.Linear(inner_audio_channel, 1)

        self.tanh = nn.Tanh()    

        self.training_criterion = forge_loss_criterion_by('MAE+Multi-STFT', self.hparams['loss_filter_coef'])
        self.validation_criterions = forge_validation_criterions_by(loss_filter_coef)

    def forward(self, x: Tensor, parameters: Tensor) -> Tensor:
        conditional_information = self.control_parameter_mlp(parameters)
        out = rearrange(x, 'B L -> B L 1')
        out = self.expand(out)
        out = rearrange(out, 'B L H -> B H L')
        for block in self.blocks:
            out = block(out, conditional_information)
        out = rearrange(out, 'B H L -> B L H')
        out = self.contract(out)
        out = rearrange(out, 'B H 1 -> B H')
        out = self.tanh(out)

        return out
    
    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        x, y, cond = batch

        y_hat = self(x, cond)
        loss = self.training_criterion(y_hat.unsqueeze(1), y.unsqueeze(1))

        self.log(f'Training Loss', loss)

        return loss
    
    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        x, y, cond = batch
        y_hat = self(x, cond)

        for criterion_name, criterion in self.validation_criterions.items():
            loss = criterion(y_hat.unsqueeze(1), y.unsqueeze(1))
            self.log(f'Validation Loss: {criterion_name}', loss)

        # Log an extra "Validation Loss" that is the same to the training loss.
        # This is for PyTorch Lightning validation loss monitoring
        # for saving checkpoints and scheduling learning rate.
        loss = self.training_criterion(y_hat.unsqueeze(1), y.unsqueeze(1))
        self.log(f'Validation Loss', loss)

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        x, y, cond = batch

        y_hat = self(x, cond)
        for criterion_name, criterion in self.validation_criterions.items():
            if '+' in criterion_name:
                continue
            loss = criterion(y_hat.unsqueeze(1), y.unsqueeze(1))
            self.log(f'Testing Loss: {criterion_name}', loss)
    
    def configure_optimizers(self):        
        optimizer = AdamW(self.parameters(), lr=self.hparams['learning_rate'])

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer, factor=0.1, patience=10, mode='min', verbose=True),
                'monitor': 'Validation Loss'
            }
        }
