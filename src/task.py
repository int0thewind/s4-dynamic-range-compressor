from typing import Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

LossType = Literal[
    'mse', 'esr', 'stft',
    'esr+stft',
]


class ModelDRCTask(pl.LightningModule):
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, x: Tensor):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
