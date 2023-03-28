from typing import Literal, get_args

import torch.nn as nn
from auraloss.freq import MultiResolutionSTFTLoss
from auraloss.perceptual import FIRFilter
from auraloss.time import DCLoss, ESRLoss
from torch import Tensor

from .layer import Rearrange

__all__ = ['forge_loss_function_from']

LossType = Literal['MSE', 'ESR', 'ESR+DC', 'Multi-STFT', 'ESR+DC+Multi-STFT']


class PreEmphasisESRLoss(nn.Module):
    def __init__(self, filter_coef: float | None) -> None:
        super().__init__()
        if filter_coef is not None and 0 < filter_coef < 1:
            self.pre_emphasis_filter = FIRFilter('hp', filter_coef, 44100)
        else:
            self.pre_emphasis_filter = None
        self.esr = ESRLoss()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.pre_emphasis_filter:
            y_hat, y = self.pre_emphasis_filter(
                y_hat.unsqueeze(1), y.unsqueeze(1))
        return self.esr(y_hat, y)


class EsrDcLoss(nn.Module):
    def __init__(self, filter_coef: float) -> None:
        super().__init__()
        self.esr = PreEmphasisESRLoss(filter_coef)
        self.dc = DCLoss()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.esr(y_hat, y) + self.dc(y_hat, y)


class EsrDcStftLoss(nn.Module):
    def __init__(self, filter_coef: float) -> None:
        super().__init__()
        self.esr = PreEmphasisESRLoss(filter_coef)
        self.dc = DCLoss()
        self.stft = MultiResolutionSTFTLoss()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.esr(y_hat, y) + self.dc(y_hat, y) + self.stft(y_hat, y)


def forge_loss_function_from(loss_type: LossType, filter_coef: float) -> nn.Module:
    if not loss_type in get_args(LossType):
        raise ValueError(f'Unsupported loss type `{loss_type}`.')

    if loss_type == 'MSE':
        return nn.MSELoss()
    if loss_type == 'ESR':
        return PreEmphasisESRLoss(filter_coef)
    if loss_type == 'ESR+DC':
        return EsrDcLoss(filter_coef)
    if loss_type == 'Multi-STFT':
        return MultiResolutionSTFTLoss()
    return EsrDcStftLoss(filter_coef)
