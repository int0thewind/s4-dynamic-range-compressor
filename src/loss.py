from functools import reduce
from typing import Literal, get_args

import torch.nn as nn
from auraloss.freq import MultiResolutionSTFTLoss
from auraloss.perceptual import FIRFilter
from auraloss.time import DCLoss, ESRLoss
from torch import Tensor

__all__ = ['forge_loss_function_from']

LossType = Literal['MAE', 'MSE', 'ESR', 'DC', 'Multi-STFT',
                   'ESR+DC', 'ESR+DC+Multi-STFT']


class Sum(nn.Module):
    losses: nn.ModuleList

    def __init__(self, *losses: nn.Module):
        super().__init__()
        self.losses = nn.ModuleList(losses)

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return reduce(
            lambda x, y: x + y,
            (loss(y_hat, y) for loss in self.losses),
        )


class MAELoss(nn.Module):
    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return (y_hat - y).abs().mean()


class PreEmphasisESRLoss(nn.Module):
    def __init__(self, filter_coef: float | None):
        super().__init__()
        if filter_coef is not None and 0 < filter_coef < 1:
            self.pre_emphasis_filter = FIRFilter('hp', filter_coef, 44100)
        else:
            self.pre_emphasis_filter = None
        self.esr = ESRLoss()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        assert y_hat.dim() == 2 and y.dim() == 2
        if self.pre_emphasis_filter:
            y_hat, y = self.pre_emphasis_filter(
                y_hat.unsqueeze(1), y.unsqueeze(1))
        return self.esr(y_hat, y)


def forge_loss_function_from(loss_type: LossType, filter_coef: float) -> nn.Module:
    if not loss_type in get_args(LossType):
        raise ValueError(f'Unsupported loss type `{loss_type}`.')
    if loss_type == 'MAE':
        return MAELoss()
    if loss_type == 'MSE':
        return nn.MSELoss()
    if loss_type == 'ESR':
        return PreEmphasisESRLoss(filter_coef)
    if loss_type == 'DC':
        return DCLoss()
    if loss_type == 'Multi-STFT':
        return MultiResolutionSTFTLoss()
    if loss_type == 'ESR+DC':
        return Sum(PreEmphasisESRLoss(filter_coef), DCLoss())
    return Sum(PreEmphasisESRLoss(filter_coef), DCLoss(), MultiResolutionSTFTLoss())
