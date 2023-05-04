from functools import reduce
from typing import Literal, get_args

import torch
import torch.nn as nn
from auraloss.freq import MultiResolutionSTFTLoss, STFTLoss
from auraloss.perceptual import FIRFilter
from auraloss.time import DCLoss, ESRLoss
from torch import Tensor

__all__ = ['forge_loss_criterion_by', 'forge_validation_criterions_by']

LossType = Literal['MAE', 'MSE', 'ESR', 'DC', 'Multi-STFT', 'STFT',
                   'ESR+DC', 'ESR+DC+Multi-STFT', 'MAE+Multi-STFT', 'MAE+ESR+DC+Multi-STFT']


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


class PreEmphasisESRLoss(nn.Module):
    def __init__(self, filter_coef: float | None):
        super().__init__()
        if filter_coef is not None and 0 < filter_coef < 1:
            self.pre_emphasis_filter = FIRFilter('hp', filter_coef, 44100)
        else:
            self.pre_emphasis_filter = None
        self.esr = ESRLoss()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.pre_emphasis_filter:
            y_hat, y = self.pre_emphasis_filter(y_hat, y)
        return self.esr(y_hat, y)


def forge_loss_criterion_by(loss_type: LossType, filter_coef: float) -> nn.Module:
    if not loss_type in get_args(LossType):
        raise ValueError(f'Unsupported loss type `{loss_type}`.')
    if loss_type == 'MAE':
        return nn.L1Loss()
    if loss_type == 'MSE':
        return nn.MSELoss()
    if loss_type == 'ESR':
        return PreEmphasisESRLoss(filter_coef)
    if loss_type == 'DC':
        return DCLoss()
    if loss_type == 'Multi-STFT':
        return MultiResolutionSTFTLoss()
    if loss_type == 'STFT':
        return STFTLoss()
    if loss_type == 'ESR+DC':
        return Sum(PreEmphasisESRLoss(filter_coef), DCLoss())
    if loss_type == 'MAE+ESR+DC+Multi-STFT':
        return Sum(PreEmphasisESRLoss(filter_coef), DCLoss(), MultiResolutionSTFTLoss(), nn.L1Loss())
    if loss_type == 'MAE+Multi-STFT':
        return Sum(MultiResolutionSTFTLoss(), nn.L1Loss())
    return Sum(PreEmphasisESRLoss(filter_coef), DCLoss(), MultiResolutionSTFTLoss())


def forge_validation_criterions_by(
    filter_coef: float, device: torch.device, *loss_to_keep: LossType,
) -> dict[LossType, nn.Module]:
    return {
        loss_type: forge_loss_criterion_by(
            loss_type, filter_coef).eval().to(device)
        for loss_type in get_args(LossType)
        if '+' not in loss_type or loss_type in loss_to_keep
    }
