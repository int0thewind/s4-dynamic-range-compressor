from typing import Literal, get_args

import torch.nn as nn
from auraloss.freq import MultiResolutionSTFTLoss
from auraloss.perceptual import FIRFilter
from auraloss.time import DCLoss, ESRLoss
from torch import Tensor

LossType = Literal[
    'MSE', 'ESR+DC', 'Multi-STFT', 'ESR+DC+Multi-STFT',
]

FilterType = Literal['hp', 'lp']


class EsrDcLoss(nn.Module):
    def __init__(self, filter_type: FilterType, filter_coef: float) -> None:
        super().__init__()
        if filter_coef <= 0:
            self.preemphasis_filter = None
        else:
            self.preemphasis_filter = FIRFilter(filter_type, filter_coef)

        self.esr = ESRLoss()
        self.dc = DCLoss()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.preemphasis_filter:
            esr_loss = self.esr(*self.preemphasis_filter(y_hat, y))
        else:
            esr_loss = self.esr(y_hat, y)

        return esr_loss + self.dc(y_hat, y)


class EsrDcStftLoss(nn.Module):
    def __init__(self, filter_type: FilterType, filter_coef: float) -> None:
        super().__init__()
        if filter_coef <= 0:
            self.preemphasis_filter = None
        else:
            self.preemphasis_filter = FIRFilter(filter_type, filter_coef)

        self.esr = ESRLoss()
        self.dc = DCLoss()
        self.stft = MultiResolutionSTFTLoss()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.preemphasis_filter:
            esr_loss = self.esr(*self.preemphasis_filter(y_hat, y))
        else:
            esr_loss = self.esr(y_hat, y)

        return esr_loss + self.dc(y_hat, y) + self.stft(y_hat, y)


def forge_loss_function_from(
    loss_type: LossType, filter_type: FilterType, filter_coef: float,
) -> nn.Module:
    if not loss_type in get_args(LossType):
        raise ValueError(f'Unsupported loss type `{loss_type}`.')

    if loss_type == 'MSE':
        return nn.MSELoss()
    elif loss_type == 'ESR+DC':
        return EsrDcLoss(filter_type, filter_coef)
    elif loss_type == 'Multi-STFT':
        return MultiResolutionSTFTLoss()

    return EsrDcStftLoss(filter_type, filter_coef)
