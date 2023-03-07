from typing import Literal, get_args

import torch
import torch.nn as nn
from auraloss.freq import MultiResolutionSTFTLoss
from auraloss.time import DCLoss, ESRLoss
from torch import Tensor

LossType = Literal[
    'mse', 'esr+dc', 'stft', 'esr+dc+stft',
]


def forge_loss_function_from(loss_type: LossType):
    if not loss_type in get_args(LossType):
        raise ValueError(f'Unsupported loss type `{loss_type}`.')
