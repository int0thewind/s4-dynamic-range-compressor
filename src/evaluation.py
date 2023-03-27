from typing import Literal, get_args

import numpy as np
import torch
from einops import rearrange
from numpy.typing import NDArray
from torch import Tensor

__all__ = ['calculate_waveform_difference', 'calculate_rms_difference']


ReductionMethod = Literal['rms', 'mean', 'sum']


@torch.no_grad()
def reduce_sequence_by(reduction_method: ReductionMethod, sequence: Tensor) -> float:
    assert sequence.dim() == 1
    if reduction_method not in get_args(ReductionMethod):
        raise ValueError(
            f'Unsupported sequence reduction method `{reduction_method}`.')

    if reduction_method == 'rms':
        return (sequence ** 2).mean().sqrt().item()
    if reduction_method == 'mean':
        return sequence.mean().item()
    return sequence.sum().item()


@torch.no_grad()
def calculate_waveform_difference(
    prediction: Tensor, target: Tensor, reduction_method: ReductionMethod = 'rms',
) -> tuple[NDArray[np.float32], float]:
    assert prediction.dim() == 1 and target.dim() == 1
    diff = target.flatten().abs() - prediction.flatten().abs()
    reduction = reduce_sequence_by(reduction_method, diff)
    return diff.detach().cpu().numpy(), reduction


@torch.no_grad()
def calculate_rms_difference(
    prediction: Tensor, target: Tensor, reduction_method: ReductionMethod = 'rms',
) -> tuple[NDArray[np.float32], float]:
    assert prediction.dim() == 1 and target.dim() == 1

    def process(t: Tensor):
        u = t.flatten()
        s = u.size()[0]
        s, st = divmod(s, 100)
        if st != 0:
            u = u[:-st]
        u = rearrange(u, '(S T) -> S T', T=100)

        return (u ** 2).sum(dim=1).sqrt().flatten()

    diff = process(prediction) - process(target)
    reduction = reduce_sequence_by(reduction_method, diff)
    return diff.detach().cpu().numpy(), reduction
