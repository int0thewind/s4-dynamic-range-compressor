import torch
import torch.nn as nn
from torch import Tensor


class FiLM(nn.Module):
    batch_norm: nn.BatchNorm1d
    adaptor: nn.Linear

    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = nn.Linear(cond_dim, num_features * 2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = g.permute(0, 2, 1)
        b = b.permute(0, 2, 1)

        x = self.batch_norm(x)
        x = x * g + b

        return x
