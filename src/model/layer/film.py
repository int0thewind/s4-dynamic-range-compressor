import torch
import torch.nn as nn
from torch import Tensor


class FiLM(nn.Module):
    batch_norm: nn.Module
    conditional_information_adaptor: nn.Linear

    def __init__(
        self, feature_numbers: int,
        conditional_information_dimension: int,
        take_batch_normalization: bool,
    ):
        super().__init__()

        if take_batch_normalization:
            self.batch_norm = nn.BatchNorm1d(feature_numbers, affine=False)
        else:
            self.batch_norm = nn.Identity()

        self.conditional_information_adaptor = nn.Linear(
            conditional_information_dimension,
            feature_numbers * 2
        )

    def forward(self, x: Tensor, conditional_information: Tensor) -> Tensor:
        cond = self.conditional_information_adaptor(conditional_information)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = g.permute(0, 2, 1)
        b = b.permute(0, 2, 1)

        x = self.batch_norm(x)
        x = x * g + b

        return x
