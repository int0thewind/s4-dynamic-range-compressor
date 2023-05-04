import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


class FiLM(nn.Module):
    conditional_information_adaptor: nn.Linear
    batchnorm: nn.BatchNorm1d | None

    def __init__(
        self, feature_numbers: int,
        conditional_information_dimension: int,
        take_batchnorm: bool = False,
    ):
        super().__init__()
        self.conditional_information_adaptor = nn.Linear(
            conditional_information_dimension,
            feature_numbers * 2
        )
        self.batchnorm = nn.BatchNorm1d(
            feature_numbers, affine=False) if take_batchnorm else None

    def forward(self, x: Tensor, conditional_information: Tensor) -> Tensor:
        cond = self.conditional_information_adaptor(conditional_information)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = rearrange(g, 'B H -> B H 1')
        b = rearrange(b, 'B H -> B H 1')

        x = rearrange(x, 'B L H -> B H L')
        if self.batchnorm:
            x = self.batchnorm(x)
        x = x * g + b
        x = rearrange(x, 'B H L -> B L H')

        return x
