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
    ):
        super().__init__()
        self.conditional_information_adaptor = nn.Linear(
            conditional_information_dimension,
            feature_numbers * 2
        )

    def forward(self, x: Tensor, conditional_information: Tensor) -> Tensor:
        conditional_information = self.conditional_information_adaptor(
            conditional_information)
        g, b = torch.chunk(conditional_information, 2, dim=-1)
        g = rearrange(g, 'B H -> B H 1')
        b = rearrange(b, 'B H -> B H 1')
        x = x * g + b
        return x
