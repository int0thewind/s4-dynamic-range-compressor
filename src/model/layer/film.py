import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


class FiLM(nn.Module):
    conditional_information_adaptor: nn.Linear

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
        cond = self.conditional_information_adaptor(conditional_information)
        g, b = torch.chunk(cond, 2, dim=-1)

        x = rearrange(x, 'B L H -> L B H')
        x = x * g + b
        x = rearrange(x, 'L B H -> B L H')

        return x
