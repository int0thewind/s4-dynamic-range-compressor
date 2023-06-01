import torch.nn as nn
from einops import rearrange
from torch import Tensor


class Rearrange(nn.Module):
    """Tensor reshape layer.

    This layer wraps the `einops.rearrange` method as an neural network layer
    to streamline tensor reshape operations.
    Useful for `nn.Sequential` layers.

    The constructor of the layer follows the same signature as the `einops.rearrange`
    except the first argument, which is the input tensor.
    See `einops.rearrange` method for help.

    ```python
    x = torch.rand(3, 6)
    layer = Rearrange('b (c d) -> b d c', c=2)
    x = layer(x)
    x.size()
    >>> torch.Size([3, 3, 2])
    ```
    """
    pattern: str
    axes_lengths: dict[str, int]

    def __init__(self, pattern: str, **axes_lengths: int) -> None:
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths

    def forward(self, x: Tensor) -> Tensor:
        return rearrange(x, self.pattern, **self.axes_lengths)
