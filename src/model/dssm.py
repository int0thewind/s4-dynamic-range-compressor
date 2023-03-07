import math

import torch
import torch.nn as nn
from einops import repeat
from torch import Tensor

c2r = torch.view_as_real
r2c = torch.view_as_complex


class DSSM(nn.Module):
    log_A_real: nn.Parameter | Tensor
    A_imag: nn.Parameter | Tensor
    C: nn.Parameter
    D: nn.Parameter
    log_dt: nn.Parameter | Tensor

    H: int
    N: int

    def __init__(
        self, input_dim: int, state_dim: int, dt_min: float = 0.001,
        dt_max: float = 0.1, lr: float | None = 1e-3,
    ):
        super().__init__()

        H = input_dim
        self.H = H
        N = state_dim
        self.N = N

        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        self.register('log_dt', log_dt, lr)

        C = torch.randn(H, N, dtype=torch.cfloat)
        self.C = nn.Parameter(c2r(C))

        log_A_real = torch.log(0.5 * torch.ones(H, N))
        A_imag = math.pi * repeat(torch.arange(N), 'n -> h n', h=H)
        self.register('log_A_real', log_A_real, lr)
        self.register('A_imag', A_imag, lr)

        self.D = nn.Parameter(torch.randn(H))

    def get_kernel(self, length: int):  # `length` is `L`
        dt = torch.exp(self.log_dt)  # (H)
        C = r2c(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        P = dtA.unsqueeze(-1) * torch.arange(
            length, device=A.device
        )  # (H N L)
        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(P)).real
        return K

    def register(self, name: str, tensor: Tensor, lr: float | None = None):
        if lr is None or lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {'weight_decay': 0.0}
            if lr is not None:
                optim['lr'] = lr
            setattr(getattr(self, name), '_optim', optim)

    def forward(self, u: Tensor, length: Tensor | None = None):
        # Input and output shape (B H L)
        assert u.dim() == 3
        B, H, L = u.size()
        assert H == self.H

        # length shape (L)
        if length is None:
            length = torch.empty(B).fill_(L)
        assert length.dim() == 1
        assert length.size(0) == B
        assert torch.all(length <= L)
        length = length.to(torch.long).cpu()

        l_s, i_s = length.sort(stable=True, descending=True)
        l_s, i_s = l_s.tolist(), i_s.tolist()
        prev_i = 0
        pair: list[tuple[int, list[int]]] = []
        for i in range(1, B):
            if l_s[i] == l_s[i - 1]:
                continue
            pair.append((l_s[prev_i], i_s[prev_i:i]))
            prev_i = i
        pair.append((l_s[prev_i], i_s[prev_i:]))

        kernel = self.get_kernel(L)  # (H L)
        out = torch.zeros_like(u)  # (B H L)

        for l, idxs in pair:
            _k = kernel[:, :l]
            _u = u[idxs, :, :l]

            k_f = torch.fft.rfft(_k, n=2*l)  # (H l)
            u_f = torch.fft.rfft(_u, n=2*l)  # (B H l)
            y = torch.fft.irfft(u_f * k_f, n=2*l)[..., :l]  # (B H l)
            y += _u * self.D.unsqueeze(-1)  # (B H l)

            out[idxs, :, :l] = y[...]

        return out
