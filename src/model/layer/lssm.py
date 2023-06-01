"""Adapted from
"https://github.com/boris-kuz/differentiable_iir_filters/blob/master/linear_state_space_model.py".
"""

import torch
import torch.nn as nn
from torch import Tensor


class LSSMCell(nn.Module):
    def __init__(self, num_states: int):
        super(LSSMCell, self).__init__()

        self.num_states = num_states

        self.A = nn.Linear(
            num_states, num_states, bias=False
        )
        self.B = nn.Linear(1, num_states, bias=False)

        bound = 1.0 / (self.A.in_features)
        nn.init.uniform_(self.A.weight, -bound, bound)

    def forward(self, input: Tensor, in_states: Tensor):
        state_output = self.A(in_states) + self.B(input)
        return state_output


class LSSM(nn.Module):
    def __init__(self, num_states: int):
        super(LSSM, self).__init__()

        self.pre_gain = nn.Parameter(torch.FloatTensor([1.0]))
        self.num_states = num_states

        self.C = nn.Linear(num_states, 1, bias=False)
        self.D = nn.Linear(1, 1, bias=False)
        self.cell = LSSMCell(num_states)

    def forward(self, u: Tensor):
        # Input shape (B H=1 L). Transpose to (B L H)
        u = u.transpose(-1, -2)
        batch_size, sequence_length = u.size()

        hidden = torch.zeros(batch_size, self.num_states).to(u)
        states_sequence = torch.zeros(
            batch_size, sequence_length, self.num_states
        ).to(u)  # (B, L, H)

        # TODO: why we need pre_gain?
        # x = x * self.pre_gain
        for i in range(sequence_length - 1):
            hidden = self.cell(u[:, i, :], hidden)
            states_sequence[:, i+1, :] = hidden[:, :]

        ouput: Tensor = self.C(states_sequence) + self.D(u)

        ouput = ouput.transpose(-1, -2)
        return ouput, states_sequence
