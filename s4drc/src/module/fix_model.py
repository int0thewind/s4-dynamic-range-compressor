# from typing import Literal, get_args

# import torch.nn as nn
# from torch import Tensor

# from .activation import Activation, get_activation_type_from
# from .layer import DSSM, Amplitude, Decibel, Rearrange

# ModelVersion = Literal[0, 1, 2, 3, 4]


# class Block(nn.Module):
#     def __init__(
#         self,
#         model_version: ModelVersion,
#         inner_audio_channel: int,
#         s4_hidden_size: int,
#         s4_learning_rate: float | None,
#         take_residual_connection: bool,
#         activation: Activation,
#     ):
#         super().__init__()
#         Act = get_activation_type_from(activation)

#         layers: list[nn.Module] = []
#         if model_version == 0:
#             layers = [
#                 nn.Linear(inner_audio_channel, inner_audio_channel),
#                 Act(),
#             ]
#         elif model_version == 1:
#             layers = [
#                 Rearrange('B L H -> B H L'),
#                 DSSM(inner_audio_channel, s4_hidden_size,
#                      lr=s4_learning_rate),
#                 Rearrange('B H L -> B L H'),
#                 Act(),
#             ]
#         elif model_version == 2:
#             layers = [
#                 nn.Linear(inner_audio_channel, inner_audio_channel),
#                 Act(),
#                 Rearrange('B L H -> B H L'),
#                 DSSM(inner_audio_channel, s4_hidden_size,
#                      lr=s4_learning_rate),
#                 Rearrange('B H L -> B L H'),
#             ]
#         elif model_version == 3:
#             layers = [
#                 nn.Linear(inner_audio_channel, inner_audio_channel),
#                 Rearrange('B L H -> B H L'),
#                 DSSM(inner_audio_channel, s4_hidden_size,
#                      lr=s4_learning_rate),
#                 Rearrange('B H L -> B L H'),
#                 Act(),
#             ]
#         elif model_version == 4:
#             layers = [
#                 nn.Linear(inner_audio_channel, inner_audio_channel),
#                 Act(),
#                 Rearrange('B L H -> B H L'),
#                 DSSM(inner_audio_channel, s4_hidden_size,
#                      lr=s4_learning_rate),
#                 Rearrange('B H L -> B L H'),
#                 Act(),
#             ]

#         self.model = nn.Sequential(*layers)

#         self.residual_connection = nn.Sequential(
#             Rearrange('B L H -> B H L'),
#             nn.Conv1d(
#                 inner_audio_channel,
#                 inner_audio_channel,
#                 kernel_size=1,
#                 groups=inner_audio_channel,
#                 bias=False
#             ),
#             Rearrange('B H L -> B L H'),
#         ) if take_residual_connection else None

#     def forward(self, x: Tensor) -> Tensor:
#         out = self.model(x)
#         if self.residual_connection:
#             return out + self.residual_connection(x)
#         return out


# class S4FixModel(nn.Module):
#     model: nn.Module

#     def __init__(
#         self,
#         model_version: ModelVersion,
#         take_side_chain: bool,
#         inner_audio_channel: int,
#         s4_hidden_size: int,
#         s4_learning_rate: float | None,
#         model_depth: int,
#         take_residual_connection: bool,
#         convert_to_decibels: bool,
#         take_tanh: bool,
#         activation: Activation,
#     ):
#         if not model_version in get_args(ModelVersion):
#             raise ValueError(
#                 f'Unsupported model version. '
#                 f'Expect one of {get_args(ModelVersion)}, but got {model_version}.'
#             )
#         if inner_audio_channel < 1:
#             raise ValueError()
#         if s4_hidden_size < 1:
#             raise ValueError()
#         if model_depth < 0:
#             raise ValueError()

#         super().__init__()

#         layers: list[nn.Module] = []
#         if convert_to_decibels:
#             layers.append(Decibel())

#         layers.extend([
#             Rearrange('B L -> B L 1'),
#             nn.Linear(1, inner_audio_channel),
#         ])

#         for _ in range(model_depth):
#             layers.append(Block(
#                 model_version,
#                 inner_audio_channel,
#                 s4_hidden_size,
#                 s4_learning_rate,
#                 take_residual_connection,
#                 activation,
#             ))

#         layers.extend([
#             nn.Linear(inner_audio_channel, 1),
#             Rearrange('B L 1 -> B L')
#         ])

#         if convert_to_decibels:
#             layers.append(Amplitude())
#         if take_tanh:
#             layers.append(nn.Tanh())

#         self.model = nn.Sequential(*layers)
#         self.take_side_chain = take_side_chain

#     def forward(self, x: Tensor) -> Tensor:
#         out = self.model(x)
#         if self.take_side_chain:
#             return x * out
#         return out
