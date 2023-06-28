from dataclasses import dataclass
from pathlib import Path

from rootconfig import RootConfig

from .loss import LossType
from .model import Activation, S4FixSideChainModelVersion


@dataclass
class FixTaskParameter(RootConfig):
    random_seed: int = 42

    dataset_dir: Path = Path('./data/SignalTrain')
    data_segment_length: float = 1.5

    epoch: int = 70
    learning_rate: float = 1e-3
    s4_learning_rate: float = 1e-3
    batch_size: int = 32

    model_version: S4FixSideChainModelVersion = 4
    model_take_side_chain: bool = True  # new
    model_inner_audio_channel: int = 32
    model_s4_hidden_size: int = 4
    model_depth: int = 4
    model_take_residual_connection: bool = False  # new
    model_convert_to_decibels: bool = True
    model_take_tanh: bool = False  # new
    model_activation: Activation = 'GELU'  # PReLU is new

    loss: LossType = 'ESR+DC+Multi-STFT'
    loss_filter_coef: float = 0.85

    log_wandb: bool = True
    wandb_entity: str = 'int0thewind'
    wandb_project_name: str = 'S4 Dynamic Range Compressor Fixed'

    save_checkpoint: bool = True
    checkpoint_dir: Path = Path('./experiment-result')


@dataclass
class ConditionalTaskParameter(RootConfig):
    random_seed: int = 42

    dataset_dir: Path = Path('./data/SignalTrain')
    data_segment_length: float = 1.0

    epoch: int = 70
    learning_rate: float = 1e-3
    s4_learning_rate: float = 1e-3
    batch_size: int = 64
    enable_learning_rate_scheduler: bool = True

    model_take_side_chain: bool = False
    model_inner_audio_channel: int = 32
    model_s4_hidden_size: int = 4
    model_depth: int = 4
    model_film_take_batchnorm: bool = True
    model_take_residual_connection: bool = True
    model_convert_to_decibels: bool = False
    model_take_tanh: bool = False
    model_tanh_parameter: int = 0  # 0 means no parameter. -1 means
    model_activation: Activation = 'PReLU'

    loss: LossType = 'ESR+DC+Multi-STFT'
    loss_filter_coef: float = 0.85

    log_wandb: bool = True
    wandb_entity: str = 'int0thewind'
    wandb_project_name: str = 'S4 Dynamic Range Compressor'

    save_checkpoint: bool = True
    checkpoint_dir: Path = Path('./experiment-result')
