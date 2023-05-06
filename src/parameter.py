from dataclasses import dataclass
from pathlib import Path

from rootconfig import RootConfig

from .loss import LossType
from .model import Activation, S4FixSideChainModelVersion


@dataclass
class FixTaskSideChainParameter(RootConfig):
    random_seed: int = 42

    dataset_dir: Path = Path('./data/SignalTrain')
    data_segment_length: float = 1.0

    epoch: int = 60
    learning_rate: float = 1e-3
    s4_learning_rate: float = 1e-3
    batch_size: int = 32

    model_version: S4FixSideChainModelVersion = 4
    model_inner_audio_channel: int = 16
    model_s4_hidden_size: int = 16
    model_activation: Activation = 'GELU'
    model_depth: int = 4
    model_convert_to_decibels: bool = False

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

    epoch: int = 60
    learning_rate: float = 1e-3
    s4_learning_rate: float = 1e-3
    batch_size: int = 32
    enable_learning_rate_scheduler: bool = True

    # New in conditional task
    model_take_side_chain: bool = False

    model_inner_audio_channel: int = 32
    model_s4_hidden_size: int = 4
    model_depth: int = 4

    # New in conditional task
    model_film_take_batchnorm: bool = False

    # New in conditional task
    model_take_residual_connection: bool = True

    model_convert_to_decibels: bool = False

    # New in conditional task
    model_take_tanh: bool = True

    model_activation: Activation = 'PReLU'

    loss: LossType = 'ESR+DC+Multi-STFT'
    loss_filter_coef: float = 0.85

    log_wandb: bool = True
    wandb_entity: str = 'int0thewind'
    wandb_project_name: str = 'S4 Dynamic Range Compressor'

    save_checkpoint: bool = True
    checkpoint_dir: Path = Path('./experiment-result')
