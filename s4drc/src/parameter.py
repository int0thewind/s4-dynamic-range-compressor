from dataclasses import dataclass
from pathlib import Path

from rootconfig import RootConfig

from .loss import LossType
from .model import Activation, S4FixSideChainModelVersion


@dataclass
class FixTaskParameter(RootConfig):
    random_seed: int = 42

    dataset_dir: Path = Path('./data/SignalTrain')
    data_segment_length: float = 1.0  # no change

    epoch: int = 70  # no change
    learning_rate: float = 1e-3  # no change
    s4_learning_rate: float = 1e-3  # no change
    batch_size: int = 32  # no change

    model_version: S4FixSideChainModelVersion = 4  # do not report
    model_take_side_chain: bool = True
    model_inner_audio_channel: int = 32
    model_s4_hidden_size: int = 4
    model_depth: int = 4
    model_take_residual_connection: bool = False  # do not report
    model_convert_to_decibels: bool = True  # do not report
    model_take_tanh: bool = False  # new
    model_activation: Activation = 'GELU'

    loss: LossType = 'ESR+DC+Multi-STFT'  # no change, do not report
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

    epoch: int = 70  # no change
    learning_rate: float = 1e-3  # no change
    s4_learning_rate: float = 1e-3  # no change
    batch_size: int = 64  # no change
    # enable_learning_rate_scheduler: bool = True  # never used

    model_take_side_chain: bool = False
    model_inner_audio_channel: int = 32  # no change
    model_s4_hidden_size: int = 4  # no change
    model_depth: int = 4  # no change
    model_film_take_batchnorm: bool = True  # no change
    model_take_residual_connection: bool = True  # no change
    model_convert_to_decibels: bool = False  # do not report
    model_convert_to_amplitude: bool = False  # do not report
    model_take_tanh: bool = False
    model_take_parametered_tanh: bool = False
    model_activation: Activation = 'PReLU'  # no change

    loss: LossType = 'ESR+DC+Multi-STFT'  # do not report
    loss_filter_coef: float = 0.85  # no change

    log_wandb: bool = True
    wandb_entity: str = 'int0thewind'
    wandb_project_name: str = 'S4 Dynamic Range Compressor'

    save_checkpoint: bool = True
    checkpoint_dir: Path = Path('./experiment-result')
