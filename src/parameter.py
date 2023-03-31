from dataclasses import dataclass
from pathlib import Path

from rootconfig import RootConfig

from .loss import LossType
from .model import Activation, DRCSideChainModelVersion


@dataclass
class FixTaskParameter(RootConfig):
    random_seed: int = 42

    dataset_dir: Path = Path('./data/SignalTrain')
    data_segment_length: float = 1.0

    epoch: int = 75
    learning_rate: float = 1e-3
    s4_learning_rate: float = 1e-3
    batch_size: int = 32

    model_version: DRCSideChainModelVersion = 3
    model_inner_audio_channel: int = 16
    model_s4_hidden_size: int = 16
    model_activation: Activation = 'GELU'
    model_depth: int = 2
    model_take_db: bool = False
    model_take_amp: bool = False

    loss: LossType = 'ESR+DC+Multi-STFT'
    loss_filter_coef: float = 0.85

    log_wandb: bool = True
    wandb_entity: str = 'int0thewind'
    wandb_project_name: str = 'S4 Dynamic Range Compressor'

    save_checkpoint: bool = True
    checkpoint_dir: Path = Path('./experiment-result')
