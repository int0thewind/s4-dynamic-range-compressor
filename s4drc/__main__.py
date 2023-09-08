from lightning.pytorch.cli import LightningCLI
import torch

from .src.dataset import SignalTrainDatasetModule
from .src.model import S4Model


def main():
    torch.set_float32_matmul_precision('high')
    LightningCLI(
        S4Model,
        SignalTrainDatasetModule,
        save_config_callback=None,  # Do not save config as we always provide configs externally
    )


if __name__ == '__main__':
    main()