from lightning.pytorch.cli import LightningCLI
import torch

from .src.dataset import SignalTrainDatasetModule
from .src.model import S4Model


def main():
    torch.set_float32_matmul_precision('high')
    LightningCLI(
        S4Model,
        SignalTrainDatasetModule,
    )


if __name__ == '__main__':
    main()