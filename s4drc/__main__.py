from pytorch_lightning import Trainer
from pytorch_lightning.cli import LightningCLI

from .src.dataset import SignalTrainDatasetModule
from .src.model import S4Model


def main():
    LightningCLI(
        S4Model,
        SignalTrainDatasetModule,
        seed_everything_default=42,
    )


if __name__ == '__main__':
    main()