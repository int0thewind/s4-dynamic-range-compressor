import os
import shutil
import tarfile
from pathlib import Path
from typing import Literal

import torch
import torchaudio
from einops import rearrange
from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset

__all__ = ['SwitchValueType', 'PeakReductionValueType',
           'download_signal_train_dataset_to', 'SignalTrainSingleFileDataset']

SIGNAL_TRAIN_DATASET_DOWNLOAD_LINK = 'https://zenodo.org/record/3824876/files/SignalTrain_LA2A_Dataset_1.1.tgz'

SwitchValueType = Literal[0, 1]
PeakReductionValueType = Literal[
    0, 5, 10, 15, 20, 25, 30,
    35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100
]


def download_signal_train_dataset_to(root: os.PathLike):
    if not isinstance(root, Path):
        root = Path(root)

    if (root / 'Train').exists():
        print('The SignalTrain dataset has been downloaded. Skipping ... ')
        return

    root.mkdir(511, True, True)

    _d = root / 'temp.tgz'
    download_url_to_file(SIGNAL_TRAIN_DATASET_DOWNLOAD_LINK, _d)
    with tarfile.open(_d, 'r') as tf:
        tf.extractall()
    _d.unlink()

    shutil.move(root / 'SignalTrain_LA2A_Dataset_1.1' / 'Train', root)
    shutil.move(root / 'SignalTrain_LA2A_Dataset_1.1' / 'Test', root)
    shutil.move(root / 'SignalTrain_LA2A_Dataset_1.1' / 'Val', root)
    (root / 'SignalTrain_LA2A_Dataset_1.1').unlink()


class SignalTrainSingleFileDataset(Dataset):
    switch: SwitchValueType
    peak_reduction: PeakReductionValueType
    input_data: Tensor
    output_data: Tensor
    segment_size: int
    sample_rate: int

    def __init__(
        self, input_file: os.PathLike, output_file: os.PathLike, segment_length: float,
        switch: SwitchValueType, peak_reduction: PeakReductionValueType,
    ):
        if not (0.0 < segment_length <= 10.0):
            raise ValueError(
                'The segment length must be a positive number smaller than 10.'
            )

        idat, sr = torchaudio.load(input_file)   # type: ignore
        odat, sr_out = torchaudio.load(output_file)   # type: ignore

        if idat.size() != odat.size():
            raise ValueError(
                'The input audio and the output audio does not have the same length.'
            )
        if idat.size(0) != 1:
            raise ValueError('The input audio is not mono.')
        if sr != sr_out:
            raise ValueError(
                'The input audio and the output audio does not have the same sample rate.'
            )

        segment_size = int(segment_length * sr)
        size, trill = divmod(idat.size(1), segment_size)
        if trill != 0:
            idat = idat[:, :-trill]
            odat = odat[:, :-trill]
        idat = rearrange(idat.squeeze(0), '(S L) -> S L', S=size)
        odat = rearrange(odat.squeeze(0), '(S L) -> S L', S=size)
        assert idat.size() == odat.size() == (
            size, segment_size), f'{idat.size() = }, {odat.size() = }'

        self.input_data = idat
        self.output_data = odat
        self.switch = switch
        self.peak_reduction = peak_reduction
        self.segment_size = segment_size
        self.sample_rate = sr

    def __len__(self):
        return self.input_data.size(0)

    def __getitem__(self, i: int):
        return (self.input_data[i, :], self.output_data[i, :])

    @staticmethod
    def collate_fn(batch: list[tuple[Tensor, Tensor]]):
        return (
            torch.stack([f for f, _ in batch]),
            torch.stack([s for _, s in batch]),
        )
