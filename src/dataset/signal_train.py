import os
import shutil
import tarfile
from pathlib import Path
from typing import Literal

import torchaudio
from torch.hub import download_url_to_file
from torch.utils.data import Dataset

Partition = Literal['train', 'test', 'validation']


def download_signal_train_dataset(root: os.PathLike):
    if not isinstance(root, Path):
        root = Path(root)
    if (root / 'Train').exists():
        return

    _d = root / 'temp.tgz'
    download_url_to_file(
        'https://zenodo.org/record/3824876/files/'
        'SignalTrain_LA2A_Dataset_1.1.tgz', _d,
    )
    with tarfile.open(_d, 'r') as tf:
        tf.extractall()
    _d.unlink()

    shutil.move(root / 'SignalTrain_LA2A_Dataset_1.1' / 'Train', root)
    shutil.move(root / 'SignalTrain_LA2A_Dataset_1.1' / 'Test', root)
    shutil.move(root / 'SignalTrain_LA2A_Dataset_1.1' / 'Val', root)
    (root / 'SignalTrain_LA2A_Dataset_1.1').unlink()


class SingleFileDataset(Dataset):
    def __init__(self, input_file: os.PathLike, output_file: os.PathLike, ):
        pass
