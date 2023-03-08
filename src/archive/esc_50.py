import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import torch
import torchaudio
from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram


class ESC50Dataset(Dataset):
    all_folds = frozenset(range(1, 6))
    download_link = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
    sample_rate = 44100

    root: Path
    metadata: pd.DataFrame
    mel_spectrogram: MelSpectrogram | None
    length: float | None

    def __init__(
        self, root: os.PathLike, download: bool = True, *,
        folds: Iterable[int] | None = None,
        mel_spectrogram: bool = False,
        length: float | None = None
    ):
        root = Path(root)
        if not root.exists():
            root.mkdir(exist_ok=True, parents=True)
        metadata_file = root / 'meta' / 'esc50.csv'

        if not metadata_file.exists() and download:
            _down = root / 'temp.zip'
            download_url_to_file(self.download_link, _down)
            with ZipFile(_down, 'r') as f:
                f.extractall()
            _down.unlink()
            del _down

            for f in (root / 'ESC-50-master').iterdir():
                if not f.name.startswith('.'):
                    shutil.move(f, root)
            (root / 'ESC-50-master').rmdir()

        if not metadata_file.exists():
            raise FileNotFoundError(
                f'Cannot find metadata file at `{metadata_file}`')

        metadata = pd.read_csv(metadata_file)
        if folds is not None:
            selected_folds = frozenset(folds)
            if not selected_folds <= self.all_folds:
                raise ValueError()
            for fold in self.all_folds - selected_folds:
                metadata = metadata[metadata['fold'] != fold]

        self.metadata = metadata
        self.root = root
        self.mel_spectrogram = MelSpectrogram(
            self.sample_rate, n_fft=2028,
            win_length=2028, hop_length=1014, n_mels=96
        ) if mel_spectrogram else None
        self.length = length

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        item = self.metadata.iloc[index]
        audio_file = self.root / 'audio' / item['filename']
        class_id: int = item['target']

        data: Tensor = torchaudio.load(audio_file)[0].flatten()
        if self.length is not None:
            data = data[:self.length * self.sample_rate]
        if self.mel_spectrogram:
            data = self.mel_spectrogram(data)

        return data, class_id

    @staticmethod
    def collate_fn(batch: list[tuple[Tensor, int]]):
        tensors = torch.stack([t for t, i in batch])
        class_ids = torch.tensor([i for t, i in batch])
        return tensors, class_ids
