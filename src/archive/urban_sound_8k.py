import os
import shutil
import tarfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torchaudio
from torch import Tensor
from torch.hub import download_url_to_file
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset


class UrbanSound8KDataset(Dataset):
    all_folds = frozenset(range(1, 11))
    download_link = 'https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1'
    sample_rate = 44100

    root: Path
    metadata: pd.DataFrame

    def __init__(
        self, root: os.PathLike, download: bool = True, *,
        folds: Iterable[int] | None = None,
    ):
        root = Path(root)
        metadata_file = root / 'metadata' / 'UrbanSound8K.csv'

        if not metadata_file.exists() and download:
            _down = root / 'temp.tar.gz'
            download_url_to_file(self.download_link, _down)
            with tarfile.open(_down, 'r') as f:
                f.extractall()
            _down.unlink()
            del _down

            for f in (root / 'UrbanSound8K').iterdir():
                if not f.name.startswith('.'):
                    shutil.move(f, root)
            (root / 'UrbanSound8K').rmdir()

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

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        item = self.metadata.iloc[index]
        audio_file = self.root / 'audio' / 'fold' + \
            item['fold'] / item['slice_file_name']
        class_id: int = item['classID']

        d: Tensor = torchaudio.load(audio_file)[0]
        start = int(item['start'] * self.sample_rate)
        end = int(item['end'] * self.sample_rate)
        data = d[..., start:end].mean(dim=0)

        return data, class_id

    @staticmethod
    def collate_fn(batch: list[tuple[Tensor, int]]):
        # TODO: how to use packed_sequence with DSSM?
        batch_sort = sorted(batch, key=lambda b: b[0].size(-1), reverse=True)
        class_ids = [i for t, i in batch_sort]
        packed_sequence = pack_sequence([t for t, i in batch_sort])

        return packed_sequence, class_ids
