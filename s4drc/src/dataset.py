import shutil
import tarfile
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Generic, TypedDict, TypeVar

import lightning.pytorch as pl
import torch
import torchaudio
from einops import rearrange
from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

T = TypeVar('T')


class SequenceDataset(Dataset, Generic[T]):
    def __init__(self, entries: Sequence[T], transform: Callable[[T], T] | None = None) -> None:
        super().__init__()
        self.entries = entries
        self.transform = transform

    def __getitem__(self, index: int):
        ret = self.entries[index]
        if self.transform:
            ret = self.transform(ret)
        return ret
    
    def __len__(self):
        return len(self.entries)
    

class SignalTrainDatasetModuleParams(TypedDict):
    root: str
    batch_size: int
    training_segment_length: int
    validation_segment_length: int
    testing_segment_length: int
    

class SignalTrainDatasetModule(pl.LightningDataModule):
    sample_rate = 44_100

    hparams: SignalTrainDatasetModuleParams

    def __init__(
        self,
        root: str = './data/SignalTrain',
        batch_size: int = 32,
        training_segment_length: int = 2 ** 16,
        validation_segment_length: int = 2 ** 18,
        testing_segment_length: int = 2 ** 23,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        link = 'https://zenodo.org/record/3824876/files/SignalTrain_LA2A_Dataset_1.1.tgz'
        root = Path(self.hparams['root'])

        if (root / 'Train').exists():
            print('The SignalTrain dataset has been downloaded. Skipping ... ')
            return

        root.mkdir(511, True, True)

        d = root / 'temp.tgz'
        download_url_to_file(link, d)
        with tarfile.open(d, 'r') as tf:
            tf.extractall()
        d.unlink()
        shutil.move(root / 'SignalTrain_LA2A_Dataset_1.1' / 'Train', root)
        shutil.move(root / 'SignalTrain_LA2A_Dataset_1.1' / 'Test', root)
        shutil.move(root / 'SignalTrain_LA2A_Dataset_1.1' / 'Val', root)
        (root / 'SignalTrain_LA2A_Dataset_1.1').unlink()
    
    def train_dataloader(self):
        entries = self._read_data(
            Path(self.hparams['root']) / 'Train',
            self.hparams['training_segment_length'],
        )
        return DataLoader(
            entries,
            self.hparams['batch_size'],
            num_workers=8,
            shuffle=True,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self):
        entries = self._read_data(
            Path(self.hparams['root']) / 'Val',
            self.hparams['validation_segment_length'],
        )
        return DataLoader(
            entries,
            self.hparams['batch_size'],
            num_workers=8,
            shuffle=False,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self):
        entries = self._read_data(
            Path(self.hparams['root']) / 'Test',
            self.hparams['testing_segment_length'],
        )
        return DataLoader(
            entries,
            self.hparams['batch_size'],
            num_workers=8,
            shuffle=False,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    @staticmethod
    def _collate_fn(batch: list[tuple[Tensor, Tensor, Tensor]]):
        return (
            torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
        )
    
    @staticmethod
    def _data_augmentation(entry: tuple[Tensor, Tensor, Tensor]):
        x, y, cond = entry
        if torch.rand([1]).item() < 0.5:
            x *= -1
            y *= -1
        return x, y, cond

    @classmethod
    def _slice_audio(cls, file: Path, segment_length: int) -> list[Tensor]:
        load_result: tuple[Tensor, int] = torchaudio.load(file)  # type: ignore
        dat, sr = load_result
        assert sr == cls.sample_rate
        dat.squeeze_(0)
        if dat.dim() != 1:
            raise ValueError(f'{file} is not a mono audio.')

        size, trill = divmod(dat.size(0), segment_length)
        if trill != 0:
            dat = dat[:-trill]
        dat = rearrange(dat, '(S L) -> S L', S=size)

        return [dat[i] for i in range(dat.size(0))]
    
    def _read_data(self, data_path: Path, segment_length: int):
        entries: list[tuple[Tensor, Tensor, Tensor]] = []
        all_files = sorted(data_path.glob('*.wav'))
        for file in tqdm(all_files, desc=f'Loading dataset from {data_path}.'):
            if file.name.startswith('input'):
                continue
            file_id = file.name[7:10]
            switch_value, peak_reduction_value = map(
                int, file.stem.split('__')[1:])
            input_file = file.with_name(f'input_{file_id}_.wav')

            input_datas = self._slice_audio(input_file, segment_length)
            output_datas = self._slice_audio(file, segment_length)
            for input_data, output_data in zip(input_datas, output_datas):
                assert input_data.size() == output_data.size()
                entries.append((
                    input_data,
                    output_data,
                    torch.tensor([
                        switch_value, peak_reduction_value
                    ], dtype=torch.float32)
                ))
        return SequenceDataset(entries, self._data_augmentation)
