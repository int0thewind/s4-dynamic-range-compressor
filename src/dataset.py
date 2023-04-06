import os
from abc import ABC
from pathlib import Path
from typing import Literal, get_args
from zipfile import ZipFile

import torch
import torchaudio
from einops import rearrange
from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset

__all__ = ['SwitchValue', 'PeakReductionValue',
           'download_signal_train_dataset_to', 'FixDataset']

SwitchValue = Literal[0, 1]
PeakReductionValue = Literal[
    0, 5, 10, 15, 20, 25, 30,
    35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100
]
DatasetFolder = Literal['Train', 'Test', 'Val']
Partition = Literal['train', 'test', 'val']

param_dict: dict[
    tuple[SwitchValue, PeakReductionValue],
    list[tuple[DatasetFolder, str, str]]
] = {
    (0, 0): [('Train',
              'input_138_.wav',
              'target_138_LA2A_3c__0__0.wav'),
             ('Train',
              'input_222_.wav',
              'target_222_LA2A_2c__0__0.wav')],
    (0, 5): [('Train',
              'input_139_.wav',
              'target_139_LA2A_3c__0__5.wav'),
             ('Val',
              'input_223_.wav',
              'target_223_LA2A_2c__0__5.wav')],
    (0, 10): [('Val',
               'input_140_.wav',
               'target_140_LA2A_3c__0__10.wav'),
              ('Val',
               'input_224_.wav',
               'target_224_LA2A_2c__0__10.wav')],
    (0, 15): [('Train',
               'input_225_.wav',
               'target_225_LA2A_2c__0__15.wav'),
              ('Train',
               'input_141_.wav',
               'target_141_LA2A_3c__0__15.wav')],
    (0, 20): [('Train',
               'input_142_.wav',
               'target_142_LA2A_3c__0__20.wav'),
              ('Train',
               'input_226_.wav',
               'target_226_LA2A_2c__0__20.wav')],
    (0, 25): [('Train',
               'input_143_.wav',
               'target_143_LA2A_3c__0__25.wav'),
              ('Val',
               'input_227_.wav',
               'target_227_LA2A_2c__0__25.wav')],
    (0, 30): [('Train',
               'input_144_.wav',
               'target_144_LA2A_3c__0__30.wav'),
              ('Train',
               'input_228_.wav',
               'target_228_LA2A_2c__0__30.wav')],
    (0, 35): [('Train',
               'input_229_.wav',
               'target_229_LA2A_2c__0__35.wav'),
              ('Train',
               'input_145_.wav',
               'target_145_LA2A_3c__0__35.wav')],
    (0, 40): [('Train',
               'input_146_.wav',
               'target_146_LA2A_3c__0__40.wav'),
              ('Train',
               'input_230_.wav',
               'target_230_LA2A_2c__0__40.wav')],
    (0, 45): [('Train',
               'input_147_.wav',
               'target_147_LA2A_3c__0__45.wav'),
              ('Val',
               'input_231_.wav',
               'target_231_LA2A_2c__0__45.wav')],
    (0, 50): [('Val',
               'input_232_.wav',
               'target_232_LA2A_2c__0__50.wav'),
              ('Val',
               'input_148_.wav',
               'target_148_LA2A_3c__0__50.wav')],
    (0, 55): [('Train',
               'input_149_.wav',
               'target_149_LA2A_3c__0__55.wav'),
              ('Train',
               'input_233_.wav',
               'target_233_LA2A_2c__0__55.wav')],
    (0, 60): [('Train',
               'input_150_.wav',
               'target_150_LA2A_3c__0__60.wav'),
              ('Train',
               'input_234_.wav',
               'target_234_LA2A_2c__0__60.wav')],
    (0, 65): [('Test',
               'input_235_.wav',
               'target_235_LA2A_2c__0__65.wav'),
              ('Train',
               'input_151_.wav',
               'target_151_LA2A_3c__0__65.wav')],
    (0, 70): [('Train',
               'input_152_.wav',
               'target_152_LA2A_3c__0__70.wav')],
    (0, 75): [('Train',
               'input_153_.wav',
               'target_153_LA2A_3c__0__75.wav'),
              ('Train',
               'input_237_.wav',
               'target_237_LA2A_2c__0__75.wav')],
    (0, 80): [('Train',
               'input_154_.wav',
               'target_154_LA2A_3c__0__80.wav'),
              ('Train',
               'input_238_.wav',
               'target_238_LA2A_2c__0__80.wav')],
    (0, 85): [('Train',
               'input_155_.wav',
               'target_155_LA2A_3c__0__85.wav'),
              ('Val',
               'input_239_.wav',
               'target_239_LA2A_2c__0__85.wav')],
    (0, 90): [('Train',
               'input_240_.wav',
               'target_240_LA2A_2c__0__90.wav'),
              ('Train',
               'input_156_.wav',
               'target_156_LA2A_3c__0__90.wav')],
    (0, 95): [('Train',
               'input_241_.wav',
               'target_241_LA2A_2c__0__95.wav'),
              ('Train',
               'input_157_.wav',
               'target_157_LA2A_3c__0__95.wav')],
    (0, 100): [('Train',
                'input_242_.wav',
                'target_242_LA2A_2c__0__100.wav'),
               ('Train',
                'input_158_.wav',
                'target_158_LA2A_3c__0__100.wav')],
    (1, 0): [('Train',
              'input_243_.wav',
              'target_243_LA2A_2c__1__0.wav'),
             ('Train',
              'input_159_.wav',
              'target_159_LA2A_3c__1__0.wav')],
    (1, 5): [('Train',
              'input_160_.wav',
              'target_160_LA2A_3c__1__5.wav'),
             ('Train',
              'input_244_.wav',
              'target_244_LA2A_2c__1__5.wav')],
    (1, 10): [('Val',
               'input_245_.wav',
               'target_245_LA2A_2c__1__10.wav'),
              ('Val',
               'input_161_.wav',
               'target_161_LA2A_3c__1__10.wav')],
    (1, 15): [('Train',
               'input_246_.wav',
               'target_246_LA2A_2c__1__15.wav'),
              ('Train',
               'input_162_.wav',
               'target_162_LA2A_3c__1__15.wav')],
    (1, 20): [('Train',
               'input_247_.wav',
               'target_247_LA2A_2c__1__20.wav'),
              ('Train',
               'input_163_.wav',
               'target_163_LA2A_3c__1__20.wav')],
    (1, 25): [('Train',
               'input_164_.wav',
               'target_164_LA2A_3c__1__25.wav'),
              ('Val',
               'input_248_.wav',
               'target_248_LA2A_2c__1__25.wav')],
    (1, 30): [('Train',
               'input_249_.wav',
               'target_249_LA2A_2c__1__30.wav'),
              ('Train',
               'input_165_.wav',
               'target_165_LA2A_3c__1__30.wav')],
    (1, 35): [('Train',
               'input_250_.wav',
               'target_250_LA2A_2c__1__35.wav'),
              ('Train',
               'input_166_.wav',
               'target_166_LA2A_3c__1__35.wav')],
    (1, 40): [('Train',
               'input_167_.wav',
               'target_167_LA2A_3c__1__40.wav'),
              ('Train',
               'input_251_.wav',
               'target_251_LA2A_2c__1__40.wav')],
    (1, 45): [('Train',
               'input_168_.wav',
               'target_168_LA2A_3c__1__45.wav'),
              ('Val',
               'input_252_.wav',
               'target_252_LA2A_2c__1__45.wav')],
    (1, 50): [('Train',
               'input_169_.wav',
               'target_169_LA2A_3c__1__50.wav'),
              ('Train',
               'input_253_.wav',
               'target_253_LA2A_2c__1__50.wav')],
    (1, 55): [('Train',
               'input_254_.wav',
               'target_254_LA2A_2c__1__55.wav'),
              ('Train',
               'input_170_.wav',
               'target_170_LA2A_3c__1__55.wav')],
    (1, 60): [('Train',
               'input_255_.wav',
               'target_255_LA2A_2c__1__60.wav'),
              ('Train',
               'input_171_.wav',
               'target_171_LA2A_3c__1__60.wav')],
    (1, 65): [('Test',
               'input_256_.wav',
               'target_256_LA2A_2c__1__65.wav'),
              ('Train',
               'input_172_.wav',
               'target_172_LA2A_3c__1__65.wav')],
    (1, 70): [('Val',
               'input_173_.wav',
               'target_173_LA2A_3c__1__70.wav'),
              ('Val',
               'input_257_.wav',
               'target_257_LA2A_2c__1__70.wav')],
    (1, 75): [('Train',
               'input_258_.wav',
               'target_258_LA2A_2c__1__75.wav'),
              ('Train',
               'input_174_.wav',
               'target_174_LA2A_3c__1__75.wav')],
    (1, 80): [('Test',
               'input_259_.wav',
               'target_259_LA2A_2c__1__80.wav'),
              ('Train',
               'input_175_.wav',
               'target_175_LA2A_3c__1__80.wav')],
    (1, 85): [('Train',
               'input_176_.wav',
               'target_176_LA2A_3c__1__85.wav'),
              ('Val',
               'input_260_.wav',
               'target_260_LA2A_2c__1__85.wav')],
    (1, 90): [('Train',
               'input_261_.wav',
               'target_261_LA2A_2c__1__90.wav'),
              ('Train',
               'input_177_.wav',
               'target_177_LA2A_3c__1__90.wav')],
    (1, 95): [('Train',
               'input_262_.wav',
               'target_262_LA2A_2c__1__95.wav'),
              ('Train',
               'input_178_.wav',
               'target_178_LA2A_3c__1__95.wav')],
    (1, 100): [('Train',
                'input_179_.wav',
                'target_179_LA2A_3c__1__100.wav'),
               ('Train',
                'input_221_.wav',
                'target_221_LA2A_3c__1__100.wav'),
               ('Train',
                'input_263_.wav',
                'target_263_LA2A_2c__1__100.wav')]}


def download_signal_train_dataset_to(root: os.PathLike):
    #     if not isinstance(root, Path):
    #         root = Path(root)

    #     if (root / 'Train').exists():
    #         print('The SignalTrain dataset has been downloaded. Skipping ... ')
    #         return

    #     root.mkdir(511, True, True)

    #     d = root / 'temp.tgz'
    #     download_url_to_file(
    #         'https://zenodo.org/record/3824876/files/SignalTrain_LA2A_Dataset_1.1.tgz', d
    #     )
    #     with tarfile.open(d, 'r') as tf:
    #         tf.extractall()
    #     d.unlink()

    #     shutil.move(root / 'SignalTrain_LA2A_Dataset_1.1' / 'Train', root)
    #     shutil.move(root / 'SignalTrain_LA2A_Dataset_1.1' / 'Test', root)
    #     shutil.move(root / 'SignalTrain_LA2A_Dataset_1.1' / 'Val', root)
    #     (root / 'SignalTrain_LA2A_Dataset_1.1').unlink()

    link = 'https://cmu.box.com/shared/static/tc9pxbh6wax37ld25vf33w5ng1nnfsbq.zip'

    if not isinstance(root, Path):
        root = Path(root)

    if (root / 'Train').exists():
        print('The SignalTrain dataset has been downloaded. Skipping ... ')
        return

    root.mkdir(511, True, True)

    d = root / 'temp.zip'
    download_url_to_file(link, d)
    with ZipFile(d, 'r') as zf:
        zf.extractall(root)
    d.unlink()


class AbstractSignalTrainDataset(ABC, Dataset):
    sample_rate = 44100

    @staticmethod
    def collate_fn(batch: list[tuple[Tensor, Tensor, Tensor]]):
        return (
            torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
        )

    @staticmethod
    def slice_audio(file: Path, segment_length: float) -> list[Tensor]:
        load_result: tuple[Tensor, int] = torchaudio.load(file)  # type: ignore
        dat, sr = load_result
        assert sr == AbstractSignalTrainDataset.sample_rate
        dat.squeeze_(0)
        if dat.dim() != 1:
            raise ValueError(f'{file} is not a mono audio.')

        size, trill = divmod(dat.size(0), int(segment_length * sr))
        if trill != 0:
            dat = dat[:-trill]
        dat = rearrange(dat, '(S L) -> S L', S=size)

        return [dat[i] for i in range(dat.size(0))]


class FixDataset(AbstractSignalTrainDataset):
    train_input_file = Path('Train') / 'input_179_.wav'
    test_input_file = Path('Train') / 'input_221_.wav'
    val_input_file = Path('Train') / 'input_263_.wav'
    train_output_file = Path('Train') / 'target_179_LA2A_3c__1__100.wav'
    test_output_file = Path('Train') / 'target_221_LA2A_3c__1__100.wav'
    val_output_file = Path('Train') / 'target_263_LA2A_2c__1__100.wav'

    input_data: list[Tensor]
    output_data: list[Tensor]

    def __init__(self, dataset_root: os.PathLike, partition: Partition, segment_length: float):
        if segment_length is not None and segment_length <= 0.0:
            raise ValueError(
                'The segment length must be a positive number smaller than 10.')

        if not isinstance(dataset_root, Path):
            dataset_root = Path(dataset_root)

        super().__init__()

        if partition == 'train':
            input_file = dataset_root / self.train_input_file
            output_file = dataset_root / self.train_output_file
        elif partition == 'test':
            input_file = dataset_root / self.test_input_file
            output_file = dataset_root / self.test_output_file
        else:
            input_file = dataset_root / self.val_input_file
            output_file = dataset_root / self.val_output_file

        self.input_data = self.slice_audio(input_file, segment_length)
        self.output_data = self.slice_audio(output_file, segment_length)
        assert len(self.input_data) == len(self.output_data)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, i: int):
        assert self.input_data[i].size() == self.output_data[i].size()
        return (
            self.input_data[i],
            self.output_data[i],
            torch.tensor([1, 100]),
        )


class SignalTrainDataset(AbstractSignalTrainDataset):
    entries: list[tuple[Tensor, Tensor, Tensor]]

    def __init__(self, dataset_root: os.PathLike, partition: Partition, segment_length: float):
        if segment_length is not None and segment_length <= 0.0:
            raise ValueError(
                'The segment length must be a positive number smaller than 10.')

        if not isinstance(dataset_root, Path):
            dataset_root = Path(dataset_root)

        super().__init__()

        if partition == 'train':
            data_path = dataset_root / 'Train'
        elif partition == 'test':
            data_path = dataset_root / 'Test'
        else:
            data_path = dataset_root / 'Val'

        self.entries = []

        for file in data_path.glob('*.wav'):
            if file.name.startswith('input'):
                continue
            assert file.name.startswith('target')
            print(f'Processing `{file}` ...')
            file_id = file.name[7:10]
            assert file_id.isnumeric()
            switch_value, peak_reduction_value = map(
                int, file.stem.split('__')[1:])
            assert switch_value in get_args(SwitchValue)
            assert peak_reduction_value in get_args(PeakReductionValue)
            input_file = file.with_name(f'input_{file_id}_.wav')
            assert input_file.is_file()

            input_datas = self.slice_audio(input_file, segment_length)
            output_datas = self.slice_audio(file, segment_length)
            for input_data, output_data in zip(input_datas, output_datas):
                assert input_data.size() == output_data.size()
                self.entries.append((
                    input_data,
                    output_data,
                    torch.tensor([switch_value, peak_reduction_value])
                ))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i: int):
        return self.entries[i]


"""
class ParameterDataset(AbstractSignalTrainDataset):
    switch_value: SwitchValue
    peak_reduction_value: PeakReductionValue

    input_data: Tensor
    output_data: Tensor

    def __init__(
        self, dataset_root: os.PathLike, segment_length: float,
        switch: SwitchValue, peak_reduction: PeakReductionValue,
    ):
        if not (0.0 < segment_length <= 10.0):
            raise ValueError(
                'The segment length must be a positive number smaller than 10.'
            )

        if not isinstance(dataset_root, Path):
            dataset_root = Path(dataset_root)

        super().__init__()

        segment_size = int(segment_length * self.sample_rate)
        input_datas = []
        output_datas = []

        for parent_folder_name, input_file_name, output_file_name in param_dict[(switch, peak_reduction)]:
            input_file = dataset_root / parent_folder_name / input_file_name
            output_file = dataset_root / parent_folder_name / output_file_name

            idat: Tensor = self.slice_audio(input_file, segment_size)
            odat: Tensor = self.slice_audio(output_file, segment_size)

            if idat.size() != odat.size():
                raise ValueError(
                    'The input audio and the output audio does not have the same length.'
                )

            input_datas.append(idat)
            output_datas.append(odat)

        self.input_data = torch.concat(input_datas, dim=0)
        self.output_data = torch.concat(output_datas, dim=0)
        assert self.input_data.size() == self.output_data.size()

        self.switch_value = switch
        self.peak_reduction_value = peak_reduction

    def __len__(self):
        return self.input_data.size(0)

    def __getitem__(self, i: int):
        return (
            self.input_data[i, :],
            self.output_data[i, :],
            torch.tensor([self.switch_value, self.peak_reduction_value]),
        )
"""


# class AllDataset(AbstractSignalTrainDataset):
#     # TODO: finish this dataset for future hyper-conditioning support
#     pass
