"""Main routine module.

Some routines are shared among all main scripts.
So, I extract them into this module.
"""
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn
from rootconfig import RootConfig
from torch.cuda import is_available as cuda_is_available
from torchinfo import summary as get_model_info_from

from .dataset import download_signal_train_dataset_to
from .utils import current_utc_time, set_random_seed_to


def do_preparatory_work(
    parameters: RootConfig,
    dataset_dir: Path, checkpoint_dir: Path, random_seed: int,
    should_save_checkpoint: bool,
):
    pprint(parameters.to_dict())

    download_signal_train_dataset_to(dataset_dir)
    set_random_seed_to(random_seed)
    if not cuda_is_available():
        raise RuntimeError(f'CUDA is not available. Aborting. ')
    device = torch.device('cuda')
    job_dir = checkpoint_dir / current_utc_time()

    if should_save_checkpoint:
        job_dir.mkdir(511, True, True)
        parameters.to_json(job_dir / 'config.json')

    return job_dir, device


def print_and_save_model_info(
    model: nn.Module, input_size: tuple[tuple[int, ...] | int, ...,],
    job_dir: Path, should_save_checkpoint: bool,
):
    model_info = get_model_info_from(model, input_size)
    if should_save_checkpoint:
        with open(job_dir / 'model-statistics.txt', 'wb') as f:
            f.write(str(model_info).encode())
