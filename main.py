from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import get_args

import torch
import torch.nn as nn
import wandb
from rootconfig import RootConfig
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from tqdm import tqdm

from src.constant import *
from src.dataset import (DatasetType, FixDataset, ParameterDataset,
                         PeakReductionValue, SwitchValue,
                         download_signal_train_dataset_to)
from src.loss import FilterType, LossType, forge_loss_function_from
from src.model import Activation, S4LinearModelV1
from src.utils import (clear_memory, current_utc_time, get_tensor_device,
                       set_random_seed_to)

if __name__ != '__main__':
    raise RuntimeError(f'The main script cannot be imported by other module.')


@dataclass
class Parameter(RootConfig):
    random_seed: int = 42

    dataset_dir: Path = Path('./data/SignalTrain')
    dataset_type: DatasetType = 'fix'
    switch_value: SwitchValue = 1
    peak_reduction_value: PeakReductionValue = 100
    data_segment_length: float = 1.0

    epoch: int = 50
    learning_rate: float = 1e-3
    s4_learning_rate: float = 1e-3
    batch_size: int = 32

    model_channel: int = 16
    model_s4_hidden_size: int = 16
    model_activation: Activation = 'GELU'
    model_depth: int = 6
    model_take_abs: bool = False
    model_take_db: bool = False
    model_take_amp: bool = False

    loss: LossType = 'ESR+DC+Multi-STFT'
    loss_filter_type: FilterType = 'hp'
    loss_filter_coef: float = -0.85

    log_wandb: bool = True
    wandb_entity: str = DEFAULT_WANDB_ENTITY
    wandb_project_name: str = DEFAULT_WANDB_PROJECT_NAME

    save_checkpoint: bool = True
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_PATH

    keep_s4: bool = True


'''Script parameters.'''
param = Parameter.parse_args()
pprint(param.to_dict())

'''The preparatory work.'''
download_signal_train_dataset_to(param.dataset_dir)
set_random_seed_to(param.random_seed)
device = get_tensor_device(apple_silicon=False)
print(f'Device {device} detected.')
job_name = hex(hash(param))[2:]
job_dir = (param.checkpoint_dir / job_name)

if param.save_checkpoint:
    job_dir.mkdir(511, True, True)
    param.to_json(job_dir / 'config.json')

'''Weight and Bias'''
if param.log_wandb:
    wandb.init(
        name=job_name,
        project=param.wandb_project_name,
        entity=param.wandb_entity,
        config=param.to_dict(),
    )

'''Prepare the dataset.'''
training_dataset = FixDataset(
    param.dataset_dir, param.data_segment_length, 'train')
validation_dataset = FixDataset(
    param.dataset_dir, param.data_segment_length, 'val')
testing_dataset = FixDataset(
    param.dataset_dir, param.data_segment_length, 'test')
training_dataloader = DataLoader(
    training_dataset, param.batch_size,
    shuffle=True, collate_fn=training_dataset.collate_fn
)
validation_dataloader = DataLoader(
    validation_dataset, param.batch_size,
    shuffle=True, collate_fn=validation_dataset.collate_fn
)

'''Prepare the model.'''
model = S4LinearModelV1(
    param.model_channel,
    param.model_s4_hidden_size,
    param.s4_learning_rate,
    param.model_depth,
    param.model_activation,
    param.model_take_db,
    param.model_take_abs,
    param.model_take_amp,
    param.keep_s4
).to(device)

model_statistics = summary(model, (
    param.batch_size,
    int(param.data_segment_length * FixDataset.sample_rate)
))
if param.save_checkpoint:
    with open(job_dir / 'model-statistics.txt', 'w') as f:
        f.write(str(model_statistics))

'''Loss function'''
criterion = forge_loss_function_from(
    param.loss, param.loss_filter_type, param.loss_filter_coef
).to(device)

validation_criterions: dict[LossType, nn.Module] = {
    loss_type: forge_loss_function_from(
        loss_type, param.loss_filter_type, param.loss_filter_coef
    ).eval().to(device) for loss_type in get_args(LossType)
}

'''Prepare the optimizer'''
optimizer = AdamW(model.parameters(), lr=param.learning_rate)

'''Training loop'''
if param.log_wandb:
    wandb.watch(model, log='all')
for epoch in range(param.epoch):
    clear_memory()

    batch_training_loss = 0.0
    model.train()
    criterion.train()
    training_bar = tqdm(
        training_dataloader,
        desc=f'Training. {epoch = }',
        total=len(training_dataloader),
    )

    for x, y, _ in training_bar:
        x: Tensor
        y: Tensor
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_hat: Tensor = model(x)
        loss: Tensor = criterion(y_hat, y)

        loss.backward()
        optimizer.step()

        training_bar.set_postfix({
            'loss': loss.item(),
        })
        batch_training_loss += loss.item() / len(training_dataloader)

    batch_validation_loss: defaultdict[str, float] = defaultdict(float)
    model.eval()
    criterion.eval()
    validation_bar = tqdm(
        validation_dataloader,
        desc=f'Validating. {epoch = }',
        total=len(validation_dataloader),
    )

    with torch.no_grad():
        for x, y, _ in validation_bar:
            x: Tensor
            y: Tensor
            x = x.to(device)
            y = y.to(device)

            y_hat: Tensor = model(x)

            for validation_loss, validation_criterion in validation_criterions.items():
                this_loss: Tensor = validation_criterion(y_hat, y)
                batch_validation_loss[f'Validation Loss: {validation_loss}'] += (
                    this_loss.item() / len(validation_dataloader)
                )

    if param.log_wandb:
        wandb.log({
            f'Training Loss: {param.loss}': batch_training_loss,
        } | batch_validation_loss)
    if param.save_checkpoint:
        torch.save(model.state_dict(), job_dir / f'model-epoch-{epoch}.pth')

if param.log_wandb:
    wandb.finish()
