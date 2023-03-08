from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

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
from src.dataset import (PeakReductionValueType, SignalTrainSingleFileDataset,
                         SwitchValueType, download_signal_train_dataset_to)
from src.loss import FilterType, LossType, forge_loss_function_from
from src.model import ActivationType, DRCModel
from src.utils import (clear_memory, current_utc_time, get_tensor_device,
                       set_random_seed_to)


@dataclass
class Parameter(RootConfig):
    random_seed: int = 42

    dataset_dir: Path = Path('./data/SignalTrain')
    input_file_path: Path = Path('./data/SignalTrain/Train/input_138_.wav')
    output_file_path: Path = Path(
        './data/SignalTrain/Train/target_138_LA2A_3c__0__0.wav'
    )
    switch_value: SwitchValueType = 0
    peak_reduction_value: PeakReductionValueType = 0
    data_segment_length: float = 1.0

    epoch: int = 25
    learning_rate: float = 1e-3
    s4_learning_rate: float = 1e-3
    batch_size: int = 64

    model_channel: int = 16
    model_s4_hidden_size: int = 16
    model_activation: ActivationType = 'GELU'
    model_depth: int = 2
    model_take_abs: bool = True
    model_take_db: bool = True
    model_take_amp: bool = True

    loss: LossType = 'MSE'
    loss_filter_type: FilterType = 'hp'
    loss_filter_coef: float = 0.85

    log_wandb: bool = True
    wandb_entity: str = DEFAULT_WANDB_ENTITY
    wandb_project_name: str = DEFAULT_WANDB_PROJECT_NAME

    save_checkpoint: bool = True
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_PATH


'''Script parameters.'''
param = Parameter.parse_args()

'''The preparatory work.'''
download_signal_train_dataset_to(param.dataset_dir)
set_random_seed_to(param.random_seed)
device = get_tensor_device(apple_silicon=False)
job_name = current_utc_time()
job_dir = (param.checkpoint_dir / job_name)

if param.save_checkpoint:
    job_dir.mkdir(511, True, True)
    param.to_json(job_dir / 'config.json')

'''Prepare the dataset.'''
dataset = SignalTrainSingleFileDataset(
    param.input_file_path,
    param.output_file_path,
    param.data_segment_length,
    param.switch_value,
    param.peak_reduction_value,
)
training_dataset, validation_dataset = random_split(dataset, [0.85, 0.15])
training_dataloader = DataLoader(
    training_dataset, param.batch_size,
    shuffle=True, collate_fn=dataset.collate_fn
)
validation_dataloader = DataLoader(
    validation_dataset, param.batch_size,
    shuffle=True, collate_fn=dataset.collate_fn
)

'''Prepare the model.'''
model = DRCModel(
    param.model_channel,
    param.model_s4_hidden_size,
    param.s4_learning_rate,
    param.model_depth,
    param.model_activation,
    param.model_take_db,
    param.model_take_abs,
    param.model_take_amp,
).to(device)

model_statistics = summary(model, (param.batch_size, dataset.segment_size))
if param.save_checkpoint:
    with open(job_dir / 'model-statistics.txt', 'w') as f:
        f.write(str(model_statistics))

'''Loss function'''
criterion = forge_loss_function_from(
    param.loss, param.loss_filter_type, param.loss_filter_coef
).to(device)
evaluation_criterions: dict[LossType, nn.Module] = {
    'MSE': forge_loss_function_from(
        'MSE', param.loss_filter_type, param.loss_filter_coef
    ).eval().to(device),
    'ESR+DC': forge_loss_function_from(
        'ESR+DC', param.loss_filter_type, param.loss_filter_coef
    ).eval().to(device),
    'Multi-STFT': forge_loss_function_from(
        'Multi-STFT', param.loss_filter_type, param.loss_filter_coef
    ).eval().to(device),
    'ESR+DC+Multi-STFT': forge_loss_function_from(
        'ESR+DC+Multi-STFT', param.loss_filter_type, param.loss_filter_coef
    ).eval().to(device)
}

'''Prepare the optimizer'''
optimizer = AdamW(model.parameters(), lr=param.learning_rate)

'''Weight and Bias'''
# TODO: finish logging.
if param.log_wandb:
    wandb.init(
        config=param.to_dict()
    )

'''Training loop'''
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

    for x, y in training_bar:
        x: Tensor = x.to(device)
        y: Tensor = y.to(device)

        optimizer.zero_grad()

        y_hat: Tensor = model(x)

        loss: Tensor = criterion(y_hat, y)

        loss.backward()
        optimizer.step()

        training_bar.set_postfix({
            'loss': loss.item(),
        })
        batch_training_loss += loss.item() / len(training_dataloader)

    batch_validation_loss: defaultdict[LossType, float] = defaultdict(float)
    model.eval()
    criterion.eval()
    validation_bar = tqdm(
        validation_dataloader,
        desc=f'Validating. {epoch = }',
        total=len(validation_dataloader),
    )

    with torch.no_grad():
        for x, y in validation_bar:
            x: Tensor = x.to(device)
            y: Tensor = y.to(device)

            y_hat: Tensor = model(x)

            for loss_type, evaluation_criterion in evaluation_criterions.items():
                this_loss: Tensor = evaluation_criterion(y_hat, y)
                batch_validation_loss[loss_type] += (
                    this_loss.item() / len(validation_dataloader)
                )

    if param.log_wandb:
        wandb.log({
            'Training Loss': batch_training_loss,
        } | batch_validation_loss)
    if param.save_checkpoint:
        torch.save(model.state_dict(), job_dir / f'model-epoch-{epoch}.pth')
