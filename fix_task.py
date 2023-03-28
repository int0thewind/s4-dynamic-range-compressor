from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import get_args

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from matplotlib.figure import Figure
from rootconfig import RootConfig
from torch import Tensor
from torch.cuda import is_available as cuda_is_available
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from torchinfo import summary as get_model_info_from
from tqdm import tqdm

from src.augmentation import invert_phase
from src.dataset import FixDataset, download_signal_train_dataset_to
from src.evaluation import (evaluate_rms_difference,
                            evaluate_waveform_difference)
from src.loss import LossType, forge_loss_function_from
from src.model import Activation, DRCSideChainModelVersion, forge_drc_model_by
from src.utils import clear_memory, current_utc_time, set_random_seed_to

if __name__ != '__main__':
    raise RuntimeError(f'The main script cannot be imported by other module.')


@dataclass
class Parameter(RootConfig):
    random_seed: int = 42

    dataset_dir: Path = Path('./data/SignalTrain')
    data_segment_length: float = 1.0

    epoch: int = 50
    learning_rate: float = 1e-3
    s4_learning_rate: float = 1e-3
    batch_size: int = 32

    model_version: DRCSideChainModelVersion = 1
    model_inner_audio_channel: int = 16
    model_s4_hidden_size: int = 16
    model_activation: Activation = 'GELU'
    model_depth: int = 6
    model_take_db: bool = False
    model_take_abs: bool = False
    model_take_amp: bool = False

    loss: LossType = 'ESR+DC+Multi-STFT'
    loss_filter_coef: float = 0.0

    log_wandb: bool = True
    wandb_entity: str = 'int0thewind'
    wandb_project_name: str = 'S4 Dynamic Range Compressor'

    save_checkpoint: bool = True
    checkpoint_dir: Path = Path('./experiment-result')


'''Script parameters.'''
param = Parameter.parse_args()
pprint(param.to_dict())

'''The preparatory work.'''
download_signal_train_dataset_to(param.dataset_dir)
set_random_seed_to(param.random_seed)
if not cuda_is_available():
    raise RuntimeError(f'CUDA is not available. Aborting. ')
device = torch.device('cuda')
job_name = current_utc_time()
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
dataset = ConcatDataset([
    FixDataset(param.dataset_dir, 'train', param.data_segment_length),
    FixDataset(param.dataset_dir, 'val', param.data_segment_length),
])
validation_dataset = FixDataset(param.dataset_dir, 'test', 30.0)
dataloader = DataLoader(
    dataset, param.batch_size,
    shuffle=True, pin_memory=True,
    collate_fn=validation_dataset.collate_fn,
)

'''Prepare the model.'''
model = forge_drc_model_by(
    param.model_version,
    param.model_inner_audio_channel,
    param.model_s4_hidden_size,
    param.s4_learning_rate,
    param.model_depth,
    param.model_activation,
    param.model_take_db,
    param.model_take_abs,
    param.model_take_amp,
).to(device)
model_info = get_model_info_from(model, (
    param.batch_size,
    int(param.data_segment_length * FixDataset.sample_rate)
))
if param.save_checkpoint:
    with open(job_dir / 'model-statistics.txt', 'w') as f:
        f.write(str(model_info))

'''Loss function'''
criterion = forge_loss_function_from(
    param.loss, param.loss_filter_coef).to(device)
validation_criterions: dict[LossType, nn.Module] = {
    loss_type: forge_loss_function_from(
        loss_type, param.loss_filter_coef).eval().to(device)
    for loss_type in get_args(LossType)
}

'''Prepare the optimizer'''
optimizer = AdamW(model.parameters(), lr=param.learning_rate)

'''Prepare the gradient scaler'''
scaler = GradScaler()

'''Training loop'''
if param.log_wandb:
    wandb.watch(model, log='all')
for epoch in range(param.epoch):
    clear_memory()

    training_loss = 0.0
    model.train()
    criterion.train()
    training_bar = tqdm(
        dataloader,
        desc=f'Training. {epoch = }',
        total=len(dataloader),
    )

    for x, y, _ in training_bar:
        x: Tensor
        y: Tensor
        if torch.rand(1).item() < 0.5:
            x, y = invert_phase(x, y)
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_hat: Tensor = model(x)
        loss: Tensor = criterion(y_hat, y)

        scaler.scale(loss).backward()  # type: ignore
        scaler.step(optimizer)
        scaler.update()

        training_bar.set_postfix({'loss': loss.item()})
        training_loss += loss.item()

    training_loss /= len(dataloader)

    clear_memory()

    validation_losses: defaultdict[str, float] = defaultdict(float)
    validation_evaluation_values: defaultdict[str, float] = defaultdict(float)
    validation_evaluation_plots: dict[str, Figure] = {}
    validation_audio: dict[str, wandb.Audio] = {}
    model.eval()
    criterion.eval()

    with torch.no_grad():
        print(f'Validating. {epoch = }')
        for i, (x, y, _) in enumerate(iter(validation_dataset)):
            x = x.to(device)
            y = y.to(device)

            y_hat: Tensor = model(x.unsqueeze(0)).squeeze(0)

            for validation_loss, validation_criterion in validation_criterions.items():
                this_loss: Tensor = validation_criterion(y_hat, y)
                validation_losses[f'Validation Loss: {validation_loss}'] += this_loss.item(
                )

            w_diff, w_val = evaluate_waveform_difference(y_hat, y, 'rms')
            validation_evaluation_values['Waveform Difference RMS'] += w_val

            rms_diff, rms_val = evaluate_rms_difference(y_hat, y, 'rms')
            validation_evaluation_values['RMS Difference RMS'] += rms_val

            if epoch >= 10:  # Only log plots when the model is stable.
                w_figure, w_ax = plt.subplots()
                w_ax.plot(w_val)
                validation_evaluation_plots[f'Epoch {epoch} Waveform Difference {i}'] = w_figure

                rms_figure, rms_ax = plt.subplots()
                rms_ax.plot(rms_val)
                validation_evaluation_plots[f'Epoch {epoch} RMS Difference {i}'] = rms_figure

            if epoch % 5 == 0:  # Only occasionally log audio.
                validation_audio[f'Epoch {epoch} Clip {i}'] = wandb.Audio(
                    y_hat.detach().cpu().numpy(), validation_dataset.sample_rate)

    for k, v in list(validation_losses.items()):
        validation_losses[k] = v / len(validation_dataset)
    for k, v in list(validation_evaluation_values.items()):
        validation_evaluation_values[k] = v / len(validation_dataset)

    if param.log_wandb:
        wandb.log({
            f'Training Loss: {param.loss}': training_loss,
        } | validation_losses | validation_audio | validation_evaluation_plots | validation_evaluation_values)
    if param.save_checkpoint:
        torch.save(model.state_dict(), job_dir / f'model-epoch-{epoch}.pth')

if param.log_wandb:
    wandb.finish()
