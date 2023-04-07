import platform
from collections import defaultdict
from pprint import pprint
from typing import get_args

import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.cuda import is_available as cuda_is_available
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchinfo import summary as get_model_info_from
from tqdm import tqdm

from src.augmentation import invert_phase
from src.dataset import SignalTrainDataset, download_signal_train_dataset_to
from src.loss import LossType, forge_loss_function_from
from src.model import S4ConditionalSideChainModel
from src.parameter import ConditionalTaskParameter
from src.utils import clear_memory, current_utc_time, set_random_seed_to

if __name__ != '__main__':
    raise RuntimeError(f'The main script cannot be imported by other module.')


# TODO: consider adding learning rate scheduler
# But firstly, how to deal with S4 custom learning rate?

'''Script parameters.'''
param = ConditionalTaskParameter.parse_args()
pprint(param.to_dict())

'''The preparatory work.'''
download_signal_train_dataset_to(param.dataset_dir)
set_random_seed_to(param.random_seed)
if not cuda_is_available():
    raise RuntimeError(f'CUDA is not available.')
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
training_dataset = SignalTrainDataset(param.dataset_dir, 'train', 1.0)
training_dataloader = DataLoader(
    training_dataset, param.batch_size,
    shuffle=True, pin_memory=True,
    collate_fn=training_dataset.collate_fn,
)
validation_dataset = SignalTrainDataset(param.dataset_dir, 'validation', 3.0)
validation_dataloader = DataLoader(
    validation_dataset, param.batch_size,
    shuffle=True, pin_memory=True,
    collate_fn=training_dataset.collate_fn,
)

'''Prepare the model.'''
model = S4ConditionalSideChainModel(
    param.model_version,
    param.model_control_parameter_mlp_depth,
    param.model_control_parameter_mlp_hidden_size,
    param.model_film_take_batch_normalization,
    param.model_inner_audio_channel,
    param.model_s4_hidden_size,
    param.s4_learning_rate,
    param.model_depth,
    param.model_activation,
    param.model_convert_to_decibels,
).to(device)
model_info = get_model_info_from(model, (
    (param.batch_size,
        int(param.data_segment_length * SignalTrainDataset.sample_rate)),
    (param.batch_size, 2),
))
if param.save_checkpoint:
    if platform.system() == 'Linux':
        with open(job_dir / 'model-statistics.txt', 'wb') as f:
            f.write(str(model_info).encode())
    else:
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
scaler = GradScaler() if param.enable_gradient_scaling else None

'''Training loop'''
if param.log_wandb:
    wandb.watch(model, log='all')
for epoch in range(param.epoch):
    clear_memory()

    training_loss = 0.0
    model.train()
    criterion.train()
    training_bar = tqdm(
        training_dataloader,
        desc=f'Training. {epoch = }',
        total=len(training_dataloader),
    )

    for side_chain, y, parameters in training_bar:
        side_chain: Tensor
        y: Tensor
        parameters: Tensor

        if torch.rand(1).item() < 0.5:
            side_chain, y = invert_phase(side_chain, y)

        side_chain = side_chain.to(device)
        y = y.to(device)
        parameters = parameters.to(device)

        optimizer.zero_grad()

        y_hat: Tensor = model(side_chain, parameters)
        loss: Tensor = criterion(y_hat, y)

        training_bar.set_postfix({'loss': loss.item()})
        training_loss += loss.item()

        if scaler is not None:
            scaler.scale(loss).backward()  # type: ignore
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    training_loss /= len(training_dataloader)

    clear_memory()

    validation_losses: defaultdict[str, float] = defaultdict(float)
    model.eval()
    criterion.eval()
    validation_bar = tqdm(
        validation_dataloader,
        desc=f'Validating. {epoch = }',
        total=len(validation_dataloader),
    )

    with torch.no_grad():
        for side_chain, y, parameters in validation_bar:
            side_chain: Tensor = side_chain.to(device)
            y: Tensor = y.to(device)
            parameters: Tensor = parameters.to(device)

            y_hat: Tensor = model(side_chain, parameters)

            for validation_loss, validation_criterion in validation_criterions.items():
                loss: Tensor = validation_criterion(y_hat, y)
                validation_losses[f'Validation Loss: {validation_loss}'] += loss.item()

    for k, v in list(validation_losses.items()):
        validation_losses[k] = v / len(validation_dataloader)

    if param.log_wandb:
        wandb.log({
            f'Training Loss: {param.loss}': training_loss,
        } | validation_losses)
    if param.save_checkpoint:
        torch.save(model.state_dict(), job_dir / f'model-epoch-{epoch}.pth')

if param.log_wandb:
    wandb.finish()
