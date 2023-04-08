from collections import defaultdict

import torch
import wandb
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from src.augmentation import invert_phase
from src.dataset import FixDataset
from src.loss import forge_loss_criterion_by, forge_validation_criterions_by
from src.main_routine import do_preparatory_work, print_and_save_model_info
from src.model import S4FixSideChainModel
from src.parameter import ConditionalTaskParameter
from src.utils import clear_memory

if __name__ != '__main__':
    raise RuntimeError(f'The main script cannot be imported by other module.')


param = ConditionalTaskParameter.parse_args()

job_dir, device = do_preparatory_work(
    param, param.dataset_dir, param.checkpoint_dir,
    param.random_seed, param.save_checkpoint
)

if param.log_wandb:
    wandb.init(
        name=job_dir.name,
        project=param.wandb_project_name,
        entity=param.wandb_entity,
        config=param.to_dict(),
    )

'''Prepare the dataset.'''
dataset = ConcatDataset([
    FixDataset(param.dataset_dir, 'train', param.data_segment_length),
    FixDataset(param.dataset_dir, 'validation', param.data_segment_length),
])
validation_dataset = FixDataset(param.dataset_dir, 'test', 30.0)
dataloader = DataLoader(
    dataset, param.batch_size,
    shuffle=True, pin_memory=True,
    collate_fn=validation_dataset.collate_fn,
)

'''Prepare the model.'''
model = S4FixSideChainModel(
    param.model_version,
    param.model_inner_audio_channel,
    param.model_s4_hidden_size,
    param.s4_learning_rate,
    param.model_depth,
    param.model_activation,
    param.model_convert_to_decibels,
).to(device)
print_and_save_model_info(
    model,
    (param.batch_size, int(param.data_segment_length * FixDataset.sample_rate)),
    job_dir,
    param.save_checkpoint
)

'''Loss function'''
criterion = forge_loss_criterion_by(
    param.loss, param.loss_filter_coef).to(device)
validation_criterions = forge_validation_criterions_by(
    param.loss_filter_coef, device)

'''Prepare the optimizer'''
optimizer = AdamW(model.parameters(), lr=param.learning_rate)

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

        training_bar.set_postfix({'loss': loss.item()})
        training_loss += loss.item()

        loss.backward()
        optimizer.step()

    training_loss /= len(dataloader)

    clear_memory()

    validation_losses: defaultdict[str, float] = defaultdict(float)
    model.eval()
    criterion.eval()

    with torch.no_grad():
        print(f'Validating. {epoch = }')
        for i, (x, y, _) in enumerate(iter(validation_dataset)):
            x = x.to(device)
            y = y.to(device)

            y_hat: Tensor = model(x.unsqueeze(0)).squeeze(0)

            for validation_loss, validation_criterion in validation_criterions.items():
                this_loss: Tensor = validation_criterion(
                    y_hat.unsqueeze(0), y.unsqueeze(0))
                validation_losses[f'Validation Loss: {validation_loss}'] += this_loss.item(
                )

    for k, v in list(validation_losses.items()):
        validation_losses[k] = v / len(validation_dataset)

    if param.log_wandb:
        wandb.log({
            f'Training Loss: {param.loss}': training_loss,
        } | validation_losses)
    if param.save_checkpoint:
        torch.save(model.state_dict(), job_dir / f'model-epoch-{epoch}.pth')

if param.log_wandb:
    wandb.finish()
