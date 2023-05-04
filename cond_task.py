import torch
import wandb
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentation import invert_phase
from src.dataset import SignalTrainDataset
from src.loss import forge_loss_criterion_by, forge_validation_criterions_by
from src.main_routine import do_preparatory_work, print_and_save_model_info
from src.model import S4ConditionalModel
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
training_dataset = SignalTrainDataset(
    param.dataset_dir, 'train', param.data_segment_length)
training_dataloader = DataLoader(
    training_dataset, param.batch_size,
    shuffle=True, pin_memory=True,
    collate_fn=training_dataset.collate_fn,
)
validation_dataset = SignalTrainDataset(
    param.dataset_dir, 'validation', param.data_segment_length * 3)
validation_dataloader = DataLoader(
    validation_dataset, param.batch_size,
    shuffle=True, pin_memory=True,
    collate_fn=validation_dataset.collate_fn,
)

'''Prepare the model.'''
model = torch.compile(S4ConditionalModel(
    param.model_inner_audio_channel,
    param.model_s4_hidden_size,
    param.s4_learning_rate,
    param.model_depth,
    param.model_film_take_batchnorm,
    param.model_activation,
).to(device))
print_and_save_model_info(
    model,
    ((param.batch_size, int(param.data_segment_length * training_dataset.sample_rate)),
     (param.batch_size, 2)),
    job_dir, param.save_checkpoint
)

'''Loss function'''
criterion = forge_loss_criterion_by(
    param.loss, param.loss_filter_coef).to(device)
validation_criterions = forge_validation_criterions_by(
    param.loss_filter_coef, device, param.loss)

'''Prepare the optimizer'''
optimizer = AdamW(model.parameters(), lr=param.learning_rate)

'''Prepare the learning rate scheduler'''

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
        loss: Tensor = criterion(y_hat.unsqueeze(1), y.unsqueeze(1))

        training_bar.set_postfix({'loss': loss.item()})
        training_loss += loss.item()

        loss.backward()
        optimizer.step()

    training_loss /= len(training_dataloader)

    clear_memory()

    validation_losses = {
        f'Validation Loss: {validation_loss}': 0.0
        for validation_loss in validation_criterions.keys()
    }
    model.eval()
    criterion.eval()
    validation_bar = tqdm(
        validation_dataloader,
        desc=f'Validating. {epoch = }',
        total=len(validation_dataloader),
    )

    with torch.no_grad():
        for x, y, parameters in validation_bar:
            x: Tensor = x.to(device)
            y: Tensor = y.to(device)
            parameters: Tensor = parameters.to(device)

            y_hat: Tensor = model(x, parameters)

            for validation_loss, validation_criterion in validation_criterions.items():
                loss: Tensor = validation_criterion(
                    y_hat.unsqueeze(1), y.unsqueeze(1))
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
