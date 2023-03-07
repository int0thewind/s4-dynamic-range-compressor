from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from auraloss.time import ESRLoss
from rootconfig import RootConfig
from torch import Tensor
from torch.optim import AdamW
from tqdm.auto import tqdm

from src.model import DSSM, Rearrange
from src.utils import current_utc_time, get_tensor_device

PROJ_NAME = 'Compressor Conditioning'


@dataclass
class Config(RootConfig):
    steps: int = 1000
    batch_size: int = 128
    lr: float = 1e-3
    loss: str = 'mse'

    sample_rate: int = 48000
    seconds: float = 0.5
    window: int = sample_rate // 1000


class Model(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.model = nn.Sequential(
            Rearrange('B L -> B L 1'),

            nn.Linear(1, 32),
            nn.GELU(),

            Rearrange('B L H -> B H L'),
            DSSM(32, 32, lr=config.lr),
            Rearrange('B H L -> B L H'),
            nn.GELU(),

            nn.Linear(32, 32),
            nn.GELU(),

            Rearrange('B L H -> B H L'),
            DSSM(32, 32, lr=config.lr),
            Rearrange('B H L -> B L H'),
            nn.GELU(),

            nn.Linear(32, 1),
            Rearrange('B L 1 -> B L'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = 20 * torch.log10(torch.clamp(torch.abs(x), 1e-4))
        x = self.model(x)
        x = 10 ** (x / 20)
        return x


@torch.no_grad()
def get_gain(x: Tensor, window: int) -> Tensor:
    assert x.dim() == 2

    def rms(i: Tensor, s: int, e: int):
        if s < 0:
            s = 0
        if s == e:
            return torch.zeros(i.size(0)).to(i)
        return torch.sqrt(torch.mean(i[:, s:e] ** 2, dim=1))

    ret = torch.stack([
        rms(x, i - window, i) for i in range(x.size(1))
    ]).to(x).transpose(0, 1)
    ret = 20 * torch.log10(torch.clamp(torch.abs(ret), 1e-4))
    ret = -ret / 2
    ret = 10 ** (ret / 20)
    # ret[:, 0] = 0.0

    assert ret.size() == x.size(), f'{ret.size() = } but {x.size() = }.'
    return ret


if __name__ == '__main__':
    config = Config.parse_args()
    job_name = current_utc_time()
    save_dir = Path('experiment-result') / PROJ_NAME / job_name
    save_dir.mkdir(parents=True, exist_ok=True)
    config.to_json(save_dir / 'config.json')
    device = get_tensor_device()

    model = Model(config).train().to(device)
    if config.loss == 'mse':
        criterion = nn.MSELoss().to(device)
    elif config.loss == 'esr':
        criterion = ESRLoss().to(device)
    else:
        raise ValueError()
    optimizer = AdamW(model.parameters(), lr=config.lr)

    wandb.init(
        name=job_name,
        project=PROJ_NAME,
        config=config.to_dict(),
    )
    wandb.watch(model, log='all')

    with tqdm(range(config.steps), desc='Training...') as t:
        for i in t:
            x = torch.rand(
                config.batch_size, int(config.sample_rate * config.seconds), device=device,
            ) * torch.rand(
                config.batch_size, int(config.sample_rate * config.seconds), device=device,
            )
            x = x * 2 - 1.0
            y = get_gain(x, config.window)

            optimizer.zero_grad()

            y_hat: Tensor = model(x)
            loss: Tensor = criterion(
                y_hat[:, config.window:],
                y[:, config.window:]
            )

            loss.backward()
            optimizer.step()
            t.set_postfix(loss=loss.detach().item())

            if i % 10 == 9:
                info = {
                    'Step': i,
                    'Training Loss': loss.detach().item()
                }
                wandb.log(info)
                torch.save(model.state_dict(), save_dir /
                           f'model-step-{i}.ckpt')

    wandb.finish()
