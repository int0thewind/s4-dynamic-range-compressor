import time
from pathlib import Path
from pprint import pprint
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Audio, display
from scipy.signal import freqz
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SignalTrainDataset, download_signal_train_dataset_to
from src.loss import forge_validation_criterions_by
from src.model import S4ConditionalModel
from src.model.layer import DSSM, convert_to_decibel
from src.parameter import ConditionalTaskParameter
from src.utils import get_tensor_device, set_random_seed_to

if __name__ != '__main__':
    raise ImportError('This script cannot be imported.')

DATASET_DIR = Path('./data/SignalTrain')

CHECKPOINT_DIR = Path('./experiment-result')
JOB_NAME = '2023-5-10-15-27-38'
EPOCH = 66

job_dir = CHECKPOINT_DIR / JOB_NAME
job_eval_dir = job_dir / 'eval'
device = get_tensor_device(apple_silicon=False)  # Some operations are not supported on Apple Silicon
param = ConditionalTaskParameter.from_json(CHECKPOINT_DIR / JOB_NAME / 'config.json')
# pprint(param.to_dict())

set_random_seed_to(param.random_seed)

download_signal_train_dataset_to(DATASET_DIR)
testing_dataset_short = SignalTrainDataset(DATASET_DIR, 'test', 1.5)
testing_dataset_mid = SignalTrainDataset(DATASET_DIR, 'test', 4)
testing_dataset_long = SignalTrainDataset(DATASET_DIR, 'test', 10)

model = S4ConditionalModel(
    param.model_take_side_chain,
    param.model_inner_audio_channel,
    param.model_s4_hidden_size,
    param.s4_learning_rate,
    param.model_depth,
    param.model_film_take_batchnorm,
    param.model_take_residual_connection,
    param.model_convert_to_decibels,
    param.model_take_tanh,
    param.model_activation,
).eval().to(device)
model.load_state_dict(torch.load(CHECKPOINT_DIR / JOB_NAME / f'model-epoch-{EPOCH}.pth', map_location=device))


@torch.no_grad()
def test():
    for dataset in [testing_dataset_short, testing_dataset_mid, testing_dataset_long]:
        dataloader = DataLoader(dataset, 64, num_workers=8, pin_memory=True)

        validation_criterions = forge_validation_criterions_by(param.loss_filter_coef, device)
        validation_losses = {
            validation_loss: 0.0
            for validation_loss in validation_criterions.keys()
        }

        for x, y, parameters in tqdm(dataloader, desc='Testing', total=len(dataloader)):
            x: Tensor = x.to(device)
            y: Tensor = y.to(device)
            parameters: Tensor = parameters.to(device)

            y_hat: Tensor = model(x, parameters)

            for validation_loss, validation_criterion in validation_criterions.items():
                loss: Tensor = validation_criterion(y_hat.unsqueeze(1), y.unsqueeze(1))
                validation_losses[validation_loss] += loss.item()
        
        for k, v in list(validation_losses.items()):
            validation_losses[k] = v / len(dataloader)
        
        with open(job_eval_dir / 'loss.txt', 'a') as f:
            pprint(validation_losses, stream=f)
            print('\n', file=f)

test()

@torch.no_grad()
def s4_frequency_response_analysis():
    for c, block in enumerate(model.blocks):
        s4 = block.s4
        assert isinstance(s4, DSSM)
        kernel = s4.get_kernel(int(SignalTrainDataset.sample_rate * 1))
        for r in range(param.model_inner_audio_channel):
            fig, ax = plt.subplots()
            impulse_response = kernel[r, :].detach().cpu().numpy()
            w, h = freqz(impulse_response)
            title = f'layer-{c + 1}-channel-{r + 1}'
            ax.set_title(title)
            ax.plot(w, 20 * np.log10(abs(h)), 'b')
            ax.set_xlabel('Frequency [rad/sample]')
            ax.set_ylabel('Amplitude [dB]', color='b')
            ax2 = ax.twinx()
            ax2.plot(w, np.unwrap(np.angle(h)), 'g')
            ax2.set_ylabel('Angle (radians)', color='g')
            ax2.grid(True)
            fig.savefig(str(job_eval_dir / 's4-impulse-response' / f'{title}.png'))

s4_frequency_response_analysis()

# @torch.no_grad()
# def evaluate_inference_efficiency():
#     if device.type == 'cpu':
#         print(f'Doing inference speed test on CPU...')
#     elif device.type == 'cuda':
#         print(f'Doing inference speed test on {torch.cuda.get_device_name()}.')
    
#     print(f'Individual sample length: {TESTING_DATASET_SEGMENT_LENGTH} seconds.')

#     inference_time: list[int] = []

#     for i in tqdm(range(10)):
#         x, _, cond = testing_dataset.collate_fn([testing_dataset[i]])
#         x = x.to(device)
#         cond = cond.to(device)

#         tic = time.perf_counter_ns()
#         model(x, cond)
#         toc = time.perf_counter_ns()
#         inference_time.append(toc - tic)
    
#     inference_time_mean = mean(inference_time) / 1e6
#     print(f'Average inference time: {inference_time_mean} ms.')
#     inference_time_stdev = stdev(inference_time) / 1e6
#     print(f'Inference time standard deviation: {inference_time_stdev} ms.')
#     speed_ratio = inference_time_mean / (TESTING_DATASET_SEGMENT_LENGTH * 1e3)
#     print(f'Real-time speed ratio: {speed_ratio}.')

# evaluate_inference_efficiency()

@torch.no_grad()
def evaluate_output_audio():
    x, y, cond = testing_dataset.collate_fn([testing_dataset[i] for i in TESTING_DATASET_SAMPLE_INDEX])
    x = x.to(device)
    y = y.to(device)
    cond = cond.to(device)
    
    y_hat: Tensor = model(y, cond)
    
    for x_audio, y_audio, y_hat_audio, cond_data in zip(
        x.split(1), y.split(1), y_hat.split(1), cond.split(1),
    ):
        switch, peak_reduction = cond_data.flatten().cpu().tolist()
        display(f'Switch value: {switch}, Peak reduction value: {peak_reduction}')
        display(Audio(x_audio.flatten().cpu().numpy(), rate=testing_dataset.sample_rate, normalize=False))
        display(Audio(y_audio.flatten().cpu().numpy(), rate=testing_dataset.sample_rate, normalize=False))
        display(f'y-audio peak {y_audio.max(), y_audio.min()}')
        display(Audio(y_hat_audio.flatten().cpu().numpy(), rate=testing_dataset.sample_rate, normalize=False))
        display(f'y-hat-audio peak {y_hat_audio.max(), y_hat_audio.min()}')

evaluate_output_audio()
