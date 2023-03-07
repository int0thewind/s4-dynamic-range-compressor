import gc
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from scipy import signal


def clear_memory():
    """Clear unused CPU or GPU memory."""
    # TODO: add apple silicon support if necessary
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_tensor_device(apple_silicon: bool = True) -> torch.device:
    """A function to detect and return
    the most efficient device available on the current machine.
    CUDA is the most preferred device.
    If Apple Silicon is available, MPS would be selected.
    """

    if torch.cuda.is_available():
        return torch.device('cuda')

    if apple_silicon:
        try:
            if torch.backends.mps.is_available():
                return torch.device('mps')
        except AttributeError:
            ...

    return torch.device('cpu')


def current_utc_time() -> str:
    """Return current time in UTC timezone as a string."""
    dtn = datetime.now(timezone.utc)
    return '-'.join(list(map(str, [
        dtn.year, dtn.month, dtn.day, dtn.hour, dtn.minute, dtn.second
    ])))


def plot_freqz(b: npt.NDArray[np.float64], a: npt.NDArray[np.float64]):
    w, h = signal.freqz(b, a)

    fig, ax1 = plt.subplots()
    ax1.set_title('Frequency and Phase Response')
    ax1.plot(w, 20 * np.log10(np.abs(h)), color='blue')
    ax1.set_xlabel('Frequency (rad)')
    ax1.set_ylabel('Amplitude (dB)', color='blue')
    ax2 = ax1.twinx()
    ax2.plot(w, np.unwrap(np.angle(h)), color='green')
    ax2.set_ylabel('Phase', color='green')
    ax2.grid(True)

    return fig, (ax1, ax2)
