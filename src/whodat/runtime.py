"""Uses CUDA when available and sets PyTorch's CPU thread count otherwise."""

import os
import warnings
from functools import cache

import torch

# pyannote turns TF32 off for reproducibility and says so on every inference.
warnings.filterwarnings("ignore", message="TensorFloat-32")


@cache
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    torch.set_num_threads(os.cpu_count() or 1)
    return torch.device("cpu")
