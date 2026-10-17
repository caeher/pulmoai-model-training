from __future__ import annotations

import random

import numpy as np
import torch

from pulmoai.config import ExperimentConfig


def configure_runtime(config: ExperimentConfig) -> torch.device:
    seed = config.seeds.global_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(config.train.cpu_threads)
    torch.use_deterministic_algorithms(config.seeds.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.seeds.deterministic
    choice = config.train.device
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    return torch.device(
        "cuda"
        if choice == "auto" and torch.cuda.is_available()
        else "cpu"
        if choice == "auto"
        else choice
    )
