"""Reproducibility and random seed management across Python, NumPy, and PyTorch."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Set random seeds across all standard libraries for complete reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, configure PyTorch CUDA backend for deterministic algorithms.

    Returns:
        The seed integer set.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed
