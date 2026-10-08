"""CPU reproducibility utilities."""
import random
import numpy as np
import torch


def set_seed(seed: int, threads: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
