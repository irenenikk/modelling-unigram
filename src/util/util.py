import pathlib
import numpy as np
import torch


def config(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def mkdir(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
