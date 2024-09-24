import torch
import pandas as pd
import numpy as np


def default_device(device):
    if device is not None:
        return device
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def parameter_to_df(theta, y_names, x_names):
    theta = pd.DataFrame(theta.detach().cpu())
    theta.columns = y_names
    theta.index = x_names
    return theta


def parameter_to_tensor(theta, device):
    return torch.from_numpy(np.array(theta)).to(device)
