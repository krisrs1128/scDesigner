from tqdm import tqdm, trange
from anndata import AnnData
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch


def glm_regression_generator(likelihood, initializer, postprocessor) -> dict:
    def estimator(
        dataloader: DataLoader,
        lr: float = 0.1,
        epochs: int = 40,
    ):
        device = check_device()
        x, y = next(iter(dataloader))
        params = initializer(x, y, device)
        optimizer = torch.optim.Adam([params], lr=lr)

        with tqdm(range(epochs), desc="Epoch", position=1) as epoch_progress:
            for _ in range(epochs):
                with tqdm(dataloader, desc="Batch", position=0, leave=True) as batch_progress:
                    for x_batch, y_batch in dataloader:
                        optimizer.zero_grad()
                        loss = likelihood(params, x_batch, y_batch)
                        loss.backward()
                        optimizer.step()
                        #epoch_progress.set_postfix()
                        batch_progress.set_postfix({"loss": loss.item()})

        return postprocessor(params, x.shape[1], y.shape[1])

    return estimator


def check_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
