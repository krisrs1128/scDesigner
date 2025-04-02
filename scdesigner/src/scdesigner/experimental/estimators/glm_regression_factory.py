import numpy as np
from . import glm_regression as glm
from torch.utils.data import DataLoader, TensorDataset
import torch


def glm_regression_generator(likelihood, initializer, postprocessor) -> dict:
    def estimator(
        x: np.array,
        y: np.array,
        batch_size: int = 512,
        lr: float = 0.1,
        epochs: int = 40,
    ):
        device = glm.check_device()
        dataset = TensorDataset(
            torch.tensor(x, dtype=torch.float32).to(device),
            torch.tensor(y, dtype=torch.float32).to(device),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size)
        params = initializer(x, y, device)
        optimizer = torch.optim.Adam([params], lr=lr)

        for _ in range(epochs):
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                loss = likelihood(params, x_batch, y_batch)
                loss.backward()
                optimizer.step()

        return postprocessor(params, x.shape[1], y.shape[1])

    return estimator
