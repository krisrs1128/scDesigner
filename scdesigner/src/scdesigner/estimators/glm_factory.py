from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch


def glm_regression_generator(likelihood, initializer, postprocessor) -> dict:
    def estimator(
        x: np.array,
        y: np.array,
        batch_size: int = 512,
        lr: float = 0.1,
        epochs: int = 40,
    ):
        device = check_device()
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


def check_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
