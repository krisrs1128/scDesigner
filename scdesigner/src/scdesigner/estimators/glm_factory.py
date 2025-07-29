from tqdm import tqdm
from torch.utils.data import DataLoader
import torch


def glm_regression_factory(likelihood, initializer, postprocessor) -> dict:
    def estimator(
        dataloader: DataLoader,
        lr: float = 0.1,
        epochs: int = 40,
    ):
        device = check_device()
        x, y = next(iter(dataloader))
        params = initializer(x, y, device)
        optimizer = torch.optim.Adam([params], lr=lr)

        for epoch in range(epochs):
            for x_batch, y_batch in (pbar := tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)):
                optimizer.zero_grad()
                loss = likelihood(params, x_batch, y_batch)
                loss.backward()
                optimizer.step()
                pbar.set_postfix_str(f"loss: {loss.item()}")


        return postprocessor(params, x.shape[1], y.shape[1])

    return estimator


def check_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
