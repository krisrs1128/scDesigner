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

def multiple_formula_regression_factory(likelihood, initializer, postprocessor) -> dict:
    def estimator(
        dataloaders: dict[str, DataLoader],
        lr: float = 0.1,
        epochs: int = 40,
    ):  
        device = check_device()
        x_dict = {}
        y_dict = {}
        for key in dataloaders.keys():
            x_dict[key], y_dict[key] = next(iter(dataloaders[key]))
        # check if all ys are the same
        y_ref = y_dict[list(dataloaders.keys())[0]]
        for key in dataloaders.keys():
            if not torch.equal(y_dict[key], y_ref):
                raise ValueError(f"Ys are not the same for {key}")
        params = initializer(x_dict, y_ref, device) # x is a dictionary of tensors, y is a tensor
        optimizer = torch.optim.Adam([params], lr=lr)
        
        keys = list(dataloaders.keys())
        loaders = list(dataloaders.values())
        
        for epoch in range(epochs):
            num_keys = len(keys)
            for batches in (pbar := tqdm(zip(*loaders), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)):
                x_batch_dict = {
                    keys[i]: batches[i][0].to(device) for i in range(num_keys)
                }
                y_batch = batches[0][1].to(device)
                optimizer.zero_grad()
                loss = likelihood(params, x_batch_dict, y_batch) 
                loss.backward()
                optimizer.step()
                pbar.set_postfix_str(f"loss: {loss.item()}")

        return postprocessor(params, x_dict, y_ref)

    return estimator


def check_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
