from anndata import AnnData
from formulaic import model_matrix
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import torch
from . import glm_regression as glm
from tqdm import tqdm

class AdataViewDataset(Dataset):
    def __init__(self, view, formula=None, chunk_size=int(1e4), device=None):
        super().__init__()
        self.view = view
        self.formula = formula
        self.len = len(view)
        self.cur_range = range(0, min(self.len, chunk_size))
        self.categories = column_levels(view.obs)
        self.x = None
        self.y = None
        self.device = device or glm.check_device()

    def __len__(self):
        return self.len
    
    def __getitem__(self, ix):
        if self.x is None or ix not in self.cur_range:
            self.cur_range = range(ix, min(ix + len(self.cur_range), self.len))
            view_inmem = self.view[self.cur_range].to_memory()
            self.x = safe_model_matrix(view_inmem.obs, self.formula, self.categories).to(self.device)
            self.y = torch.from_numpy(view_inmem.X.toarray().astype(np.float32)).to(self.device)
        return self.x[ix - self.cur_range[0]], self.y[ix - self.cur_range[0]]

def safe_model_matrix(obs, formula, categories):
    if formula is None:
        return obs

    for k in obs.columns:
        if str(obs[k].dtype) == "category":
            obs[k] = obs[k].astype(pd.CategoricalDtype(categories[k]))

    x = model_matrix(formula, obs)
    return torch.from_numpy(np.array(x).astype(np.float32))

def column_levels(obs):
    categories = {}
    for k in obs.columns:
        obs_type = str(obs[k].dtype)
        if obs_type in ["category", "object"]:
            categories[k] = obs[k].unique()
    return categories

def negative_binomial_regression_array_ondisk(
    adata: AnnData, formula, chunk_size: int = int(1e4),
    batch_size: int = 1024, lr: float = 0.1, epochs: int = 40,
    device: str = None
) -> dict:
    device = device or glm.check_device()
    dataset = AdataViewDataset(adata, formula, chunk_size, device)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    x, y = next(iter(dataloader))
    n_features, n_outcomes = x.shape[1], y.shape[1]
    params = torch.zeros(
        n_features * n_outcomes + n_outcomes, requires_grad=True, device=device
    )
    optimizer = torch.optim.Adam([params], lr=lr)

    for _ in range(epochs):
        with tqdm(dataloader, desc="Epoch Progress", leave=False) as progress_bar:
            for x_batch, y_batch in progress_bar:
                optimizer.zero_grad()
                loss = glm.negative_binomial_regression_likelihood(params, x_batch, y_batch)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix({"loss": loss.item()})

    b_elem = n_features * n_outcomes
    beta = glm.to_np(params[:b_elem]).reshape(n_features, n_outcomes)
    dispersion = np.exp(glm.to_np(params[b_elem:]))
    return {"coefficient": beta, "dispersion": dispersion}


def negative_binomial_regression_ondisk(adata: AnnData, formula: str, **kwargs) -> dict:
    parameters = negative_binomial_regression_array_ondisk(adata, formula, **kwargs)
    return glm.format_nb_parameters(parameters, list(adata.var_names), list(adata.X.columns))
