from typing import Dict, Callable, Tuple
import torch
from anndata import AnnData
from .loader import adata_loader

class Copula:
    def __init__(self, formula: str, **kwargs):
        self.formula = formula
        self.loader = None
        self.n_outcomes = None
        self.parameters = None

    def setup_data(self, adata: AnnData, marginal_formula: Dict[str, str], batch_size: int = 1024, **kwargs):
        self.adata = adata
        self.formula = self.formula | marginal_formula
        self.loader = adata_loader(adata, self.formula, batch_size=batch_size, **kwargs)
        X_batch, _ = next(iter(self.loader))
        self.n_outcomes = X_batch.shape[1]

    def fit(self, uniformizer: Callable, **kwargs):
        raise NotImplementedError

    def pseudo_obs(self, x_dict: Dict):
        raise NotImplementedError

    def likelihood(self, uniformizer: Callable, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]]):
        raise NotImplementedError

    def num_params(self, **kwargs):
        raise NotImplementedError

    def format_parameters(self):
        raise NotImplementedError