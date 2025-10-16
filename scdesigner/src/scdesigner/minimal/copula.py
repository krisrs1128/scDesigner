from typing import Dict, Callable, Tuple
import torch
from anndata import AnnData
from .loader import adata_loader
from abc import ABC, abstractmethod
import numpy as np
class Copula(ABC):
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

    @abstractmethod
    def fit(self, uniformizer: Callable, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def pseudo_obs(self, x_dict: Dict):
        raise NotImplementedError

    @abstractmethod
    def likelihood(self, uniformizer: Callable, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]]):
        raise NotImplementedError

    @abstractmethod
    def num_params(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def format_parameters(self):
        raise NotImplementedError