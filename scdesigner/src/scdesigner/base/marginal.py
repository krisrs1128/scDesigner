from ..utils.kwargs import DEFAULT_ALLOWED_KWARGS, _filter_kwargs
from ..utils.loader import adata_loader, get_device
from anndata import AnnData
from typing import Union, Dict, Optional, Tuple
import pandas as pd
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Marginal(ABC):
    def __init__(self, formula: Union[Dict, str]):
        self.formula = formula
        self.feature_dims = None
        self.loader = None
        self.n_outcomes = None
        self.predict = None
        self.predictor_names = None
        self.parameters = None
        self.device = get_device()

    def setup_data(self, adata: AnnData, batch_size: int = 1024, **kwargs):
        """Set up the dataloader for the AnnData object."""
        # keep a reference to the AnnData for later use (e.g., var_names)
        self.adata = adata
        self.loader = adata_loader(adata, self.formula, batch_size=batch_size, **kwargs)
        X_batch, obs_batch = next(iter(self.loader))
        self.n_outcomes = X_batch.shape[1]
        self.feature_dims = {k: v.shape[1] for k, v in obs_batch.items()}
        self.predictor_names = self.loader.dataset.predictor_names

    def fit(self, max_epochs: int = 100, **kwargs):
        """Fit the marginal predictor using vanilla PyTorch training loop."""
        if self.predict is None:
            self.setup_optimizer(**kwargs)

        for epoch in range(max_epochs):
            epoch_loss, n_batches = 0.0, 0

            for batch in self.loader:
                y, x = batch
                if y.device != self.device:
                    y = y.to(self.device)
                    x = {k: v.to(self.device) for k, v in x.items()}

                self.predict.optimizer.zero_grad()
                loss = self.predict.loss_fn((y, x))
                loss.backward()
                self.predict.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            print(f"Epoch {epoch}/{max_epochs}, Loss: {avg_loss:.4f}", end='\r')
        self.parameters = self.format_parameters()

    def format_parameters(self):
        """Convert fitted coefficient tensors into pandas DataFrames.

        Returns:
            dict: mapping from parameter name -> pandas.DataFrame with rows
                corresponding to predictor column names (from
                `self.predictor_names[param]`) and columns corresponding to
                `self.adata.var_names` (gene names). The values are moved to
                CPU and converted to numpy floats.
        """
        var_names = list(self.adata.var_names)

        dfs = {}
        for param, tensor in self.predict.coefs.items():
            coef_np = tensor.detach().cpu().numpy()
            row_names = list(self.predictor_names[param])
            dfs[param] = pd.DataFrame(coef_np, index=row_names, columns=var_names)
        return dfs

    def num_params(self):
        """Return the number of parameters."""
        if self.predict is None:
            return 0
        return sum(p.numel() for p in self.predict.parameters() if p.requires_grad)

    @abstractmethod
    def setup_optimizer(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def likelihood(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]]):
        """Compute the (negative) log-likelihood or loss for a batch.
        """
        raise NotImplementedError

    @abstractmethod
    def invert(self, u: torch.Tensor, x: Dict[str, torch.Tensor]):
        """Invert pseudoobservations."""
        raise NotImplementedError

    @abstractmethod
    def uniformize(self, y: torch.Tensor, x: Dict[str, torch.Tensor]):
        """Uniformize using learned CDF.
        """
        raise NotImplementedError


class GLMPredictor(nn.Module):
    """GLM-style predictor with arbitrary named parameters.

    Args:
        n_outcomes: number of model outputs (e.g. genes)
        feature_dims: mapping from param name -> number of covariate features
        link_fns: optional mapping from param name -> callable(link) applied to linear predictor

    The module will create one coefficient matrix per named parameter with shape
    (n_features_for_param, n_outcomes) and expose them as Parameters under
    `self.coefs[param_name]`.
    """
    def __init__(
        self,
        n_outcomes: int,
        feature_dims: Dict[str, int],
        link_fns: Dict[str, callable] = None,
        loss_fn: Optional[callable] = None,
        optimizer_class: Optional[callable] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.n_outcomes = int(n_outcomes)
        self.feature_dims = dict(feature_dims)
        self.param_names = list(self.feature_dims.keys())

        self.link_fns = link_fns or {k: torch.exp for k in self.param_names}
        self.coefs = nn.ParameterDict()
        for key, dim in self.feature_dims.items():
            self.coefs[key] = nn.Parameter(torch.zeros(dim, self.n_outcomes))
        self.reset_parameters()

        self.loss_fn = loss_fn
        self.to(get_device())

        optimizer_kwargs = optimizer_kwargs or {}
        filtered_kwargs = _filter_kwargs(optimizer_kwargs, DEFAULT_ALLOWED_KWARGS['optimizer'])
        self.optimizer = optimizer_class(self.parameters(), **filtered_kwargs)

    def reset_parameters(self):
        for p in self.coefs.values():
            nn.init.normal_(p, mean=0.0, std=1e-4)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for name in self.param_names:
            x_beta = obs_dict[name] @ self.coefs[name]
            link = self.link_fns.get(name, torch.exp)
            out[name] = link(x_beta)
        return out