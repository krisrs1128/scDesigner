from .copula import Copula
from .loader import obs_loader, adata_loader
from .marginal import Marginal
from .simulator import Simulator
from anndata import AnnData
from tqdm import tqdm
import torch
import numpy as np

class SCD3Simulator(Simulator):
    """Simulation wrapper"""

    def __init__(self, marginal: Marginal, copula: Copula):
        self.marginal = marginal
        self.copula = copula
        self.template = None
        self.parameters = None

    def fit(
        self,
        adata: AnnData,
        **kwargs):
        """Fit the simulator"""
        self.template = adata
        self.marginal.setup_data(adata, **kwargs)
        self.marginal.setup_optimizer(**kwargs)
        self.marginal.fit(**kwargs)

        # copula simulator
        self.copula.setup_data(adata, self.marginal.formula, **kwargs)
        self.copula.fit(self.marginal.uniformize, **kwargs)
        self.parameters = {
            "marginal": self.marginal.parameters,
            "copula": self.copula.parameters
        }

    def predict(self, obs=None, batch_size: int = 1000, **kwargs):
        """Predict from an obs dataframe"""
        # prepare an internal data loader for this obs
        if obs is None:
            obs = self.template.obs
        loader = obs_loader(
            obs,
            self.marginal.formula,
            batch_size=batch_size,
            **kwargs
        )

        # get predictions across batches
        local_parameters = []
        for _, x_dict in loader:
            l = self.marginal.predict(x_dict)
            local_parameters.append(l)

        # convert to a merged dictionary
        keys = list(local_parameters[0].keys())
        return {
            k: torch.cat([d[k] for d in local_parameters]).detach().cpu().numpy()
            for k in keys
        }

    def sample(self, obs=None, batch_size: int = 1000, **kwargs):
        """Generate samples."""
        if obs is None:
            obs = self.template.obs
        loader = obs_loader(
            obs,
            self.copula.formula | self.marginal.formula,
            batch_size=batch_size,
            **kwargs
        )

        # get samples across batches
        samples = []
        for _, x_dict in loader:
            u = self.copula.pseudo_obs(x_dict)
            u = torch.from_numpy(u)
            samples.append(self.marginal.invert(u, x_dict))
        samples = torch.cat(samples).detach().cpu().numpy()
        return AnnData(X = samples, obs=obs)

    def complexity(self, adata: AnnData = None, **kwargs):
        if adata is None:
            adata = self.template

        N, ll = 0, 0
        loader = adata_loader(adata, self.marginal.formula | self.copula.formula, **kwargs)
        for batch in tqdm(loader, desc="Computing log-likelihood..."):
            ll += self.copula.likelihood(self.marginal.uniformize, batch).sum()
            N += len(batch[0])

        return {
            "aic": -2 * ll + 2 * self.copula.num_params(),
            "bic": -2 * ll + np.log(N) * self.copula.num_params()
        }