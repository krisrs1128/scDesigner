from .loader import obs_loader
from .scd3 import SCD3Simulator
from .standard_covariance import StandardCovariance
from anndata import AnnData
from typing import Dict, Optional, List
import numpy as np
import torch

class CompositeCopula(SCD3Simulator):
    def __init__(self, marginals: List,
                 copula_formula: Optional[str] = None) -> None:
        self.marginals = marginals
        self.copula = StandardCovariance(copula_formula)
        self.template = None
        self.parameters = None
        self.merged_formula = None

    def fit(
        self,
        adata: AnnData,
        **kwargs):
        """Fit the simulator"""
        self.template = adata
        merged_formula = {}

        # fit each marginal model
        for m in range(len(self.marginals)):
            self.marginals[m][1].setup_data(adata[:, self.marginals[m][0]], **kwargs)
            self.marginals[m][1].setup_optimizer(**kwargs)
            self.marginals[m][1].fit(**kwargs)

            # prepare formula for copula loader
            f = self.marginals[m][1].formula
            prefixed_f = {f"group{m}_{k}": v for k, v in f.items()}
            merged_formula = merged_formula | prefixed_f

        # copula simulator
        self.merged_formula = merged_formula
        self.copula.setup_data(adata, merged_formula, **kwargs)
        self.copula.fit(self.merged_uniformize, **kwargs)
        self.parameters = {
            "marginal": [m[1].parameters for m in self.marginals],
            "copula": self.copula.parameters
        }

    def merged_uniformize(self, y: torch.Tensor, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Produce a merged uniformized matrix for all marginals.

        Delegates to each marginal's `uniformize` method and places the
        result into the columns of a full matrix according to the variable
        selection given in `self.marginals[m][0]`.
        """
        y_np = y.detach().cpu().numpy()
        u = np.empty_like(y_np, dtype=float)

        for m in range(len(self.marginals)):
            sel = self.marginals[m][0]
            ix = _var_indices(sel, self.template)

            # remove the `group{m}_` prefix we used to distinguish the marginals
            prefix = f"group{m}_"
            cur_x = {k.removeprefix(prefix): v if k.startswith(prefix) else v for k, v in x.items()}

            # slice the subset of y for this marginal and call its uniformize
            y_sub = torch.from_numpy(y_np[:, ix])
            u[:, ix] = self.marginals[m][1].uniformize(y_sub, cur_x)
        return torch.from_numpy(u)

    def predict(self, obs=None, batch_size: int = 1000, **kwargs):
        """Predict from an obs dataframe"""
        # prepare an internal data loader for this obs
        if obs is None:
            obs = self.template.obs
        loader = obs_loader(
            obs,
            self.merged_formula,
            batch_size=batch_size,
            **kwargs
        )

        # prepare per-marginal collectors
        n_marginals = len(self.marginals)
        local_pred = [[] for _ in range(n_marginals)]

        # for each batch, call each marginal's predict on its subset of x
        for _, x_dict in loader:
            for m in range(n_marginals):
                prefix = f"group{m}_"
                # build cur_x where prefixed keys are unprefixed for the marginal
                cur_x = {k.removeprefix(prefix): v for k, v in x_dict.items()}
                params = self.marginals[m][1].predict(cur_x)
                local_pred[m].append(params)

        # merge batch-wise parameter dicts for each marginal and return
        results = []
        for m in range(n_marginals):
            parts = local_pred[m]
            keys = list(parts[0].keys())
            results.append({k: torch.cat([d[k] for d in parts]).detach().cpu().numpy() for k in keys})

        return results


def _var_indices(sel, adata: AnnData) -> np.ndarray:
    """Return integer indices of `sel` within `adata.var_names`.

    Expected use: `sel` is a list (or tuple) of variable names (strings).
    """
    # If sel is a single string, make it a list so we return consistent shape
    single_string = False
    if isinstance(sel, str):
        sel = [sel]
        single_string = True

    idx = np.asarray(adata.var_names.get_indexer(sel), dtype=int)
    if (idx < 0).any():
        missing = [s for s, i in zip(sel, idx) if i < 0]
        raise KeyError(f"Variables not found in adata.var_names: {missing}")
    return idx if not single_string else idx.reshape(-1)