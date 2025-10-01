from .scd3 import SCD3Simulator
from .standard_covariance import StandardCovariance
from anndata import AnnData
from typing import Dict, Optional
import torch
import numpy as np

class CompositeCopula(SCD3Simulator):
    def __init__(self, marginals: Dict,
                 copula_formula: Optional[str] = None) -> None:
        self.marginals = marginals
        self.copula = StandardCovariance(copula_formula)
        self.template = None
        self.parameters = None

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