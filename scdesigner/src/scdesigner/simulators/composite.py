"""Composite simulator that combines multiple marginals with a Gaussian copula.

This module provides :class:`CompositeCopula`, a simulator that fits several
marginal models and then couples their dependence structure with a 
:class:`~scdesigner.copulas.standard_copula.StandardCopula`.
"""

from ..data.loader import obs_loader
from .scd3 import SCD3Simulator
from ..copulas.standard_copula import StandardCopula
from anndata import AnnData
from typing import Dict, Optional, List
import numpy as np
import torch

class CompositeCopula(SCD3Simulator):
    """
    Composite simulator: multiple marginals + a shared Gaussian copula.

    The composite simulator fits each marginal model independently on a
    (potentially different) subset of variables, and then fits a Gaussian
    copula on the *merged* uniformized outputs from all marginals to capture
    cross-feature dependence.

    Each marginal is provided as a pair ``(sel, marginal)`` where:

    - ``sel`` selects which variables in ``adata`` the marginal is responsible
      for (e.g. a list of gene names, a single gene name).
    - ``marginal`` is an object implementing the marginal simulator interface

    Parameters
    ----------
    marginals : list
        List of ``(sel, marginal)`` pairs.
    copula_formula : str, optional
        Formula passed to :class:`~scdesigner.copulas.standard_copula.StandardCopula`
        to determine copula grouping structure (e.g. ``"group ~ 1"``). If
        ``None``, uses the copula's default.

    Attributes
    ----------
    marginals : list
        The provided marginal specifications.
    copula : StandardCopula
        The fitted copula component.
    template : AnnData or None
        Training dataset (set during :meth:`fit`).
    parameters : dict or None
        Fitted parameters, with keys ``"marginal"`` and ``"copula"``.
    merged_formula : dict or None
        Merged (prefixed) formula dictionary used to construct the copula data loader.

    Examples
    --------
    Fit two marginal models on disjoint gene sets and then fit a copula:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from anndata import AnnData
    >>> from scdesigner.simulators import CompositeCopula
    >>> from scdesigner.distributions import NegBin, Poisson
    >>>
    >>> X = np.random.poisson(1.0, size=(100, 10)).astype(float)
    >>> obs = pd.DataFrame({"cell_type": np.random.choice(["A", "B"], size=100)})
    >>> adata = AnnData(X=X, obs=obs)
    >>> adata.var_names = [f"g{i}" for i in range(adata.n_vars)]
    >>>
    >>> # Example selectors: first 5 genes vs last 5 genes
    >>> sel1 = adata.var_names[:5].tolist()
    >>> sel2 = adata.var_names[5:].tolist()
    >>> m1 = NegBin(formula={"mean": "~ cell_type", "dispersion": "~ 1"})
    >>> m2 = Poisson(formula={"mean": "~ cell_type"})
    >>>
    >>> composite = CompositeCopula([(sel1, m1), (sel2, m2)])
    >>> composite.fit(adata, batch_size=256, verbose=False)
    >>> params = composite.predict(adata.obs.iloc[:3], batch_size=3)
    """
    def __init__(self, marginals: List,
                 copula_formula: Optional[str] = None) -> None:
        """Create a composite simulator.

        Parameters
        ----------
        marginals : list
            List of ``(sel, marginal)`` pairs. See class docstring for details.
        copula_formula : str, optional
            Copula grouping formula passed to :class:`StandardCopula`.
        """
        self.marginals = marginals
        self.copula = StandardCopula(copula_formula) if copula_formula is not None else StandardCopula()
        self.template = None
        self.parameters = None
        self.merged_formula = None

    def fit(
        self,
        adata: AnnData,
        verbose: bool = True,
        **kwargs,):
        """Fit all marginals and then fit the copula on merged uniforms.

        Parameters
        ----------
        adata : AnnData
            Training dataset.
        **kwargs
            Additional keyword arguments forwarded to marginal setup/fit methods
            and to the copula's ``setup_data`` / ``fit`` calls (e.g.
            ``batch_size``).
        verbose : bool, optional
            Whether to print verbose output.
        """
        self.template = adata
        merged_formula = {}

        # fit each marginal model
        for m in range(len(self.marginals)):
            self.marginals[m][1].setup_data(adata[:, self.marginals[m][0]], **kwargs)
            self.marginals[m][1].setup_optimizer(**kwargs)
            self.marginals[m][1].fit(**kwargs, verbose=verbose)

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
        """Predict marginal parameters for observations (batched).

        This method constructs an internal loader for ``obs`` using the merged
        (prefixed) formula dictionary, then dispatches per-marginal ``predict``
        calls on each batch after stripping the prefixes.

        Parameters
        ----------
        obs : pandas.DataFrame, optional
            Observation metadata. Defaults to ``self.template.obs``.
        batch_size : int, optional
            Batch size for the internal observation loader.
        **kwargs
            Forwarded to :func:`~scdesigner.data.loader.obs_loader`.

        Returns
        -------
        list[dict[str, np.ndarray]]
            List with one element per marginal. Each element is a dict mapping
            parameter names to numpy arrays, concatenated across batches.
        """
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
    """Return integer indices of ``sel`` within ``adata.var_names``.

    Parameters
    ----------
    sel : str or list of str
        The variable names to select.
    adata : AnnData
        The AnnData object to select variables from.

    Returns
    -------
    np.ndarray
        The integer indices of the selected variables.
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