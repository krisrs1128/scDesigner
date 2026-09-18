import time
import inspect
import numpy as np
import pandas as pd
import anndata as ad
from scdesigner import simulators
from scdesigner.base import Simulator
import warnings

def subsample(adata, n_cells, n_genes, seed=42):
    """Randomly subsample cells and genes from an AnnData object."""
    rng = np.random.default_rng(seed)
    cidx = (
        slice(None)
        if n_cells == adata.n_obs
        else rng.choice(adata.n_obs, n_cells, replace=False)
    )
    gidx = (
        slice(None)
        if n_genes == adata.n_vars
        else rng.choice(adata.n_vars, n_genes, replace=False)
    )
    return adata.copy()[cidx, gidx]

def prepare_formula_kwargs(formulas, simulator_class):
    """Filter formulas to those accepted by a simulator.

    Parameters
    ----------
    formulas : dict
        Mapping from formula argument names to formula strings.
    simulator_class : type
        scDesigner simulator class whose constructor will receive formulas.

    Returns
    -------
    dict
        Formula keyword arguments accepted by ``simulator_class``.
    """
    init_params = inspect.signature(simulator_class.__init__).parameters
    formula_kwargs = {}

    for name, formula in formulas.items():
        if name in init_params:
            formula_kwargs[name] = formula
        else:
            warnings.warn(
                f"Warning: {name} is not a valid parameter for "
                f"{simulator_class.__name__} and will be ignored.",
                stacklevel=2
            )
    return formula_kwargs

def simulate(
    adata,
    n_cells,
    n_genes,
    model: type[Simulator],
    formulas=None,
    top_k=None,
    subsample_seed=42,
    **fit_kwargs,
):
    """Fit a scDesigner model and sample simulated data.

    Parameters
    ----------
    adata : anndata.AnnData
        Reference dataset used for fitting.
    n_cells : int or None
        Number of cells to subsample. If ``None``, all cells are used.
    n_genes : int or None
        Number of genes to subsample. If ``None``, all genes are used.
    model : type[Simulator]
        scDesigner simulator class to instantiate.
    formulas : dict, optional
        Formula keyword arguments passed to the simulator constructor.
    top_k : int, optional
        Number of genes used by the copula approximation. Passed to the
        simulator constructor.
    **fit_kwargs
        Additional keyword arguments passed to ``simulator.fit``.

    Returns
    -------
    tuple
        ``(real_data, sim_data, metrics_df, simulator)``.
    """
    n_cells = adata.n_obs if n_cells is None else n_cells
    n_genes = adata.n_vars if n_genes is None else n_genes

    if n_cells == adata.n_obs and n_genes == adata.n_vars:
        real_data = adata
    else:
        real_data = subsample(adata, n_cells, n_genes, subsample_seed)

    formulas = formulas or {}
    init_kwargs = prepare_formula_kwargs(formulas, model)
    if top_k is not None:
        init_kwargs["top_k"] = top_k
    simulator = model(**init_kwargs)

    start_time = time.time()

    simulator.fit(real_data, **fit_kwargs)
    sim_data = simulator.sample(real_data.obs)

    runtime = time.time() - start_time

    try:
        diagnose = simulator.complexity()
        marginal_aic, marginal_bic = diagnose["marginal_aic"], diagnose["marginal_bic"]
        copula_aic, copula_bic = diagnose["copula_aic"], diagnose["copula_bic"]
    except Exception as e:
        print(f"Error calculating complexity: {e}")
        aic, bic = np.nan, np.nan

    metrics = {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "runtime": runtime,
        "model": model.__name__,
        "simulator": "scdesigner",
        "max_epochs": fit_kwargs.get("max_epochs"),
        "lr": fit_kwargs.get("lr"),
        "mean_formula": formula_kwargs.get("mean_formula"),
        "dispersion_formula": formula_kwargs.get("dispersion_formula"),
        "zero_inflation_formula": formula_kwargs.get("zero_inflation_formula"),
        "copula_formula": formula_kwargs.get("copula_formula"),
        "sdev_formula": formula_kwargs.get("sdev_formula"),
        "marginal_aic": marginal_aic,
        "marginal_bic": marginal_bic,
        "copula_aic": copula_aic,
        "copula_bic": copula_bic
    }

    return real_data, sim_data, metrics, simulator


if __name__ == "__main__":


    # read data
    data_dir = "../data/HVG_embryoatlas.h5ad"
    adata = ad.read_h5ad(data_dir)
    
    # specify the number of cells and genes to subsample from the data
    # set either dimension to None to use all cells or all genes
    n_genes = 1000
    n_cells = None
    
    # specify the simulator to test
    model = simulators.NegBinCopula 
    
    # define formulas for the simulator
    formulas = {
        "mean_formula": "~ celltype + stage",
        "dispersion_formula": "~ celltype + stage",
        "copula_formula": "~ 1"
    }

    real_data, sim_data, metrics, _ = simulate(
        adata,
        n_cells=n_cells,
        n_genes=n_genes,
        model=model,
        formulas=formulas,
        top_k=None,
        subsample_seed=42,
        # training parameters below
        max_epochs=500,
        lr = 0.005
    )

    print(pd.DataFrame([metrics]).T)
