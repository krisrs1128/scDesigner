import time
import tracemalloc
import inspect
import numpy as np
import pandas as pd
import torch
import anndata as ad
from scdesigner.simulators import (
    NegBinCopula, 
    ZeroInflatedNegBinCopula,
    BernoulliCopula,
    GaussianCopula,
    PoissonCopula, 
    ZeroInflatedPoissonCopula,
    NegBinIRLSCopula,
)

FUNC_MAP = {
    "nb": NegBinCopula,
    "poisson": PoissonCopula,
    "bernoulli": BernoulliCopula,
    "gaussian": GaussianCopula,
    "zinb": ZeroInflatedNegBinCopula,
    "zip": ZeroInflatedPoissonCopula,
    "nbirls": NegBinIRLSCopula,
}

def subsample(adata, ncell, ngene):
    """Randomly subsample cells and genes from an AnnData object."""
    cidx = np.random.choice(adata.n_obs, ncell, replace=False)
    gidx = np.random.choice(adata.n_vars, ngene, replace=False)
    return adata.copy()[cidx, gidx]

def prepare_formulas(formula_params, init_params):
    """Prepare formulas for the simulator based on accepted parameters."""
    valid_params = {}
    used_formulas = {}
    
    for key, value in formula_params.items():
        if key in init_params:
            valid_params[key] = f"~ {value}" if value is not None else value
            used_formulas[key] = value
        else:
            used_formulas[key] = None
            
    return valid_params, used_formulas

def simulate(
    adata,
    ncell,
    ngene,
    mean_formula='1',
    dispersion_formula='1',
    zero_inflation_formula='1',
    sdev_formula='1',
    copula_formula='1',
    model='zinb',
    top_k=None,
    lr=0.005,
    epochs=500,
    batch_size=1024,
    chunk_size=5000,
):
    """
    Fit a scDesigner model and sample simulated data with resource monitoring.
    Note: The peak RAM is not the total RAM used by the process, 
    but the peak RAM used by python interpreter. 
    The actual consumed RAM could be larger.
    
    Returns:
        tuple: (real_data, sim_data, metrics_df, simulator)
    """
    real_data = subsample(adata, ncell, ngene)
    
    simulator_class = FUNC_MAP[model]
    init_params = inspect.signature(simulator_class.__init__).parameters
    
    formula_params = {
        'mean_formula': mean_formula,
        'dispersion_formula': dispersion_formula,
        'zero_inflation_formula': zero_inflation_formula,
        'copula_formula': copula_formula,
        'sdev_formula': sdev_formula,
    }
    
    valid_params, used_formulas = prepare_formulas(formula_params, init_params)
    
    tracemalloc.start()
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    simulator = simulator_class(**valid_params)
    simulator.fit(
        real_data, 
        max_epochs=epochs, 
        lr=lr, 
        batch_size=batch_size, 
        chunk_size=chunk_size, 
        top_k=top_k,
    )
    sim_data = simulator.sample(real_data.obs)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    if torch.cuda.is_available():
        peak_gpu_mib = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_gpu_mib = np.nan
    
    total_ram_mib = current_mem / 1024**2
    peak_ram_mib = peak_mem / 1024**2
    runtime = time.time() - start_time

    try: 
        diagnose = simulator.complexity()
        aic, bic = diagnose['aic'], diagnose['bic']
    except Exception as e:
        print(f"Error calculating complexity: {e}")
        aic, bic = np.nan, np.nan

    metrics = {
        "ncell": ncell,
        "ngene": ngene,
        "runtime": runtime,
        "peak_RAM": peak_ram_mib, # The peak memory usage during the simulation.
        "total_RAM": total_ram_mib, # The memory alloacted after running the simulation.
        "peak_GPU": peak_gpu_mib,
        "model": model,
        "simulator": "scdesigner",
        "epochs": epochs,
        "lr": lr,
        "mu": used_formulas.get('mean_formula'),
        "sigma": used_formulas.get('dispersion_formula'),
        "zero_inflation": used_formulas.get('zero_inflation_formula'),
        "copula": used_formulas.get('copula_formula'),
        "sdev": used_formulas.get('sdev_formula'),
        "aic": aic,
        "bic": bic
    }

    return real_data, sim_data, pd.DataFrame([metrics]), simulator

if __name__ == "__main__":

    n_genes = 2000
    n_cells = 2000

    # We test the performance up to 5 covariates for both mean and dispersion
    mean_formula = "celltype + stage + theiler"
    dispersion_formula = "celltype + stage + theiler"
    
    data_dir = "../data/HVG_embryoatlas.h5ad"
    adata = ad.read_h5ad(data_dir)

    _, _, metrics_df, _ = simulate(adata, 
    ncell=n_cells, 
    ngene=n_genes, 
    mean_formula=mean_formula,
    dispersion_formula=dispersion_formula,
    top_k=None,
    model='zinb',
    epochs = 50)
    print(metrics_df)
