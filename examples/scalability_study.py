import argparse
import numpy as np
import pandas as pd
import anndata
import time
import torch
from scdesigner.margins.marginal import NB
from scdesigner.simulator import scdesigner

def main(config):
    # setup the configuration
    configurations = pd.read_csv("data/scalability_configurations.csv")
    n_cell, n_gene, replicate = configurations.iloc[config, :].values
    n_cell = int(n_cell)
    n_gene = int(n_gene)
    
    # run the simulation
    np.random.seed(config)
    torch.set_float32_matmul_precision('medium')
    sce = anndata.read_h5ad("data/million_cells.h5ad", backed=True)
    
    # time the simulation
    start = time.time()
    sim = scdesigner(sce, NB("~ cell_type + `CoVID-19 severity`"), multivariate=None, max_epochs=5, lr=1e-2, batch_size=int(1e3), num_workers=4)
    delta = time.time() - start
    
    # save the timing results
    pd.DataFrame({
        "n_gene": n_gene,
        "n_cell": n_cell,
        "replicate": replicate,
        "seconds": delta
    }, index=[0]).to_csv(f"scdesigner_timing_{replicate}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest='config', type=int, help='Which row of scalability_configurations.csv to run?')
    args = parser.parse_args()
    main(args.config)
