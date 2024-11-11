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
    
    # load and subsample the data
    print("reading the backed anndata...")
    np.random.seed(config)
    torch.set_float32_matmul_precision("medium")
    sce = anndata.read_h5ad("data/million_cells.h5ad", backed="r")

    print("copying into a subset...")
    cell_ix = np.random.choice(n_cell, n_cell, replace=False)
    gene_ix = np.random.choice(n_gene, n_gene, replace=False)
    sce[cell_ix, gene_ix].copy(filename="subset_tmp.h5ad")
    print("reading the subset...")
    if n_cell > 1e4 or n_gene > 1e4:
        sce = anndata.read_h5ad("subset_tmp.h5ad", backed="r")
    else:
        sce = anndata.read_h5ad("subset_tmp.h5ad")
    
    # time the estimation step
    print("estimation start...")
    start = time.time()
    scdesigner(sce, NB("~ cell_type + `CoVID-19 severity`"), multivariate=None, batch_size=int(1e3), lr=0.01, max_epochs=100)
    delta = time.time() - start
    
    # save the timing results
    pd.DataFrame({
        "n_gene": n_gene,
        "n_cell": n_cell,
        "replicate": replicate,
        "seconds": delta
    }, index=[0]).to_csv(f"scdesigner_timing_{int(config)}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", type=int, help="Which row of scalability_configurations.csv to run?")
    args = parser.parse_args()
    main(args.config)
