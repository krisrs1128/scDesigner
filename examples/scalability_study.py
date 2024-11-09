import argparse
import numpy as np
import pandas as pd
import anndata
import time
from scdesigner.margins.marginal import NB
from scdesigner.simulator import scdesigner

def main(config):
    # setup the configuration
    configurations = pd.read_csv("data/scalability_configurations.csv")
    n_cell, n_gene, replicate = configurations.iloc[config, :].values
    n_cell = int(n_cell)
    n_gene = int(n_gene)
    
    n_gene = config['n_gene']
    n_cell = config['n_cell']
    replicate = config['replicate']
    
    # run the simulation
    np.random.seed(config)
    sce = anndata.read_h5ad("data/million_cells.h5ad", backed=True)
    
    # time the simulation
    start = time.time()
    sim = scdesigner(sce, NB("~ cell_type + `CoVID-19 severity`"), multivariate=None, max_epochs=5, lr=1e-2)
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
    parser.add_argument('config', type=int, help='Which row of scalability_configurations.csv to run?')
    args = parser.parse_args()
    main(args.config)
