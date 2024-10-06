import anndata as ad
import numpy as np
import muon as mu
from copulas.multivariate import GaussianMultivariate
from scdesigner import simulator as sim
from .copula import ScCopula
        
def update_margins(sim, i="1"):
    new_margins = []
    for genes, margin in sim.margins:
        new_genes = [f"source{i}_{g}" for g in genes]
        new_margins.append((new_genes, margin))
    return new_margins

def join_copula(sim1, sim2, copula_type=GaussianMultivariate, cov_fun=np.cov):
    a1, a2 = sim1.anndata, sim2.anndata

    # reconcile variable names if there is overlap
    if set(a1.var_names) & set(a2.var_names):
        a1.var_names = [f"source1_{a}" for a in a1.var_names]
        a2.var_names = [f"source2_{a}" for a in a2.var_names]
        sim1.margins = update_margins(sim1)
        sim2.margins = update_margins(sim2, "2")

    # combine features
    adata = ad.concat([a1, a2], axis=1, merge="same", label="modality_")
    margins = list(sim1.margins + sim2.margins)
    joined = sim.Simulator(margins, ScCopula(copula_type, cov_fun))

    # refit the copula
    joined.multivariate.fit(margins, adata)
    joined.anndata = adata
    joined.anndata.var 
    return joined

def split_adata(adata):
    var_names = adata.var.groupby("modality_")\
        .apply(lambda x: list(x.index))\
        .to_dict()

    modalities = {}
    for k, v in var_names.items():
        modalities[k] = adata[:, adata.var_names.isin(v)].copy()

    return mu.MuData(modalities)