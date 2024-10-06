import anndata as ad
import muon as mu
import numpy as np
import pandas as pd
from copy import deepcopy
from pamona import Pamona
from copulas.multivariate import GaussianMultivariate
from scdesigner import simulator as sim
from .copula import ScCopula
        
def update_margin_varnames(sim, i="1"):
    new_margins = []
    for genes, margin in sim.margins:
        new_genes = [f"source{i}_{g}" for g in genes]
        new_margins.append((new_genes, margin))
    return new_margins

def reconcile_names(adata, sims):
    if not (set(adata["s1"].var_names) & set(adata["s2"].var_names)):
        return adata, sims

    new_adata, new_sims = deepcopy(adata), deepcopy(sims)
    for k, v in adata.items():
        new_adata[k].var_names = [f"source{k}_{a}" for a in v.var_names]
        new_sims[k].margins = update_margin_varnames(new_sims[k], k)
    return new_adata, new_sims

def join_copula(sim1, sim2, copula_type=GaussianMultivariate, cov_fun=np.cov):
    sims = {"s1": sim1, "s2": sim2}
    adata = {"s1": sim1.anndata, "s2": sim2.anndata}
    adata, sims = reconcile_names(adata, sims)

    # combine features
    adata = ad.concat(adata.values(), axis=1, merge="same", label="modality_")
    margins = list(sims["s1"].margins + sims["s2"].margins)
    joined = sim.Simulator(margins, ScCopula(copula_type, cov_fun))

    # refit the copula
    joined.multivariate.fit(margins, adata)
    joined.anndata = adata
    joined.anndata.var 
    return joined

def join_pamona(sim1, sim2, multivariate=None, priority="mu", pamona_opts={}, **kwargs):
    sims = {"s1": sim1, "s2": sim2}
    adata = {"s1": sim1.anndata, "s2": sim2.anndata}
    adata, sims = reconcile_names(adata, sims)
    Pa = Pamona.Pamona(**pamona_opts)
    embeddings, _ = Pa.run_Pamona([adata["s1"].X, adata["s2"].X])

    # add the learned embedding information to the anndata
    dims = [f"pamona_{i}" for i in range(embeddings[0].shape[1])]
    embeddings = {
        "s1": pd.DataFrame(embeddings[0], columns=dims),
        "s2": pd.DataFrame(embeddings[1], columns=dims)
    }

    margins = []
    for k in adata.keys():
        obs = adata[k].obs.reset_index(drop=True)
        adata[k].obs = pd.concat([obs, embeddings[k]], axis=1)
        for genes, margin in sims[k].margins:
            new_margin = deepcopy(margin)
            if isinstance(margin.formula, str):
                new_margin.formula = f"{margin.formula} + {' + '.join(dims)}"
            else:
                new_margin.formula[priority] = f"{margin.formula[k]} + {' + '.join(dims)}"
            margins.append((genes, new_margin))

    # define and fit the simulator
    joined = sim.Simulator(margins, multivariate)
    adata = ad.concat(adata.values(), join="outer", label="modality_", axis=0)
    joined.fit(adata, **kwargs)
    return joined
    

def split_adata(adata):
    var_names = adata.var.groupby("modality_")\
        .apply(lambda x: list(x.index))\
        .to_dict()

    modalities = {}
    for k, v in var_names.items():
        modalities[k] = adata[:, adata.var_names.isin(v)].copy()

    return mu.MuData(modalities)