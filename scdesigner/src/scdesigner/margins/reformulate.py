from torch import nn
import numpy as np
from copy import deepcopy
from .marginal import MarginalModel


def filter_marginal(model, genes):
    # get indices for the subset
    gene_names = model.module.gene_names
    ix = np.where(np.isin(gene_names, genes))[0]

    # define the filtered linear parameter
    linear = model.module.linear
    new_linear = {}
    for k, l in linear.items():
        new_linear[k] = nn.Linear(l.in_features, len(genes), bias=False)
        new_linear[k].weight.data = l.weight.data[ix].clone()

    # update module definition
    m = deepcopy(model)
    m.module.linear = nn.ModuleDict(new_linear)
    m.module.gene_names = genes
    return m


def reformulate(model, genes, formula, anndata=None):
    """
    Modify the formula for a subset of genes
    """
    # keep the model for unchanged genes
    complement = [g for g in model.module.gene_names if g not in genes]
    submodel = filter_marginal(model, complement)

    # new marginal for the changed genes
    new_model = MarginalModel(formula, type(submodel.module))
    new_model.parameter_names = submodel.parameter_names
    if anndata is not None:
        new_model.configure_module(anndata[:, genes])

    return new_model, submodel


def match_marginal(margins, genes):
    result = []
    ix = []
    for i, (gene_subset, margin) in enumerate(margins):
        if np.any(np.isin(genes, gene_subset)):
            result.append((gene_subset, margin))
            ix.append(i)
    return ix, result
