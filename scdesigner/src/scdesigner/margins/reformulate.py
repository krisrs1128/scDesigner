from torch import nn
import numpy as np
from .marginal import MarginalModel

def filter_marginal(model, genes):
    # get indices for the subset
    gene_names = model.module.gene_names
    ix = np.where(np.isin(gene_names, genes))[0]

    # define the filtered linear parameter
    linear = model.module.linear
    new_linear = {}
    for k, l in linear.items():
        new_linear[k] = nn.Linear(l.in_features, len(gene_names))
        new_linear[k].weight.data = l.weight.data[ix].clone()
        new_linear[k].bias.data = l.bias.data[ix].clone()

    # update module definition
    model.module.linear = new_linear
    model.module.gene_names = genes
    return model

def reformulate(model, genes, formula, anndata=None):
    """
    Modify the formula for a subset of genes 
    """
    # keep the model for unchanged genes
    complement = list(set(model.module.gene_names) - set(genes))
    submodel = filter_marginal(model, complement)

    # new marginal for the changed genes
    new_model = MarginalModel(formula, type(submodel.module))
    new_model.parameter_names = submodel.parameter_names
    if anndata is not None:
        new_model.configure_module(anndata[:, genes])

    return new_model, submodel