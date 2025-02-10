import numpy as np
from formulaic import Formula, model_matrix
import anndata

def str_match(string, string_list):
    for l in string_list:
        if l.contains(string):
            return True
    return False


def anndata_formula_mask(outcomes: list, inputs: list, formula: Formula, adata: anndata.AnnData, subset: int=10):
    """
    Mask for Queried Parameters 
    """
    response_index = [adata.var_names[i] in outcomes for i in range(len(adata.var_names))]
    features = model_matrix(formula, adata.obs.iloc[:subset, :]).columns
    feature_index = [str_match(features[i], inputs) for i in range(len(features))]

    mask = np.zeros((len(adata.var_names), len(features)))
    mask[response_index, feature_index] = 1
    return mask