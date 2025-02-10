import numpy as np
from formulaic import Formula, model_matrix
import anndata


def str_match(string: str, string_list: list) -> bool:
    for l in string_list:
        if l in string:
            return True
    return False


def anndata_formula_mask(
    outcomes: list,
    inputs: list,
    formula: Formula,
    adata: anndata.AnnData,
    subset: int = 10,
) -> np.array:
    """
    Mask for Queried Parameters
    """
    response_index = [
        adata.var_names[i] in outcomes for i in range(len(adata.var_names))
    ]
    features = list(model_matrix(formula, adata.obs.iloc[:subset, :]).columns)
    feature_index = [str_match(f, inputs) for f in features]

    mask = np.zeros((len(features), len(adata.var_names)))
    mask[np.ix_(feature_index, response_index)] = 1
    return mask.astype(np.int32)
