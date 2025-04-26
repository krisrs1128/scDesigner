import anndata
import pandas as pd


class CompositeGLMSimulator:
    def __init__(self, specification: dict, **kwargs):
        self.specification = specification
        self.params = {}
        self.hyperparams = kwargs

        for k in self.specification:
            if "_fitted" not in self.specification[k].keys():
                self.specification[k]["_fitted"] = False


    def fit(self, adata: anndata.AnnData) -> dict:
        self.specification = fill_var_names(self.specification, list(adata.var_names))

        for k, spec in self.specification.items():
            if not spec["_fitted"]:
                spec["simulator"].fit(adata[:, spec["var_names"]], spec["formula"])
            self.params[k] = subset_params(spec["simulator"].params, list(spec["var_names"]))
            self.specification[k]["simulator"].params = self.params[k]
            self.specification[k]["_fitted"] = True

    def sample(self, obs: pd.DataFrame) -> anndata.AnnData:
        anndata_list = []
        for spec in self.specification.values():
            anndata_list.append(spec["simulator"].sample(obs))
        return anndata.concat(anndata_list, axis="var")

    def predict(self, obs: pd.DataFrame) -> dict:
        preds = {}
        for k, spec in self.specification.items():
            preds[k] = spec["simulator"].predict(obs)
        return preds

    def __repr__(self):
        var_names = {k: list_string(v["var_names"]) for k, v in self.specification.items()}
        simulators = {k: v["simulator"] for k, v in self.specification.items()}
        return f"""scDesigner simulator object with
    method: 'Composite'
    features: {var_names}
    simulators: {simulators}"""


def list_string(l):
    if l is None:
        return
    if len(l) <= 3:
        return ", ".join(l)
    return f"[{l[0]},{l[1]}, ..., {l[-1]}]"

def fill_var_names(specification, var_names):
    all_names = []
    for k, v in specification.items():
        if v["var_names"] is not None:
            all_names += list(v["var_names"])

    for k, v in specification.items():
        if v["var_names"] is None:
            specification[k]["var_names"] = list(set(var_names) - set(all_names))
    return specification

def subset_params(params, var_names):
    result = {}
    for k, v in params.items():
        if "covariance" in k:
            result[k] = v.loc[var_names, var_names]
        else:
            result[k] = v.loc[:, var_names]
    return result