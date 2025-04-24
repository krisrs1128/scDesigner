import anndata
from .. import estimators as est
from .. import predictors as prd
from .. import samplers as smp
import pandas as pd


class CompositeGLMSimulator:
    def __init__(self, specification: dict, **kwargs):
        self.specification = specification
        self.params = {}
        self.hyperparams = kwargs

    def fit(self, adata: anndata.AnnData) -> dict:
        for k, spec in self.specification.items():
            match spec["distribution"]:
                case "negbin":
                    self.params[k] = est.negbin_regression(
                        adata[:, spec["var_names"]], spec["formula"], **self.hyperparams
                    )
                case "poisson":
                    self.params[k] = est.poisson_regression(
                        adata[:, spec["var_names"]], spec["formula"], **self.hyperparams
                    )

    def sample(self, obs: pd.DataFrame) -> anndata.AnnData:
        anndata_list = []
        local_parameters = self.predict(obs)

        for k, spec in self.specification.items():
            match spec["distribution"]:
                case "negbin":
                    anndata_list.append(smp.negbin_sample(local_parameters[k], obs))
                case "poisson":
                    anndata_list.append(smp.poisson_sample(local_parameters[k], obs))

        return anndata.concat(anndata_list, axis="var")

    def predict(self, obs: pd.DataFrame) -> dict:
        preds = {}
        for k, spec in self.specification.items():
            match spec["distribution"]:
                case "negbin":
                    preds[k] = prd.negbin_predict(self.params[k], obs, spec["formula"])
                case "poisson":
                    preds[k] = prd.poisson_predict(self.params[k], obs, spec["formula"])

        return preds

    def __repr__(self):
        spec = {
            k: (v["distribution"], v["formula"], list_string(v["var_names"]))
            for k, v in self.specification.items()
        }
        return f"""scDesigner simulator object with
    method: 'Composite'
    specification: {spec}
    parameters: {nested_keys(self.params)}"""


def nested_keys(d, parent_key=""):
    keys = []
    for k, v in d.items():
        full_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            keys.extend(nested_keys(v, full_key))
        else:
            keys.append(full_key)
    return keys


def list_string(l):
    if len(l) <= 3:
        return ", ".join(l)
    return f"[{l[0]},{l[1]}, ..., {l[-1]}]"
