from anndata import AnnData
from scipy.stats import nbinom, norm
from formulaic import model_matrix
from ..estimators.glm_regression import (
    negative_binomial_copula_array,
    format_input_anndata,
    format_copula_parameters,
    format_nb_parameters,
    group_indices,
)
import numpy as np
import pandas as pd


class NegBinCopulaSimulator:
    def __init__(self):  # default input: cell x gene
        self.var_names = None
        self.formula = None
        self.copula_formula = None
        self.shape = None

    def estimate(
        self,
        adata: AnnData,
        formula: str = "~ 1",
        formula_copula: str = "~ 1",
        **kwargs,
    ) -> dict:
        adata = format_input_anndata(adata)
        self.formula = formula
        self.copula_formula = formula_copula
        self.shape = adata.X.shape
        x = model_matrix(formula, adata.obs)

        groups = group_indices(formula_copula, adata.obs)
        parameters = negative_binomial_copula_array(
            np.array(x), adata.X, groups, **kwargs
        )
        parameters = format_nb_parameters(
            parameters, list(adata.var_names), list(x.columns)
        )
        parameters["covariance"] = format_copula_parameters(
            parameters, list(adata.var_names)
        )
        return parameters

    def sample(
        self, parameters: dict, obs: pd.DataFrame, formula="~ 1", formula_copula="~ 1"
    ) -> AnnData:
        x = model_matrix(formula, obs)
        groups = group_indices(formula_copula, obs)

        G = parameters["coefficient"].shape[1]
        u = np.zeros((x.shape[0], G))

        # cycle across groups
        for group, ix in groups.items():
            if type(parameters["covariance"]) is not dict:
                parameters["covariance"] = {group: parameters["covariance"]}

            z = np.random.multivariate_normal(
                mean=np.zeros(G), cov=parameters["covariance"][group], size=len(ix)
            )
            normal_distn = norm(0, np.diag(parameters["covariance"][group] ** 0.5))
            u[ix] = normal_distn.cdf(z)

        r, mu = (
            self.predict(parameters, obs, formula)["dispersion"],
            self.predict(parameters, obs, formula)["coefficient"],
        )
        samples = nbinom(n=r, p=r / (r + mu)).ppf(u)
        result = AnnData(X=samples, obs=obs)
        result.var_names = parameters["dispersion"].columns
        return result

    def predict(self, parameters: dict, obs: pd.DataFrame, formula="~ 1") -> dict:
        x = model_matrix(formula, obs)
        r, mu = np.exp(parameters["dispersion"]), np.exp(x @ parameters["coefficient"])
        r = np.repeat(r, mu.shape[0], axis=0)
        return {
            "coefficient": mu,
            "dispersion": r,
        }

    def __repr__(self):
        return f"""scDesigner simulator object with
    method: 'Negtive Binomial Copula'
    formula: '{self.formula}'
    copula formula: '{self.copula_formula}'
    parameters: 'coefficient', 'dispersion', 'covariance'"""
