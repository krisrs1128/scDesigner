from anndata import AnnData
from scipy.stats import nbinom, norm
from formulaic import model_matrix
from .nb_regression import (
    format_nb_parameters,
    negative_binomial_regression_array,
)
import scipy.sparse
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

        r, mu, u = negative_binomial_copula_sample_array(parameters, x, groups)
        samples = nbinom(n=r, p=r / (r + mu)).ppf(u)
        result = AnnData(X=samples, obs=obs)
        result.var_names = parameters["dispersion"].columns
        return result

    def predict(
        self, parameters: dict, obs: pd.DataFrame, formula="~ 1", formula_copula="~ 1"
    ) -> dict:
        x = model_matrix(formula, obs)
        groups = group_indices(formula_copula, obs)
        r, mu, _ = negative_binomial_copula_sample_array(parameters, x, groups)
        return {
            "coefficient": mu,
            "dispersion": r,
            "covariance": parameters["covariance"],
        }

    def __str__(self):
        return f"""scDesigner object with n_obs x n_vars = {self.shape[0]} x {self.shape[1]}
    method: 'NBCopula'
    formula: '{self.formula}'
    copula formula: '{self.copula_formula}'
    parameters: 'coefficient', 'dispersion', 'covariance'"""


def negative_binomial_copula_array(
    x: np.array,
    y: np.array,
    groups: dict,
    batch_size: int = 512,
    lr: float = 0.1,
    epochs: int = 20,
) -> dict:
    """
    A minimal NB copula model

    # simulate data
    n_samples, n_features, n_outcomes = 1000, 2, 4
    x_sim = np.random.normal(size=(n_samples, n_features))
    beta_sim = np.random.normal(size=(n_features, n_outcomes))
    mu_sim = np.exp(x_sim @ beta_sim)
    r_sim = np.random.uniform(.5, 1.5, n_outcomes)
    y_sim = np.random.negative_binomial(r_sim, r_sim / (r_sim + mu_sim))
    y_sim[:, 1] = y_sim[:, 0]

    # estimate model
    negative_binomial_copula(x_sim, y_sim)
    """
    # get predicted mean and dispersions
    parameters = negative_binomial_regression_array(x, y, batch_size, lr, epochs)
    r, mu = np.exp(parameters["dispersion"]), np.exp(x @ parameters["coefficient"])
    nb_distn = nbinom(n=r, p=r / (r + mu))

    # gaussianize and estimate covariance
    alpha = np.random.uniform(size=y.shape)
    u = clip(alpha * nb_distn.cdf(y) + (1 - alpha) * nb_distn.cdf(1 + y))
    parameters["covariance"] = copula_covariance(u, groups)
    return parameters


def negative_binomial_copula_sample_array(
    parameters: dict, x: np.array, groups: dict
) -> np.array:
    # initialize uniformized gaussian samples
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

    # invert using negative binomial margins
    r, mu = np.exp(parameters["dispersion"]), np.exp(x @ parameters["coefficient"])
    r = np.repeat(r, mu.shape[0], axis=0)
    return r, mu, u


###############################################################################
## Helpers for fitting & sampling NB Copula
###############################################################################


def copula_covariance(u: np.array, groups: dict):
    result = {}
    for group, ix in groups.items():
        result[group] = np.cov(norm().ppf(u[ix]).T)

    if len(result) == 1:
        return list(result.values())[0]
    return result


def clip(u: np.array, min: float = 1e-5, max: float = 1 - 1e-5) -> np.array:
    u[u < min] = min
    u[u > max] = max
    return u


def format_input_anndata(adata: AnnData) -> AnnData:
    result = adata.copy()
    if isinstance(result.X, scipy.sparse._csc.csc_matrix):
        result.X = result.X.todense()
    return result


def format_copula_parameters(parameters: dict, var_names: list):
    covariance = parameters["covariance"]
    if type(covariance) is not dict:
        covariance = pd.DataFrame(
            parameters["covariance"], columns=list(var_names), index=list(var_names)
        )
    else:
        for group in covariance.keys():
            covariance[group] = pd.DataFrame(
                parameters["covariance"][group],
                columns=list(var_names),
                index=list(var_names),
            )
    return covariance


def group_indices(formula: str, obs: pd.DataFrame) -> dict:
    group_matrix = model_matrix(formula, obs)
    result = {}

    for group in group_matrix.columns:
        result[group] = np.where(group_matrix[group].values == 1)[0]
    return result
