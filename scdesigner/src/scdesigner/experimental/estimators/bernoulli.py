from . import gaussian_copula_factory as gcf
from . import glm_factory as factory
from . import format
from . import poisson as poi
from anndata import AnnData
from formulaic import model_matrix
from scipy.stats import bernoulli
import numpy as np
import torch

###############################################################################
## Regression functions that operate on numpy arrays
###############################################################################


def bernoulli_regression_likelihood(params, X, y):
    # get appropriate parameter shape
    n_features = X.shape[1]
    n_outcomes = y.shape[1]

    # compute the negative log likelihood
    beta = params.reshape(n_features, n_outcomes)
    theta = torch.sigmoid(X @ beta)
    log_likelihood = y * torch.log(theta) + (1 - y) * torch.log(1 - theta)
    return -torch.sum(log_likelihood)


bernoulli_regression_array = factory.glm_regression_generator(
    bernoulli_regression_likelihood, poi.poisson_initializer, poi.poisson_postprocessor
)

###############################################################################
## Regression functions that operate on AnnData objects
###############################################################################


def bernoulli_regression(adata: AnnData, formula: str, **kwargs) -> dict:
    adata = format.format_input_anndata(adata)
    x = model_matrix(formula, adata.obs)
    parameters = bernoulli_regression_array(np.array(x), adata.X, **kwargs)
    return poi.format_poisson_parameters(
        parameters, list(adata.var_names), list(x.columns)
    )


###############################################################################
## Copula versions for bernoulli regression
###############################################################################


def bernoulli_uniformizer(parameters, x, y):
    theta = torch.sigmoid(torch.from_numpy(x @ parameters["beta"]))
    nb_distn = bernoulli(theta.numpy())
    alpha = np.random.uniform(size=y.shape)
    return gcf.clip(alpha * nb_distn.cdf(y) + (1 - alpha) * nb_distn.cdf(1 + y))


bernoulli_copula = gcf.gaussian_copula_factory(
    gcf.gaussian_copula_array_factory(bernoulli_regression_array, bernoulli_uniformizer),
    poi.format_poisson_parameters
)
