import numpy as np
import pandas as pd
import torch
import torch.distributions
from copy import deepcopy
from itertools import chain
from scdesigner.formula import formula_to_groups
from scipy.stats import norm
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import UniformUnivariate

def cov2cor(c):
    D = 1 / np.sqrt(np.diag(c))
    return D * c * D

def margins_invert(margins, u, obs):
    var_names, counts = [], []
    for genes, margin in margins:
        var_names += list(genes)
        u_ = torch.from_numpy(np.array(u[genes]))
        counts.append(margin.icdf(u_, obs).numpy())
    return var_names, counts


def margins_uniformize(margins, anndata):
    u, var_names = [], []
    for genes, margin in margins:
        var_names += list(genes)
        X = torch.from_numpy(anndata[:, genes].X)
        u_ = margin.cdf(X, anndata.obs).numpy()
        u.append(np.clip(u_, 0.0001, 0.9999)) # avoid nan in inverse quantiles
    return pd.DataFrame(np.concatenate(u, axis=1), columns=var_names)


class ScCopula:
    def __init__(self, formula=None, copula_type=GaussianMultivariate, cov_fun=np.cov, **kwargs):
        if formula is None:
            copula = CopulaFixed(copula_type, cov_fun, **kwargs)
        else:
            copula = CopulaFormula(formula, copula_type, cov_fun, **kwargs)

        self.__class__ = copula.__class__
        self.__dict__ = copula.__dict__


class CopulaFixed:
    def __init__(self, copula_type=GaussianMultivariate, cov_fun=np.cov):
        self.copula = copula_type(distribution=UniformUnivariate)
        self.cov_fun = cov_fun

    def sample(self, margins, obs):
        u = self.copula.sample(len(obs))
        return margins_invert(margins, u, obs)

    def fit(self, margins, anndata):
        # fit the copula
        u = margins_uniformize(margins, anndata)
        self.copula.fit(u)

        # apply custom covariance estimator
        if self.cov_fun is not np.cov:
            z = norm.ppf(u)
            self.copula.covariance = self.cov_fun(z.T)
            self.copula.correlation = cov2cor(self.copula.covariance)

class CopulaFormula:
    def __init__(self, formula, copula_type=GaussianMultivariate, cov_fun=np.cov):
        self.copulas = copula_type(distribution=UniformUnivariate)
        self.cov_fun = cov_fun

        # extract formula for group labels
        if isinstance(formula, dict):
            formula = formula["copula"]
        self.formula = formula

    def sample(self, margins, obs):
        ix, K = formula_to_groups(self.formula, obs)
        gene_names = list(chain(*[g for g, _ in margins]))

        # sample copulas one group at a time
        u = np.zeros((len(obs), len(gene_names)))
        for k in range(K):
            u[ix == k, :] = self.copulas[k].sample(sum(ix == k))
        u = pd.DataFrame(u, columns=gene_names)
        return margins_invert(margins, u, obs)

    def fit(self, margins, anndata):
        # initialize the data structures 
        u = margins_uniformize(margins, anndata)
        ix, K = formula_to_groups(self.formula, anndata.obs)
        self.copulas = [deepcopy(self.copulas) for _ in range(K)]

        # fit one copula per group
        for k in range(K):
            self.copulas[k].fit(u.iloc[ix == k, :])
            if self.cov_fun is not np.cov:
                z = norm.ppf(u.iloc[ix == k, :])
                self.copulas[k].covariance = self.cov_fun(z.T)
                self.copulas[k].correlation = cov2cor(self.copula.covariance)