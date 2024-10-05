import numpy as np
import pandas as pd
import torch
import torch.distributions
from scipy.stats import norm
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import UniformUnivariate

def cov2cor(c):
    D = 1 / np.sqrt(np.diag(c))
    return D * c * D

class ScCopula:
    def __init__(self, copula_type=GaussianMultivariate, cov_fun=np.cov):
        self.copula = copula_type(distribution=UniformUnivariate)
        self.cov_fun = cov_fun

    def sample(self, margins, obs):
        u = self.copula.sample(len(obs))
        var_names, counts = [], []
        for genes, margin in margins:
            var_names += list(genes)
            u_ = torch.from_numpy(np.array(u[genes]))
            counts.append(margin.icdf(u_, obs).numpy())

        return var_names, counts

    def fit(self, margins, anndata):
        # define the uniform data for estimation
        u, var_names = [], []
        for genes, margin in margins:
            var_names += list(genes)
            X = torch.from_numpy(anndata[:, genes].X)
            u_ = margin.cdf(X, anndata.obs).numpy()
            u.append(np.clip(u_, 0.0001, 0.9999)) # avoid nan in inverse quantiles

        # fit the copula
        u = pd.DataFrame(np.concatenate(u, axis=1), columns=var_names)
        self.copula.fit(u)

        # apply custom covariance estimator
        if self.cov_fun is not np.cov:
            z = norm.ppf(u)
            self.copula.covariance = self.cov_fun(z.T)
            self.copula.correlation = cov2cor(self.copula.covariance)
