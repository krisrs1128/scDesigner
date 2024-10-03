import numpy as np
import pandas as pd
import torch
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import UniformUnivariate

class ScCopula:
    def __init__(self, copula_type=GaussianMultivariate, corr_fun=np.cov):
        self.copula = copula_type(distribution=UniformUnivariate)
        self.corr_fun = corr_fun

    def sample(self, margins, obs):
        u = self.copula.sample(len(obs))
        var_names, counts = [], []
        for genes, margin in margins:
            var_names += list(genes)
            u_ = torch.from_numpy(np.array(u[genes]))
            counts.append(margin.icdf(u_, obs).numpy())

        return var_names, counts

    def fit(self, margins, anndata):
        Sigma = self.corr_fun(anndata.X)
        self.copula.covariance = Sigma

        # define the uniform data for estimation
        u, var_names = [], []
        for genes, margin in margins:
            var_names += list(genes)
            X = torch.from_numpy(anndata[:, genes].X)
            u.append(margin.cdf(X, anndata.obs).numpy())

        # estimate the copula
        u = pd.DataFrame(np.concatenate(u, axis=1), columns=genes)
        self.copula.fit(u)