import numpy as np
import torch
import torch.random
import torch.optim
from scdesigner.margins.negative_binomial import NegativeBinomial

def test_nb_mean():
    # define ground truth mean
    n_samples, n_genes = 1000, 20
    alpha = np.random.uniform(2.5, 5, n_genes)
    mu = np.ones((n_samples, 1)) @ np.exp(np.random.normal(size=(1, n_genes)))

    # generate samples
    Y = np.random.negative_binomial(1 / alpha, 1 / (1 + alpha * mu))
    Y = torch.from_numpy(Y)

    # estimate means
    nb_model = NegativeBinomial("~ 1")
    nb_model.fit(Y)
    mu_hat = nb_model.parameters["beta"]
    mu = torch.Tensor(mu[0]).log()
    assert torch.mean(torch.abs(mu_hat - mu)) < 0.1

def test_nb_coefs():
    pass

def test_nb_dispersion():
    pass

def test_nb_regression():
    pass