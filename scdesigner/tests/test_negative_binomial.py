from scdesigner.margins.negative_binomial import NegativeBinomial
import numpy as np
import pandas as pd
import torch
import torch.optim
import torch.random


def test_nb_mean():
    # define ground truth mean
    N, G = 1000, 20
    alpha = np.random.uniform(2.5, 5, G)
    mu = np.ones((N, 1)) @ np.exp(np.random.normal(size=(1, G)))

    # generate samples
    Y = np.random.negative_binomial(1 / alpha, 1 / (1 + alpha * mu))
    Y = torch.from_numpy(Y)

    # estimate means
    nb_model = NegativeBinomial("~ 1")
    nb_model.fit(Y)
    mu_hat = nb_model.parameters["B"].values
    assert np.mean(np.abs(mu_hat - np.log(mu[0]))) < 0.1


def test_mu_regression():
    # define ground truth coefficients
    N, G, D = 1000, 20, 4
    alpha = np.random.uniform(2.5, 5, G)
    X = np.random.normal(size=(N, D))
    B = np.random.normal(size=(D, G))
    mu = np.exp(X @ B)

    # generate samples
    Y = np.random.negative_binomial(1 / alpha, 1 / (1 + alpha * mu))
    Y = torch.from_numpy(Y)

    # estimate means
    X = pd.DataFrame(X, columns=[f"dim{j}" for j in range(D)])
    nb_model = NegativeBinomial("~ . - 1")
    nb_model.fit(Y, X, max_iter=50)
    Bhat = nb_model.parameters["B"].values
    assert np.mean(np.abs(Bhat - B)) < 0.1


def test_nb_dispersion():
    # define ground truth dispersion
    N, G = 1000, 20
    mu = np.random.uniform(2.5, 5, G)
    alpha = np.ones((N, 1)) @ np.exp(np.random.normal(size=(1, G)))

    # generate samples
    Y = np.random.negative_binomial(1 / alpha, 1 / (1 + alpha * mu))
    Y = torch.from_numpy(Y)

    # estimate dispersion parameters
    nb_model = NegativeBinomial({"alpha": "~ 1"})
    nb_model.fit(Y)
    Ahat = nb_model.parameters["A"].values
    assert np.mean(np.abs(Ahat - np.log(alpha[0]))) < 0.1


def test_alpha_regression():
    # define ground truth coefficients
    N, G, D = 1000, 20, 4
    mu = np.random.uniform(2, 5, G)
    X = np.random.normal(size=(N, D))
    X = pd.DataFrame(X, columns=[f"dim{j}" for j in range(D)])
    A = np.random.normal(size=(D, G))
    alpha = np.exp(X @ A)

    # generate samples
    Y = np.random.negative_binomial(1 / alpha, 1 / (1 + alpha * mu))
    Y = torch.from_numpy(Y)

    # estimate means
    nb_model = NegativeBinomial({"alpha": "~ . - 1"})
    nb_model.fit(Y, pd.DataFrame(X))
    Ahat = nb_model.parameters["A"].values
    assert np.mean(np.abs(Ahat - A)) < 0.1
