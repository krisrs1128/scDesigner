from scdesigner.margins.marginal import Marginal
from formulaic import Formula
import numpy as np
import pandas as pd
import torch


class NegativeBinomial(Marginal):
    def __init__(self, formula, device="cpu"):
        super().__init__(formula)
        self.formula = formula
        self.device = device
        self.parameters = {
            "alpha": None,
            "beta": None
        }


    def initialize(self, D, G):
        beta = torch.normal(
            0.0, 0.1, size=(D, G), requires_grad=True, device=self.device
        )
        logalpha = torch.rand(G, requires_grad=True, device=self.device)
        return logalpha, beta


    def design(self, X):
        X = Formula(self.formula).get_model_matrix(X, output="numpy")
        X = np.array(X).astype(np.float32)
        return torch.from_numpy(X)


    def fit(self, Y, X=None, max_iter=50, lr=1e-1):
        if X is None:
            X = pd.DataFrame({"intercept": np.ones((Y.shape[0]))})
        X = self.design(X)

        def newton_closure():
            optim.zero_grad()
            ll = -self.loglikelihood(logalpha, beta, Y, X)
            ll.backward()
            return ll

        X = X.to(self.device)
        Y = Y.to(self.device)
    
        logalpha, beta = self.initialize(X.shape[1], Y.shape[1])
        optim = torch.optim.LBFGS([logalpha, beta], lr=lr)
        for _ in range(max_iter):
            optim.step(newton_closure)

        print(beta.shape)
        self.parameters["beta"] = beta
        self.parameters["alpha"] = torch.exp(logalpha)


    def loglikelihood(self, logalpha, beta, Y, X, eps=1e-6):
        alpha = torch.exp(logalpha)
        ones = torch.ones(Y.shape).to(self.device)
        alpha_mu = alpha * torch.exp(X @ beta)

        return (
            Y * torch.log(alpha_mu + eps)
            - (1 / alpha + Y) * torch.log(1 + alpha_mu)
            + torch.lgamma(1 / alpha + Y)
            - torch.lgamma(1 + Y)
            - ones * torch.lgamma(1 / alpha)
        ).mean()

