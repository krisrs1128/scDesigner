from marginal import Marginal
import numpy as np
import torch


class NB(Marginal):
    def __init__(self, formula):
        super().__init__()
        self.formula = formula
        self.alpha = None
        self.beta = None
        self.mu = None
        self.device = "mps"


    def initialize(self, D, G):
        B = torch.normal(
            0.0, 0.1, size=(D, G), requires_grad=True, device=self.device
        )
        logalpha = torch.rand(G, requires_grad=True, device=self.device)
        return logalpha, B


    def fit(self, Y, X, optimizer=None, max_iter=5):
        if optimizer is None:
            optimizer = torch.optim.LBFGS

        def newton_closure():
            optimizer.zero_grad()
            ll = -self.loglikelihood(X, Y)
            ll.backward()
            return ll

        X = X.to(self.device)
        Y = Y.to(self.device)
    
        logalpha, B = self.initialize(X.shape[1], Y.shape[1])
        optim = optimizer([logalpha, B])
        for _ in range(max_iter):
            optim.step(newton_closure)

        self.beta = B
        self.alpha = np.exp(logalpha)


    def loglikelihood(self, Y, X, eps=1e-6):
        ones = torch.ones(Y.shape).to(self.device)
        alpha_mu = self.alpha * torch.exp(X @ self.beta)

        return (
            Y * torch.log(alpha_mu + eps)
            - (1 / self.alpha + Y) * torch.log(1 + alpha_mu)
            + torch.lgamma(1 / self.alpha + Y)
            - torch.lgamma(1 + Y)
            - ones * torch.lgamma(1 / self.alpha)
        ).mean()

