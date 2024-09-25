from ..margins.marginal import Marginal
from .. import design as ds
from ..margins import parameter as pm
import torch

def linear_predict(Xs, theta, g, device):
    Xs = {k: v.to(device) for k, v in Xs.items()}
    return {g[k](v.to(device)) for k, v in theta.items()}

def formula_df(fmla, fmla_str):
    return ""

class NB(Marginal):
    def __init__(self, formula, device=None):
        super().__init__(formula, device)
        self.theta = {"A": None, "B": None}
        self.g = {"B": torch.exp, "A": torch.exp}

    def predict(self, Xs):
        theta_hat = linear_predict(Xs, self.theta, self.g, self.device)
        theta_hat["alpha"] = theta_hat.pop("A")
        theta_hat["mu"] = theta_hat.pop("B")
        return theta_hat

    def loglikelihood(self, Y, Xs, eps=1e-6):
        theta = self.predict(Xs)
        theta["mu"] = theta["alpha"] * theta["mu"]

        return (
            Y * torch.log(theta["mu"] + eps)
            - (1 / theta["alpha"] + Y) * torch.log(1 + theta["mu"])
            + torch.lgamma(1 / theta["alpha"] + Y)
            - torch.lgamma(1 + Y)
            - torch.lgamma(1 / theta["alpha"])
        ).mean()

    def to_df(self):
        return formula_df(self.formula, "NegativeBinomial(\u03BC, \u03B1)")