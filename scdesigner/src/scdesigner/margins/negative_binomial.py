from ..margins.marginal import Marginal
from ..design import design
import numpy as np
import pandas as pd
import rich
import torch

def parameter_to_df(theta, y_names, x_names):
    theta = pd.DataFrame(theta.detach().cpu())
    theta.columns = y_names
    theta.index = x_names
    return theta


def parameter_to_tensor(theta, device):
    return torch.from_numpy(np.array(theta)).to(device)


def default_device(device):
    if device is not None:
        return device
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def reconcile_formulas(formula):
    values = formula.values()
    if len(set(values)) == 1:
        return f"""all: {formula["mu"]}"""
    return f"""mu: {formula["mu"]}, alpha: {formula["alpha"]}"""

def initialize_formula(f):
    parameters = ["alpha", "mu"]
    for k in parameters:
        if k not in f.keys():
            f[k] = "~ 1"
    return f

class NegativeBinomial(Marginal):
    def __init__(self, formula, device=None):
        super().__init__(formula)
        self.device = default_device(device)
        self.parameters = {"B": None, "A": None}

        if isinstance(formula, str):
            formula = {"mu": formula}
        self.formula = initialize_formula(formula)


    def initialize(self, Xs, G):
        A = torch.normal(0.0, 0.1, size=(Xs["alpha"].shape[1], G), requires_grad=True, device=self.device)
        B = torch.normal(0.0, 0.1, size=(Xs["mu"].shape[1], G), requires_grad=True, device=self.device)
        return A, B


    def fit(self, Y, X=None, y_names=None, max_iter=10, lr=1e-3):
        def newton_closure():
            optim.zero_grad()
            ll = -self.loglikelihood(A, B, Y, Xs)
            ll.backward()
            return ll

        designs = {k: design(f, X) for k, f in self.formula.items()}
        Xs = {k: v[0].to(self.device) for k, v in designs.items()}
        Y = Y.to(self.device)
    
        A, B = self.initialize(Xs, Y.shape[1])
        optim = torch.optim.LBFGS([A, B], lr=lr)
        for _ in range(max_iter):
            optim.step(newton_closure)

        self.parameters["B"] = parameter_to_df(B, y_names, designs["mu"][1])
        self.parameters["A"] = parameter_to_df(A, y_names, designs["alpha"][1])

    def predict(self, X):
        Xs = {k: design(f, X)[0] for k, f in self.formula.items()}
        Xs = {k: v.to(self.device) for k, v in Xs.items()}

        A = parameter_to_tensor(self.parameters["A"], self.device)
        B = parameter_to_tensor(self.parameters["B"], self.device)
        mu_hat = pd.DataFrame(
            torch.exp(Xs["mu"] @ B).cpu(),
            columns=self.parameters["B"].columns
        )
        alpha_hat = pd.DataFrame(
            torch.exp(Xs["alpha"] @ A).cpu(),
            columns=self.parameters["A"].columns
        )
        return {"mu": mu_hat, "alpha": alpha_hat}

    def loglikelihood(self, A, B, Y, Xs, eps=1e-6):
        alpha = torch.exp(Xs["alpha"] @ A)
        mu = torch.exp(Xs["mu"] @ B)
        alpha_mu = alpha * mu

        return (
            Y * torch.log(alpha_mu + eps)
            - (1 / alpha + Y) * torch.log(1 + alpha_mu)
            + torch.lgamma(1 / alpha + Y)
            - torch.lgamma(1 + Y)
            - torch.lgamma(1 / alpha)
        ).mean()

    def to_df(self):
        fmla = reconcile_formulas(self.formula)
        return pd.DataFrame({
            "formula": fmla,
            "distribution": "NegativeBinomial(\u03BC, \u03B1)"
        }, index=[0])

    def to_table(self):
        table = rich.table.Table(title="[bold magenta]Marginal Model[/bold magenta]")
        table.add_column("formula")
        table.add_column("distribution")
        table.add_row(*tuple(self.to_df().iloc[0, :]))
        return table

    def __repr__(self):
        rich.print(self.to_table())
        return ""

    def __str__(self):
        rich.print(self.to_table())
        return ""