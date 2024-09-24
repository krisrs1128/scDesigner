from ..margins.marginal import Marginal
from .. import design as ds
from ..margins import parameter as pm
import pandas as pd
import rich
import rich.table
import torch


class NegativeBinomial(Marginal):
    def __init__(self, formula, device=None):
        super().__init__(formula)
        self.device = pm.default_device(device)
        self.parameters = {"B": None, "A": None}

        if isinstance(formula, str):
            formula = {"mu": formula}
        self.formula = ds.initialize_formula(formula)

    def initialize(self, Xs, G):
        A = torch.normal(
            0.0,
            0.1,
            size=(Xs["alpha"].shape[1], G),
            requires_grad=True,
            device=self.device,
        )
        B = torch.normal(
            0.0,
            0.1,
            size=(Xs["mu"].shape[1], G),
            requires_grad=True,
            device=self.device,
        )
        return A, B

    def fit(self, Y, X=None, y_names=None, max_iter=10, lr=0.1):
        if y_names is None:
            y_names = range(Y.shape[1])

        def newton_closure():
            optim.zero_grad()
            ll = -self.loglikelihood(A, B, Y, Xs)
            ll.backward()
            return ll

        # get design matrix from the model formula
        designs = {k: ds.design(f, X, Y.shape[0]) for k, f in self.formula.items()}
        Xs = {k: v[0].to(self.device) for k, v in designs.items()}
        Y = Y.to(self.device)

        # optimize mean and dispersion parameters
        A, B = self.initialize(Xs, Y.shape[1])
        optim = torch.optim.LBFGS([A, B], lr=lr)
        for _ in range(max_iter):
            optim.step(newton_closure)

        # for easier interpretation, process into named data.frames
        self.parameters["B"] = pm.parameter_to_df(B, y_names, designs["mu"][1])
        self.parameters["A"] = pm.parameter_to_df(A, y_names, designs["alpha"][1])

    def predict(self, X):
        # process inputs and parameters into tensors on device
        Xs = {k: ds.design(f, X)[0] for k, f in self.formula.items()}
        Xs = {k: v.to(self.device) for k, v in Xs.items()}
        A = pm.parameter_to_tensor(self.parameters["A"], self.device)
        B = pm.parameter_to_tensor(self.parameters["B"], self.device)

        # generate and give names to predictions
        mu_hat = pd.DataFrame(
            torch.exp(Xs["mu"] @ B).cpu(), columns=self.parameters["B"].columns
        )
        alpha_hat = pd.DataFrame(
            torch.exp(Xs["alpha"] @ A).cpu(), columns=self.parameters["A"].columns
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
        fmla = ds.reconcile_formulas(self.formula)
        return pd.DataFrame(
            {"formula": fmla, "distribution": "NegativeBinomial(\u03BC, \u03B1)"},
            index=[0],
        )

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
