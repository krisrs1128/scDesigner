from scdesigner.margins.marginal import Marginal
from scdesigner.design import design
import torch

def default_device(device):
    if device is not None:
        return device
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


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


    def fit(self, Y, X=None, max_iter=50, lr=1e-1):
        def newton_closure():
            optim.zero_grad()
            ll = -self.loglikelihood(A, B, Y, Xs)
            ll.backward()
            return ll

        Xs = {k: design(f, X, Y).to(self.device) for k, f in self.formula.items()}
        Y = Y.to(self.device)
    
        A, B = self.initialize(Xs, Y.shape[1])
        optim = torch.optim.LBFGS([A, B], lr=lr)
        for _ in range(max_iter):
            optim.step(newton_closure)

        self.parameters["B"] = B.detach().cpu()
        self.parameters["A"] = A.detach().cpu()


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
