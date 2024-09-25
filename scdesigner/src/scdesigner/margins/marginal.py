import torch
import torch.optim
from .. import design as ds
from . import parameter as pm

def gaussian_initializer(Ds, G, device, sigma=0.1):
    return {k: torch.normal(
        0.0, sigma, 
        size=(D, G), 
        requires_grad=True, device=device
    ) for k, D in Ds.items()}

def predictors(X, formula, n_features, device):
    designs = {k: ds.design(f, X, n_features) for k, f in formula.items()}
    return {k: v[0].to(device) for k, v in designs.items()}

class Marginal:
    def __init__(self, formula, device=None):
        super().__init__()
        self.theta = None
        self.formula = ds.initialize_formula(formula)
        self.device = pm.default_device(device)

    def configure_optimizers(self, lr=0.1, **kwargs):
        return torch.optim.LBFGS(self.theta.values(), lr=lr, **kwargs)

    def optimizer_step(self, Y, Xs, optimizer):
        def newton_closure():
            optimizer.zero_grad()
            ll = -self.loglikelihood(Y, Xs)
            ll.backward()
            return ll
    
        optimizer.step(newton_closure)

    def fit(self, Y, X=None, max_iter=10, initializer=gaussian_initializer, **kwargs):
        Xs = predictors(X, self.formula, Y.shape[1], self.device)
        Ds = {k: v.shape[1] for k, v in Xs.items()}
        self.theta = initializer(Ds, Y.shape[1], self.device)
        optimizer = self.configure_optimizers(**kwargs)
    
        for _ in range(max_iter):
            self.optimizer_step(Y, Xs, optimizer)

    def sample(self, Y, Xs=None, **kargs):
        pass

    def logliklihood(self, Y, Xs=None):
        pass

    def quantile(self, Y, q):
        pass
