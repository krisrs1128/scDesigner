import torch
import torch.utils.data as td
import numpy as np
from ..estimators import Estimator

class LinearMixedEffectsEstimator(Estimator):
    def __init__(self, lr: float = 0.01, max_iter: int=1000, init_sigma_b: torch.Tensor=None, init_sigma_e: torch.Tensor=None):
        self.lr = lr
        self.max_iter = max_iter
        self.init_sigma_b = init_sigma_b
        self.init_sigma_e = init_sigma_e

    def estimate(self, loader: td.DataLoader) -> dict:
        return linear_mixed_effects(loader, self.init_sigma_b, self.init_sigma_e, self.lr, self.max_iter)


def initialize_lme(n_predictors, n_responses, init_sigma_b, init_sigma_e):
    init_beta = torch.zeros((n_predictors, n_responses), dtype=torch.float32)
    if init_sigma_b is None:
        init_sigma_b = torch.ones(n_responses, dtype=torch.float32)
    if init_sigma_e is None:
        init_sigma_e = torch.ones(n_responses, dtype=torch.float32)
    
    init_params = torch.cat([init_beta.flatten(), torch.log(init_sigma_b), torch.log(init_sigma_e)])
    return init_params.requires_grad_(True)


def linear_mixed_effects(data_loader, init_sigma_b=None, init_sigma_e=None, lr=0.01, max_iter=1000):
    for X, Y, groups in data_loader:
        n_samples, n_predictors = X.shape
        n_responses = Y.shape[1]
        unique_groups = torch.unique(groups)
        n_groups = len(unique_groups)
        
        # Create group indicator matrix Z
        Z = torch.zeros((n_samples, n_groups), dtype=torch.float32)
        for i, group in enumerate(groups):
            group_idx = (unique_groups == group).nonzero(as_tuple=True)[0][0]
            Z[i, group_idx] = 1
        
        # Optimization
        init_params = initialize_lme(n_predictors, n_responses, init_sigma_b, init_sigma_e)
        optimizer = torch.optim.Adam([init_params], lr=lr)
        
        for _ in range(max_iter):
            optimizer.zero_grad()
            beta = init_params[:n_predictors * n_responses].reshape((n_predictors, n_responses))
            sigma_b = torch.exp(init_params[n_predictors * n_responses:n_predictors * n_responses + n_responses])
            sigma_e = torch.exp(init_params[n_predictors * n_responses + n_responses:])
            
            V = torch.stack([sigma_e[i]**2 * torch.eye(n_samples) + sigma_b[i]**2 * Z @ Z.T for i in range(n_responses)])
            V_inv = torch.stack([torch.inverse(V[i]) for i in range(n_responses)])
            log_likelihood = -0.5 * sum(n_samples * np.log(2 * np.pi) + torch.logdet(V[i]) + (Y[:, i] - X @ beta[:, i]).T @ V_inv[i] @ (Y[:, i] - X @ beta[:, i]) for i in range(n_responses))
            loss = -log_likelihood
            loss.backward()
            optimizer.step()
        
        optimized_params = init_params.detach()
        beta = optimized_params[:n_predictors * n_responses].reshape((n_predictors, n_responses))
        sigma_b = torch.exp(optimized_params[n_predictors * n_responses:n_predictors * n_responses + n_responses])
        sigma_e = torch.exp(optimized_params[n_predictors * n_responses + n_responses:])
        return {"beta": beta.numpy(), "sigma_b": sigma_b.numpy(), "sigma_e": sigma_e.numpy()}
