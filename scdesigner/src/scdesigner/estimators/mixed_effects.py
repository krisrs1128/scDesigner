import torch
import torch.utils.data as td
import numpy as np
from tqdm import tqdm


class LinearMixedEffectsEstimator():
    """
    Example
    -------
    # example_sce is the data from https://go.wisc.edu/69435h
    # can create it using example_sce = anndata.read_h5ad(downloaded_data_path)

    loader = FormulaWithGroupsLoader(example_sce, "~ pseudotime", "cell_type", batch_size=10)
    lme = LinearMixedEffectsEstimator()
    lme.estimate(loader.loader)
    """

    def __init__(
        self,
        lr: float = 0.01,
        max_iter: int = 1000,
        init_sigma_b: torch.Tensor = None,
        init_sigma_e: torch.Tensor = None,
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.init_sigma_b = init_sigma_b
        self.init_sigma_e = init_sigma_e

    def estimate(self, loader: td.DataLoader) -> dict:
        return lme_estimate(
            loader, self.init_sigma_b, self.init_sigma_e, self.lr, self.max_iter
        )


def initialize_lme(n_predictors, n_responses, init_sigma_b, init_sigma_e):
    init_beta = torch.zeros((n_predictors, n_responses), dtype=torch.float32)
    if init_sigma_b is None:
        init_sigma_b = torch.ones(n_responses, dtype=torch.float32)
    if init_sigma_e is None:
        init_sigma_e = torch.ones(n_responses, dtype=torch.float32)

    init_params = torch.cat(
        [init_beta.flatten(), torch.log(init_sigma_b), torch.log(init_sigma_e)]
    )
    return init_params.requires_grad_(True)


def lme_estimate(data_loader, init_sb=None, init_se=None, lr=0.01, max_iter=1000):
    for Y, X, Z in data_loader:
        n_samples, n_preds = X.shape
        n_resps = Y.shape[1]

        # initialize optimizers, marginalize b, and run maximum likelihood
        init_params = initialize_lme(n_preds, n_resps, init_sb, init_se)
        optimizer = torch.optim.Adam([init_params], lr=lr)

        for _ in range(max_iter):
            optimizer.zero_grad()
            beta = init_params[: n_preds * n_resps].reshape((n_preds, n_resps))
            sb = torch.exp(init_params[n_preds * n_resps : n_preds * n_resps + n_resps])
            se = torch.exp(init_params[n_preds * n_resps + n_resps :])

            V = torch.stack(
                [
                    se[i] ** 2 * torch.eye(n_samples) + sb[i] ** 2 * Z @ Z.T
                    for i in range(n_resps)
                ]
            )
            V_inv = torch.stack([torch.inverse(V[i]) for i in range(n_resps)])
            log_lik = -0.5 * sum(
                n_samples * np.log(2 * np.pi)
                + torch.logdet(V[i])
                + (Y[:, i] - X @ beta[:, i]).T @ V_inv[i] @ (Y[:, i] - X @ beta[:, i])
                for i in range(n_resps)
            )
            loss = -log_lik
            loss.backward()
            optimizer.step()

        # postprocess estimated parameters
        opt_params = init_params.detach()
        beta = opt_params[: n_preds * n_resps].reshape((n_preds, n_resps))
        sb = torch.exp(opt_params[n_preds * n_resps : n_preds * n_resps + n_resps])
        se = torch.exp(opt_params[n_preds * n_resps + n_resps :])
        return {
            "beta": beta.numpy(),
            "sigma_b": sb.numpy(),
            "sigma_e": se.numpy(),
        }


class PoissonMixedEffectsEstimator():
    """
    Example
    -------
    # example_sce is the data from https://go.wisc.edu/69435h
    # can create it using example_sce = anndata.read_h5ad(downloaded_data_path)

    loader = FormulaWithGroupsLoader(example_sce, "~ pseudotime", "cell_type", batch_size=100)
    pme = PoissonMixedEffectsEstimator()
    pme.estimate(loader.loader)
    """

    def __init__(
        self, lr: float = 0.01, max_iter: int = 1000, init_sigma_b: torch.Tensor = None
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.init_sigma_b = init_sigma_b

    def estimate(self, loader: td.DataLoader) -> dict:
        return poisson_mixed_effects(loader, self.init_sigma_b, self.lr, self.max_iter)


def poisson_mixed_effects_loglik(params, X, Y, Z):
    n_predictors = X.shape[1]
    n_responses = Y.shape[1]
    n_groups = Z.shape[1]

    # extract parameters by group
    beta_start = 0
    beta_end = n_predictors * n_responses
    beta = params[beta_start:beta_end].reshape((n_predictors, n_responses))

    sigma_b_start = beta_end
    sigma_b_end = sigma_b_start + n_responses
    sigma_b = torch.exp(params[sigma_b_start:sigma_b_end])

    b_start = sigma_b_end
    b = params[b_start:].reshape((n_groups, n_responses))

    # compute log likelihood
    eta = X @ beta + Z @ b
    mu = torch.exp(eta)

    log_likelihood = torch.sum(Y * eta - mu - torch.lgamma(Y + 1))
    penalty = -0.5 * torch.sum((b**2) / sigma_b**2)
    return -(log_likelihood + penalty)


def initialize_poisson_me(n_predictors, n_responses, init_sigma_b, n_groups):
    init_beta = torch.zeros((n_predictors, n_responses), dtype=torch.float32)
    init_b = torch.zeros((n_groups, n_responses), dtype=torch.float32)
    if init_sigma_b is None:
        init_sigma_b = torch.ones(n_responses, dtype=torch.float32)

    initial_params = torch.cat(
        [init_beta.flatten(), torch.log(init_sigma_b), init_b.flatten()]
    )
    return initial_params.requires_grad_(True)


def poisson_mixed_effects(data_loader, init_sigma_b=1.0, lr=0.01, max_iter=1000):
    for Y, X, Z in data_loader:
        n_predictors = X.shape[1]
        n_responses = Y.shape[1]
        n_groups = Z.shape[1]

        # Optimization
        init_params = initialize_poisson_me(
            n_predictors, n_responses, init_sigma_b, n_groups
        )
        optimizer = torch.optim.Adam([init_params], lr=lr)

        for _ in tqdm(range(max_iter), desc="Estimating Poisson Mixed Effects"):
            optimizer.zero_grad()
            loss = poisson_mixed_effects_loglik(init_params, X, Y, Z)
            loss.backward()
            optimizer.step()

        # Extract optimized parameters
        optimized_params = init_params.detach()
        beta_start = 0
        beta_end = n_predictors * n_responses
        beta = optimized_params[beta_start:beta_end].reshape(
            (n_predictors, n_responses)
        )

        b_start = beta_end + n_responses
        b = optimized_params[b_start:].reshape((n_groups, n_responses))
        return {"beta": beta.numpy(), "sigma_b": np.std(b.numpy()), "b": b.numpy()}
