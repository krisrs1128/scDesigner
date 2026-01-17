import torch
from .negbin import NegBin
from .negbin_irls_funs import initialize_parameters, step_stochastic_irls
from ..data.formula import standardize_formula
from ..utils.kwargs import _filter_kwargs, DEFAULT_ALLOWED_KWARGS
from typing import Union, Dict


class NegBinIRLS(NegBin):
    """
    Negative-Binomial Marginal using Stochastic IRLS with
    active response tracking and log-likelihood convergence.
    """
    def __init__(self, formula: Union[Dict, str], **kwargs):
        formula = standardize_formula(formula, allowed_keys=['mean', 'dispersion'])
        super().__init__(formula, device="cpu")


    def fit(self, max_epochs=10, tol=1e-4, eta=0.1, verbose=True, **kwargs):
        if self.predict is None:
                self.setup_optimizer(**kwargs)

        # 1. Initialization using poisson fit
        initialize_kwargs = _filter_kwargs(kwargs, DEFAULT_ALLOWED_KWARGS['initialize'])
        beta_init, gamma_init = initialize_parameters(
            self.loader, self.n_outcomes, self.feature_dims['mean'],
            self.feature_dims['dispersion'],
            **initialize_kwargs
        )

        with torch.no_grad():
            self.predict.coefs['mean'].copy_(beta_init)
            self.predict.coefs['dispersion'].copy_(gamma_init)

        # 2. All genes are active at the start
        active_mask = torch.ones(self.n_outcomes, dtype=torch.bool)
        ll_ = - 1e9 * torch.ones(self.n_outcomes, dtype=torch.float32)

        for epoch in range(max_epochs):
            if not active_mask.any(): break
            ll, n_batches = 0.0, 0

            with torch.no_grad():
                for y_batch, x_dict in self.loader:

                    # Slice active genes
                    idx = torch.where(active_mask)[0]
                    y_act = y_batch[:, active_mask]
                    X = x_dict['mean']
                    Z = x_dict['dispersion']

                    # Fetch current coefficients and update
                    b_curr = self.predict.coefs['mean'][:, active_mask]
                    g_curr = self.predict.coefs['dispersion'][:, active_mask]
                    b_next, g_next, conv_mask, ll_cur = step_stochastic_irls(y_act, X, Z, b_curr, g_curr, eta, tol, ll_[active_mask])
                    ll_[active_mask] = ll_cur

                    # Update Parameters and de-activate converged genes
                    with torch.no_grad():
                        self.predict.coefs['mean'][:, active_mask] = b_next
                        self.predict.coefs['dispersion'][:, active_mask] = g_next
                        active_mask[idx[conv_mask]] = False

                    # Accumulate batch log-likelihood using `ll` from the IRLS step
                    ll += ll_.sum().item()
                    n_batches += 1

                if verbose and ((epoch + 1) % 10) == 0:
                    print(f"Epoch {epoch+1}/{max_epochs} | Genes remaining: {active_mask.sum().item()} | Loss: {-ll / n_batches:.4f}", end='\r')
                    if not active_mask.any(): break

        self.parameters = self.format_parameters()