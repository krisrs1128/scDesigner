import torch
from .negbin import NegBin
from .negbin_irls_funs import initialize_parameters_full_pass, step_stochastic_irls

class NegBinIRLS(NegBin):
    """
    Negative-Binomial Marginal using Stochastic IRLS with
    active response tracking and log-likelihood convergence.
    """
    def fit(self, max_epochs=10, tol=1e-4, eta=0.6, verbose=True, **kwargs):
            if self.predict is None: self.setup_optimizer(**kwargs) #

            # 1. Initialization using poisson fit
            beta_init, gamma_init = initialize_parameters_full_pass(
                self.loader, self.device, self.n_outcomes,
                self.feature_dims['mean'], self.feature_dims['dispersion']
            )
            with torch.no_grad():
                self.predict.coefs['mean'].copy_(beta_init)
                self.predict.coefs['dispersion'].copy_(gamma_init)

            # 2. All genes are active at the start
            active_mask = torch.ones(self.n_outcomes, dtype=torch.bool, device=self.device)

            for epoch in range(max_epochs):
                for y_batch, x_dict in self.loader:
                    if not active_mask.any(): break

                    # Slice active genes and move to device
                    idx = torch.where(active_mask)[0]
                    #y_act = y_batch.to(self.device)[:, active_mask]
                    y_act = y_batch[:, active_mask]
                    X, Z = x_dict['mean'], x_dict['dispersion']

                    # Fetch current coefficients
                    b_curr = self.predict.coefs['mean'][:, active_mask].detach()
                    g_curr = self.predict.coefs['dispersion'][:, active_mask].detach()

                    # Perform the update
                    b_next, g_next, conv_mask, ll = step_stochastic_irls(y_act, X, Z, b_curr, g_curr, eta, tol)

                    # Update Parameters and de-activate converged genes
                    with torch.no_grad():
                        self.predict.coefs['mean'][:, active_mask] = b_next
                        self.predict.coefs['dispersion'][:, active_mask] = g_next
                        active_mask[idx[conv_mask]] = False

                if verbose and ((epoch + 1) % 10) == 0:
                    print(f"Epoch {epoch+1} | Genes remaining: {active_mask.sum().item()} | relchange: {ll.max().numpy()}")
                    if not active_mask.any(): break

            self.parameters = self.format_parameters() #