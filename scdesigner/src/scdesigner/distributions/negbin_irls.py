import torch
import copy
from .negbin import NegBin
from .negbin_irls_funs import initialize_parameters, step_stochastic_irls
from ..data.formula import standardize_formula
from ..utils.kwargs import _filter_kwargs, DEFAULT_ALLOWED_KWARGS
from typing import Union, Dict, Optional


class NegBinIRLS(NegBin):
    """
    Negative-Binomial Marginal using Stochastic IRLS with
    active response tracking and log-likelihood convergence.
    """
    def __init__(self, formula: Union[Dict, str], **kwargs):
        formula = standardize_formula(formula, allowed_keys=['mean', 'dispersion'])
        super().__init__(formula, device="cpu")


    def fit(
        self,
        max_epochs: int = 500,
        tol=1e-4,
        eta=0.1,
        verbose=True,
        log_dir: Optional[str] = None,
        disp_ridge=1e-4,
        val_frac: float = 0.1,
        min_epochs: int = 10,
        loss_tol: float = 0.01,
        patience: int = 6,
        validation_seed: int = 0,
        **kwargs,
    ):
        self._validate_early_stopping_args(min_epochs, loss_tol, patience)
        self.setup_validation_split(val_frac=val_frac, validation_seed=validation_seed)

        if self.predict is None:
            self.setup_optimizer(**kwargs)

        # 1. Initialization using poisson fit
        initialize_kwargs = _filter_kwargs(kwargs, DEFAULT_ALLOWED_KWARGS['initialize'])
        beta_init, gamma_init = initialize_parameters(
            self._active_train_loader(), self.n_outcomes, self.feature_dims['mean'],
            self.feature_dims['dispersion'],
            **initialize_kwargs
        )

        with torch.no_grad():
            self.predict.coefs['mean'].copy_(beta_init)
            self.predict.coefs['dispersion'].copy_(gamma_init)

        # 2. All genes are active at the start
        active_mask = torch.ones(self.n_outcomes, dtype=torch.bool)
        ll_ = - 1e9 * torch.ones(self.n_outcomes, dtype=torch.float32)

        if log_dir is not None:
            import os
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)
        else:
            writer = None

        self.fit_history = []
        self.best_epoch = None
        self.best_validation_loss = None
        self.stopped_epoch = None
        best_loss = float("inf")
        best_state = None
        wait = 0
        train_loader = self._active_train_loader()

        for epoch in range(1, max_epochs + 1):
            if not active_mask.any():
                break
            ll, n_batches = 0.0, 0

            with torch.no_grad():
                for batch in train_loader:
                    y_batch, x_dict = self._move_batch_to_device(batch)

                    # Slice active genes
                    idx = torch.where(active_mask)[0]
                    y_act = y_batch[:, active_mask]
                    X = x_dict['mean']
                    Z = x_dict['dispersion']

                    # Fetch current coefficients and update
                    b_curr = self.predict.coefs['mean'][:, active_mask]
                    g_curr = self.predict.coefs['dispersion'][:, active_mask]
                    b_next, g_next, conv_mask, ll_cur = step_stochastic_irls(y_act, X, Z, b_curr, g_curr, eta, tol, ll_[active_mask],
                                                                                 disp_ridge=disp_ridge)
                    ll_[active_mask] = ll_cur

                    # Update Parameters and de-activate converged genes
                    with torch.no_grad():
                        self.predict.coefs['mean'][:, active_mask] = b_next
                        self.predict.coefs['dispersion'][:, active_mask] = g_next
                        active_mask[idx[conv_mask]] = False

                    # Accumulate batch log-likelihood using `ll` from the IRLS step
                    ll += ll_.sum().item()
                    n_batches += 1

                train_loss = -ll / n_batches if n_batches > 0 else float("nan")
                val_loss = self._validation_loss()
                is_best = False
                stopped = False
                if val_loss is not None:
                    if best_loss - val_loss >= loss_tol:
                        best_loss = val_loss
                        wait = 0
                        best_state = copy.deepcopy(self.predict.state_dict())
                        self.best_epoch = epoch
                        self.best_validation_loss = best_loss
                        is_best = True
                    else:
                        wait += 1

                    if epoch >= min_epochs and wait >= patience:
                        stopped = True
                        self.stopped_epoch = epoch

                self.fit_history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "best": is_best,
                        "stopped": stopped,
                    }
                )

                if verbose and (epoch % 10 == 0 or stopped):
                    msg = (
                        f"Epoch {epoch}/{max_epochs} | Genes remaining: "
                        f"{active_mask.sum().item()} | Loss: {train_loss:.4f}"
                    )
                    if val_loss is not None:
                        msg += f" | Val Loss: {val_loss:.4f}"
                    print(msg, end='\r')

            if writer is not None:
                writer.add_scalar("loss/train", train_loss if n_batches > 0 else 0, epoch)
                if val_loss is not None:
                    writer.add_scalar("loss/validation", val_loss, epoch)
            if stopped or not active_mask.any():
                break

        if verbose:
            print() # Maintain the loss output

        if writer is not None:
            writer.close()

        if best_state is not None:
            self.predict.load_state_dict(best_state)
        self.parameters = self.format_parameters()
