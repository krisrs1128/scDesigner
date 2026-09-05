import torch
from typing import Optional
import torch.special as spec

# ==============================================================================
# Weighted Least Squares Solver
# ==============================================================================

def solve_weighted_least_squares(X, weights, responses):
    """
    Solve multiple independent weighted least squares problems in parallel.

    For each column j, solves: (X'W_j X)β_j = X'W_j z_j
    where W_j is a diagonal matrix with weights[:, j] on the diagonal.

    Parameters
    ----------
    X : torch.Tensor
        Design matrix (n × p)
    weights : torch.Tensor
        Weight matrix (n × m), one weight vector per response
    responses : torch.Tensor
        Working responses (n × m)

    Returns
    -------
    torch.Tensor
        Coefficient matrix (p × m)
    """
    # Precompute outer products X_i ⊗ X_i for each observation
    X_outer = torch.einsum("ni,nj->nij", X, X)  # (n × p × p)

    # Compute weighted normal equations: (X'WX) for all m responses at once
    eye = torch.eye(X.shape[1], device=X.device).unsqueeze(0)
    weighted_XX = torch.einsum("nm,nij->mij", weights, X_outer)  # (m × p × p)
    weighted_XX = weighted_XX + 1e-5 * eye

    # Compute X'Wz for all responses
    weighted_Xy = torch.einsum("ni,nm->mi", X, weights * responses)  # (m × p)

    # Solve all systems at once
    coefficients = torch.linalg.solve(weighted_XX, weighted_Xy.unsqueeze(-1))
    return coefficients.squeeze(-1).T  # (p × m)


# ==============================================================================
# Mean Parameter Updates (Beta)
# ==============================================================================

def update_mean_coefficients(X, counts, beta, dispersion, clip: float = 50.0):
    """
    Update mean model coefficients using one Newton-Raphson step.

    Uses IRLS (Iteratively Reweighted Least Squares) with:
    - Working weights: W = μ/(1 + μ/θ)
    - Working response: Z = Xβ + (Y - μ)/μ

    Parameters
    ----------
    X : torch.Tensor
        Design matrix (n × p)
    counts : torch.Tensor
        Observed counts (n × m)
    beta : torch.Tensor
        Current coefficients (p × m)
    dispersion : torch.Tensor
        Current dispersion parameters (n × m)
    clip : float, optional
        Maximum absolute value for linear predictor, by default 5.0

    Returns
    -------
    torch.Tensor
        Updated coefficients (p × m)
    """
    linear_pred = torch.clip(X @ beta, min=-clip, max=clip)
    mean = torch.exp(linear_pred)
    weights = mean / (1 + mean / dispersion)
    working_response = linear_pred + (counts - mean) / mean
    return solve_weighted_least_squares(X, weights, working_response)


# ==============================================================================
# Dispersion Parameter Updates (Gamma)
# ==============================================================================

def update_dispersion_coefficients(Z, counts, mean, gamma, clip: float = 50.0,
                                   disp_ridge: float = 1e-4):
    """
    Update dispersion model coefficients using one Fisher scoring step.

    Computes the score (∂ℓ/∂θ) and approximate Fisher information
    (θY/(θ+Y)) per observation, then solves the weighted least squares
    system directly using the products W·η and θ·score to avoid
    dividing by near-zero weights for zero-count cells.

    A ridge penalty disp_ridge · I is added to the information matrix
    Z'WZ to regularize the update when many observations have zero
    counts and thus zero Fisher information.

    Parameters
    ----------
    Z : torch.Tensor
        Dispersion design matrix (n × q)
    counts : torch.Tensor
        Observed counts (n × m)
    mean : torch.Tensor
        Current mean estimates (n × m)
    gamma : torch.Tensor
        Current dispersion coefficients (q × m)
    clip : float, optional
        Maximum absolute value for linear predictor, by default 5.0
    disp_ridge : float, optional
        Ridge penalty added to Z'WZ for the dispersion update, by default 1e-4

    Returns
    -------
    torch.Tensor
        Updated dispersion coefficients (q × m)
    """
    linear_pred = torch.clip(Z @ gamma, min=-clip, max=clip)
    dispersion = torch.exp(linear_pred)

    # Score: ∂ℓ/∂θ
    psi_diff = spec.digamma(counts + dispersion) - spec.digamma(dispersion)
    score = (psi_diff + torch.log(dispersion) - torch.log(mean + dispersion) +
             (mean - counts) / (mean + dispersion))

    # Approximate Fisher information (replaces exact Hessian) Uses the fact that
    # θY/(θ + Y) ≈ θ²[ψ₁(θ) - ψ₁(Y + θ)] and the approximation ψ₁(x) ≈ 1/x
    weights = (dispersion * counts) / (dispersion + counts)

    # Solve (Z'WZ + λI) γ = Z'(Wz) via normal equations. We will refer to this
    # as normal_lhs and normal_rhs, respectively. The first step builds
    #    W·z = W·η + θ·score = weights * linear_pred + dispersion * score,
    # which helps avoid any division by small weights in the usual IRLS update.
    Wz = weights * linear_pred + dispersion * score  # (n × m)
    sample_outer = torch.einsum("ni,nj->nij", Z, Z)  # (n × q × q), {zₙzₙᵀ} for n
    ZtWZ = torch.einsum("nm,nij->mij", weights, sample_outer)  # (m × q × q)
    I = torch.eye(Z.shape[1], device=Z.device).unsqueeze(0)
    normal_lhs = ZtWZ + disp_ridge * I
    normal_rhs = torch.einsum("ni,nm->mi", Z, Wz)  # (m × q), Z'(Wz)

    coefficients = torch.linalg.solve(normal_lhs, normal_rhs.unsqueeze(-1))
    return coefficients.squeeze(-1).T  # (q × m)


# ==============================================================================
# Initialization
# ==============================================================================

def accumulate_poisson_statistics(loader, beta, n_genes, p_mean, clip = 5):
    """
    Accumulate weighted normal equations for Poisson IRLS across batches.

    Parameters
    ----------
    loader : DataLoader
        DataLoader yielding (y_batch, x_dict)
    beta : torch.Tensor
        Current coefficients (p_mean × n_genes)
    n_genes : int
        Number of genes
    p_mean : int
        Number of mean predictors
    clip : float, optional
        Maximum absolute value for linear predictor, by default 5

    Returns
    -------
    weighted_XX : torch.Tensor
        Accumulated X'WX (n_genes × p_mean × p_mean)
    weighted_Xy : torch.Tensor
        Accumulated X'Wz (p_mean × n_genes)
    """
    weighted_XX = torch.zeros((n_genes, p_mean, p_mean), device=beta.device)
    weighted_Xy = torch.zeros((p_mean, n_genes), device=beta.device)

    for y_batch, x_dict in loader:
        X = x_dict['mean']

        linear_pred = torch.clip(X @ beta, min=-clip, max=clip)
        mean = torch.exp(linear_pred)
        adjustment = torch.clamp((y_batch - mean) / mean, min=-clip, max=clip)
        working_response = linear_pred + adjustment

        X_outer = torch.einsum("ni,nj->nij", X, X)
        weighted_XX += torch.einsum("nm,nij->mij", mean, X_outer)
        weighted_Xy += torch.einsum("ni,nm->im", X, mean * working_response)

    return weighted_XX, weighted_Xy


def accumulate_dispersion_statistics(loader, beta, clip = 5):
    """
    Accumulate statistics for variance-based method of moments dispersion estimation.

    Uses E[(Y-μ)²] = μ + μ²/θ, giving θ = Σμ² / (Σ(Y-μ)² - Σμ).

    Parameters
    ----------
    loader : DataLoader
        DataLoader yielding (y_batch, x_dict)
    beta : torch.Tensor
        Mean coefficients (p_mean × n_genes)
    clip : float, optional
        Maximum absolute value for linear predictor, by default 5

    Returns
    -------
    sum_mu : torch.Tensor
        Total predicted mean Σμ (n_genes,)
    sum_mu2 : torch.Tensor
        Total predicted squared mean Σμ² (n_genes,)
    sum_sq_resid : torch.Tensor
        Total squared residuals Σ(Y-μ)² (n_genes,)
    n_total : int
        Total number of observations
    """
    n_genes = beta.shape[1]
    sum_mu = torch.zeros(n_genes, device=beta.device)
    sum_mu2 = torch.zeros(n_genes, device=beta.device)
    sum_sq_resid = torch.zeros(n_genes, device=beta.device)
    n_total = 0

    for y_batch, x_dict in loader:
        X = x_dict['mean']
        linear_pred = torch.clip(X @ beta, min=-clip, max=clip)
        mean_batch = torch.exp(linear_pred)

        sum_mu += mean_batch.sum(dim=0)
        sum_mu2 += (mean_batch ** 2).sum(dim=0)
        sum_sq_resid += ((y_batch - mean_batch) ** 2).sum(dim=0)
        n_total += y_batch.shape[0]

    return sum_mu, sum_mu2, sum_sq_resid, n_total


def _is_intercept_column(col):
    """Return True if col is all ones (i.e. an intercept column)."""
    return torch.allclose(col, torch.ones_like(col), atol=1e-6)


def _initialize_beta_intercept(loader, n_genes, p_mean):
    """Initialize using Gene Means

    Initialize beta in the poisson regression so that predicted mean ≈ observed
    gene mean. When the model has no intercept (checked by looking at the first
    column of the first batch), we return zeros without any initialization.
    """
    _, x_peek = next(iter(loader))
    device = x_peek["mean"].device
    beta = torch.zeros((p_mean, n_genes), device=device)
    if not _is_intercept_column(x_peek["mean"][:, 0]):
        return beta

    gene_sums = torch.zeros(n_genes, device=device)
    intercept_sum = 0.0
    n_total = 0

    for y_batch, x_dict in loader:
        gene_sums += y_batch.sum(dim=0)
        intercept_sum += x_dict["mean"][:, 0].sum()
        n_total += y_batch.shape[0]

    gene_means = torch.clamp(gene_sums / n_total, min=1e-3)
    intercept_mean = intercept_sum / n_total
    beta[0, :] = torch.log(gene_means) / intercept_mean

    return beta


def initialize_parameters(loader, n_genes, p_mean, p_disp, max_iter = 10,
                          tol = 1e-5, clip = 5, damping = 0.5):
    """
    Initialize parameters using batched Poisson IRLS followed by
    Method-of-Moments dispersion.

    Approach:
        1. Iteratively fit Poisson GLM by accumulating X'WX and X'WZ across batches
        2. Use fitted Poisson means to estimate dispersion via variance-based
        MoM: θ = Σμ² / max(Σ(Y-μ)² - Σμ, ε) This estimator comes from the fact
        that Var(y) = μ + μ²/θ.

    Parameters
    ----------
    loader : DataLoader
        DataLoader yielding (y_batch, x_dict)
    n_genes : int
        Number of response columns (genes)
    p_mean : int
        Number of predictors in the mean model
    p_disp : int
        Number of predictors in the dispersion model
    max_iter : int, optional
        Maximum Poisson IRLS iterations, by default 10
    tol : float, optional
        Convergence tolerance for beta coefficients, by default 1e-3
    clip : float, optional
        Maximum absolute value for linear predictor, by default 5
    damping : float, optional
        Step-size damping factor for IRLS updates. beta_new = (1-damping)*beta +
        damping*beta_target. Use < 1 to prevent overshooting, by default 0.5

    Returns
    -------
    beta_init : torch.Tensor
        (p_mean × n_genes) tensor
    gamma_init : torch.Tensor
        (p_disp × n_genes) tensor
    """
    # Initialize predicted mean ≈ observed gene mean across cells.
    beta = _initialize_beta_intercept(loader, n_genes, p_mean)

    for _ in range(max_iter):
        weighted_XX, weighted_Xy = accumulate_poisson_statistics(
            loader, beta, n_genes, p_mean, clip
        )

        eye = torch.eye(p_mean, device=beta.device).unsqueeze(0)
        weighted_XX_reg = weighted_XX + 1e-6 * eye
        beta_target = torch.linalg.solve(
            weighted_XX_reg, weighted_Xy.T.unsqueeze(-1)
        ).squeeze(-1).T

        beta_new = (1 - damping) * beta + damping * beta_target
        if torch.max(torch.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # MoM E[(Y-μ)²] = μ + μ²/θ, so θ = Σμ² / (Σ(Y-μ)² - Σμ)
    sum_mu, sum_mu2, sum_sq_resid, _ = accumulate_dispersion_statistics(
        loader, beta, clip
    )
    dispersion = sum_mu2 / torch.clamp(sum_sq_resid - sum_mu, min=0.1)

    gamma = torch.zeros((p_disp, n_genes), device=beta.device)
    gamma[0, :] = torch.log(torch.clamp(dispersion, min=1e-3))
    return beta, gamma


# ==============================================================================
# Batch Log-Likelihood
# ==============================================================================

def compute_batch_loglikelihood(y, mu, r):
    """
    Compute the negative binomial log-likelihood for a batch.

    Formula:
        ℓ = Σ [log Γ(Y+θ) - log Γ(θ) - log Γ(Y+1) + θ log θ + Y log μ - (Y+θ)log(μ+θ)]

    Parameters
    ----------
    y : torch.Tensor
        Observed counts (n_batch × m_active)
    mu : torch.Tensor
        Predicted means (n_batch × m_active)
    r : torch.Tensor
        Dispersion parameters (n_batch × m_active)

    Returns
    -------
    torch.Tensor
        Total log-likelihood per response (m_active,)
    """
    ll = (
        torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1.0)
        + r * torch.log(r) + y * torch.log(mu)
        - (r + y) * torch.log(r + mu)
    )
    return torch.sum(ll, dim=0)


# ==============================================================================
# Stochastic IRLS Step
# ==============================================================================

def step_stochastic_irls(
    y,
    X,
    Z,
    beta,
    gamma,
    eta: float = 0.8,
    tol: float = 1e-4,
    ll_prev: Optional[torch.Tensor] = None,
    clip_mean: float = 5.0,
    clip_disp: float = 5.0,
    disp_ridge: float = 1e-4
):
    """
    Perform a single damped Newton-Raphson update on a minibatch.

    Logic:
        1. Compute log-likelihood with current coefficients.
        2. Perform one IRLS step for Mean (Beta) and Dispersion (Gamma).
        3. Re-compute log-likelihood to determine convergence.
        4. Return updated coefficients and boolean convergence mask.

    Parameters
    ----------
    y : torch.Tensor
        Count batch (n × m)
    X : torch.Tensor
        Mean design matrix (n × p)
    Z : torch.Tensor
        Dispersion design matrix (n × q)
    beta : torch.Tensor
        Current mean coefficients (p × m)
    gamma : torch.Tensor
        Current dispersion coefficients (q × m)
    eta : float, optional
        Damping factor (learning rate), 1.0 is pure Newton step, by default 0.8
    tol : float, optional
        Relative log-likelihood change threshold for convergence, by default 1e-4
    ll_prev : torch.Tensor, optional
        Previous log-likelihood values, by default None
    clip_mean : float, optional
        Maximum absolute value for mean linear predictor, by default 5.0
    clip_disp : float, optional
        Maximum absolute value for dispersion linear predictor, by default 5.0

    Returns
    -------
    beta_next : torch.Tensor
        Updated mean coefficients (p × m)
    gamma_next : torch.Tensor
        Updated dispersion coefficients (q × m)
    converged : torch.Tensor
        Boolean mask of converged responses (m,)
    ll_next : torch.Tensor
        Updated log-likelihood values (m,)
    """
    # --- 2. Update Mean (Beta) ---
    # Working weights W = μ/(1 + μ/θ)
    beta_target = update_mean_coefficients(X, y, beta, torch.exp(Z @ gamma), clip=clip_mean)
    beta_next = (1 - eta) * beta + eta * beta_target

    # --- 3. Update Dispersion (Gamma) ---
    # Update depends on the latest mean estimates
    linear_pred_mu = torch.clip(X @ beta_next, min=-clip_mean, max=clip_mean)
    mu = torch.exp(linear_pred_mu)
    gamma_target = update_dispersion_coefficients(Z, y, mu, gamma, clip=clip_disp,
                                                     disp_ridge=disp_ridge)
    gamma_next = (1 - eta) * gamma + eta * gamma_target

    # --- 4. Convergence Check ---
    linear_pred_r_next = torch.clip(Z @ gamma_next, min=-clip_disp, max=clip_disp)
    ll_next = compute_batch_loglikelihood(y, mu, torch.exp(linear_pred_r_next))

    # Relative improvement in the objective function
    if ll_prev is not None:
        rel_change = torch.abs(ll_next - ll_prev) / (torch.abs(ll_prev) + 1e-10)
        converged = rel_change <= tol
    else:
        converged = False
    return beta_next, gamma_next, converged, ll_next