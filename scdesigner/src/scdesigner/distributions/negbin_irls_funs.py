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
    eye = torch.eye(X.shape[1]).unsqueeze(0)
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

def update_mean_coefficients(X, counts, beta, dispersion, clip: float = 5.0):
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

def update_dispersion_coefficients(Z, counts, mean, gamma, clip: float = 5.0):
    """
    Update dispersion model coefficients using one Fisher scoring step.

    Uses working response U = η + θ·s/w where:
    - η = Zγ (linear predictor)
    - s = score with respect to θ
    - w = approximate Fisher information

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

    # Approximate Fisher information (replaces exact Hessian)
    # Approximation: θY/(θ + Y) ≈ θ²[ψ₁(θ) - ψ₁(Y + θ)]
    weights = ((dispersion * counts) / (dispersion + counts)).clip(min=1e-6)
    working_response = linear_pred + (dispersion * score) / weights
    return solve_weighted_least_squares(Z, weights, working_response)


# ==============================================================================
# Initialization
# ==============================================================================

def estimate_constant_dispersion(X, counts, beta):
    """
    Estimate constant dispersion for each response using method of moments.

    Uses Pearson residuals: θ̂ = (Σμ) / max(χ² - df, 0.1)
    where χ² = Σ(Y - μ)²/μ and df = n - p.

    Parameters
    ----------
    X : torch.Tensor
        Design matrix (n × p)
    counts : torch.Tensor
        Observed counts (n × m)
    beta : torch.Tensor
        Mean coefficients (p × m)

    Returns
    -------
    torch.Tensor
        Dispersion estimates (m,)
    """
    mean = torch.exp(X @ beta)
    pearson_chi2 = torch.sum((counts - mean)**2 / mean, dim=0)
    sum_mean = torch.sum(mean, dim=0)

    degrees_freedom = counts.shape[0] - X.shape[1]
    dispersion = sum_mean / torch.clip(pearson_chi2 - degrees_freedom, min=0.1)
    return torch.clip(dispersion, min=0.1)


def fit_poisson_initial(X, counts, tol: float = 1e-3, max_iter: int = 100, clip: float = 5.0):
    """
    Fit Poisson GLM to initialize mean parameters.

    Parameters
    ----------
    X : torch.Tensor
        Design matrix (n × p)
    counts : torch.Tensor
        Observed counts (n × m)
    tol : float, optional
        Convergence tolerance, by default 1e-3
    max_iter : int, optional
        Maximum iterations, by default 100
    clip : float, optional
        Maximum absolute value for linear predictor, by default 5.0

    Returns
    -------
    torch.Tensor
        Initial coefficients (p × m)
    """
    n_features, n_responses = X.shape[1], counts.shape[1]
    beta = torch.zeros((n_features, n_responses))

    for _ in range(max_iter):
        beta_old = beta.clone()
        linear_pred = torch.clip(X @ beta, min=-clip, max=clip)
        mean = torch.exp(linear_pred)
        working_response = linear_pred + (counts - mean) / mean

        beta = solve_weighted_least_squares(X, mean, working_response)
        if torch.max(torch.abs(beta - beta_old)) < tol:
            break

    return beta


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
    weighted_XX = torch.zeros((n_genes, p_mean, p_mean))
    weighted_Xy = torch.zeros((p_mean, n_genes))

    for y_batch, x_dict in loader:
        X = x_dict['mean'].to("cpu")

        linear_pred = torch.clip(X @ beta, min=-clip, max=clip)
        mean = torch.exp(linear_pred)
        working_response = linear_pred + (y_batch.to("cpu") - mean) / mean

        X_outer = torch.einsum("ni,nj->nij", X, X)
        weighted_XX += torch.einsum("nm,nij->mij", mean, X_outer)
        weighted_Xy += torch.einsum("ni,nm->im", X, mean * working_response)

    return weighted_XX, weighted_Xy


def accumulate_dispersion_statistics(loader, beta, clip = 5):
    """
    Accumulate Pearson statistics for method of moments dispersion estimation.

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
    sum_mean : torch.Tensor
        Total predicted mean (n_genes,)
    sum_pearson : torch.Tensor
        Total Pearson chi-squared (n_genes,)
    n_total : int
        Total number of observations
    """
    sum_mean = torch.zeros(beta.shape[1])
    sum_pearson = torch.zeros(beta.shape[1])
    n_total = 0

    for y_batch, x_dict in loader:
        X = x_dict['mean'].to('cpu')
        linear_pred = torch.clip(X @ beta, min=-clip, max=clip)
        mean_batch = torch.exp(linear_pred)

        sum_mean += mean_batch.sum(dim=0)
        sum_pearson += ((y_batch.to('cpu') - mean_batch)**2 / mean_batch).sum(dim=0)
        n_total += y_batch.shape[0]

    return sum_mean, sum_pearson, n_total


def initialize_parameters(loader, n_genes, p_mean, p_disp, max_iter = 10,
                          tol = 1e-3, clip = 5):
    """
    Initialize parameters using batched Poisson IRLS followed by MoM dispersion.

    Logic:
        1. Iteratively fit Poisson GLM by accumulating X'WX and X'WZ across batches
        2. Use fitted Poisson means to estimate dispersion via Method of Moments

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
        Maximum absolute value for linear predictor, by default 10

    Returns
    -------
    beta_init : torch.Tensor
        (p_mean × n_genes) tensor
    gamma_init : torch.Tensor
        (p_disp × n_genes) tensor
    """
    beta = torch.zeros((p_mean, n_genes))
    for _ in range(max_iter):
        weighted_XX, weighted_Xy = accumulate_poisson_statistics(
            loader, beta, n_genes, p_mean, clip
        )

        eye = torch.eye(p_mean).unsqueeze(0)
        weighted_XX_reg = weighted_XX + 1e-6 * eye
        beta_new = torch.linalg.solve(
            weighted_XX_reg, weighted_Xy.T.unsqueeze(-1)
        ).squeeze(-1).T

        if torch.max(torch.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    sum_mean, sum_pearson, n_total = accumulate_dispersion_statistics(
        loader, beta, clip
    )

    degrees_freedom = n_total - p_mean
    dispersion = sum_mean / torch.clip(sum_pearson - degrees_freedom, min=0.1)

    gamma = torch.zeros((p_disp, n_genes))
    gamma[0, :] = torch.log(torch.clip(dispersion, min=0.1))
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
    clip_disp: float = 5.0
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
    gamma_target = update_dispersion_coefficients(Z, y, mu, gamma, clip=clip_disp)
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