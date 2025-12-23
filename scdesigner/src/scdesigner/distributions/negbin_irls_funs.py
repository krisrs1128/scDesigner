import torch
import torch.special as spec

# ==============================================================================
# Weighted Least Squares Solver
# ==============================================================================

def solve_weighted_least_squares(X, weights, responses):
    """
    Solve multiple independent weighted least squares problems in parallel.

    For each column j, solves: (X'W_j X)β_j = X'W_j z_j
    where W_j is a diagonal matrix with weights[:, j] on the diagonal.

    Args:
        X: Design matrix (n × p)
        weights: Weight matrix (n × m), one weight vector per response
        responses: Working responses (n × m)

    Returns:
        Coefficient matrix (p × m)
    """
    # Precompute outer products X_i ⊗ X_i for each observation
    X_outer = torch.einsum("ni,nj->nij", X, X)  # (n × p × p)

    # Compute weighted normal equations: (X'WX) for all m responses at once
    eye = torch.eye(X.shape[1], device=X.device).unsqueeze(0)
    weighted_XX = torch.einsum("nm,nij->mij", weights, X_outer)  # (m × p × p)
    weighted_XX = weighted_XX + 1e-6 * eye

    # Compute X'Wz for all responses
    weighted_Xy = torch.einsum("ni,nm->mi", X, weights * responses)  # (m × p)

    # Solve all systems at once
    coefficients = torch.linalg.solve(weighted_XX, weighted_Xy.unsqueeze(-1))
    return coefficients.squeeze(-1).T  # (p × m)


# ==============================================================================
# Mean Parameter Updates (Beta)
# ==============================================================================

def update_mean_coefficients(X, counts, beta, dispersion):
    """
    Update mean model coefficients using one Newton-Raphson step.

    Uses IRLS (Iteratively Reweighted Least Squares) with:
    - Working weights: W = μ/(1 + μ/θ)
    - Working response: Z = Xβ + (Y - μ)/μ

    Args:
        X: Design matrix (n × p)
        counts: Observed counts (n × m)
        beta: Current coefficients (p × m)
        dispersion: Current dispersion parameters (n × m)

    Returns:
        Updated coefficients (p × m)
    """
    mean = torch.exp(X @ beta)
    weights = mean / (1 + mean / dispersion)
    working_response = X @ beta + (counts - mean) / mean
    return solve_weighted_least_squares(X, weights, working_response)


# ==============================================================================
# Dispersion Parameter Updates (Gamma)
# ==============================================================================

def update_dispersion_coefficients(Z, counts, mean, gamma):
    """
    Update dispersion model coefficients using one Fisher scoring step.

    Uses working response U = η + θ·s/w where:
    - η = Zγ (linear predictor)
    - s = score with respect to θ
    - w = approximate Fisher information

    Args:
        Z: Dispersion design matrix (n × q)
        counts: Observed counts (n × m)
        mean: Current mean estimates (n × m)
        gamma: Current dispersion coefficients (q × m)

    Returns:
        Updated dispersion coefficients (q × m)
    """
    linear_pred = Z @ gamma
    dispersion = torch.exp(linear_pred)

    # Score: ∂ℓ/∂θ
    psi_diff = spec.digamma(counts + dispersion) - spec.digamma(dispersion)
    score = (psi_diff + torch.log(dispersion) - torch.log(mean + dispersion) +
             (mean - counts) / (mean + dispersion))

    # Approximate Fisher information (replaces exact Hessian)
    # Approximation: θY/(θ + Y) ≈ θ²[ψ₁(θ) - ψ₁(Y + θ)]
    info = (dispersion * counts) / (dispersion + counts)

    weights = torch.clamp(info, min=1e-6)
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

    Args:
        X: Design matrix (n × p)
        counts: Observed counts (n × m)
        beta: Mean coefficients (p × m)

    Returns:
        Dispersion estimates (m,)
    """
    mean = torch.exp(X @ beta)
    pearson_chi2 = torch.sum((counts - mean)**2 / mean, dim=0)
    sum_mean = torch.sum(mean, dim=0)

    degrees_freedom = counts.shape[0] - X.shape[1]
    dispersion = sum_mean / torch.clamp(pearson_chi2 - degrees_freedom, min=0.1)
    return torch.clamp(dispersion, min=0.1)


def fit_poisson_initial(X, counts, tol=1e-3, max_iter=100):
    """
    Fit Poisson GLM to initialize mean parameters.

    Args:
        X: Design matrix (n × p)
        counts: Observed counts (n × m)
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        Initial coefficients (p × m)
    """
    n_features, n_responses = X.shape[1], counts.shape[1]
    beta = torch.zeros((n_features, n_responses))

    for _ in range(max_iter):
        beta_old = beta.clone()
        mean = torch.exp(X @ beta)
        working_response = X @ beta + (counts - mean) / mean

        # Fit with Poisson weights (mean)
        beta = solve_weighted_least_squares(X, mean, working_response)
        if torch.max(torch.abs(beta - beta_old)) < tol:
            break

    return beta

# ==============================================================================
# Global Initialization (Full Pass)
# ==============================================================================

def initialize_parameters_full_pass(loader, device, n_genes, p_mean, p_disp):
    """
    Compute global parameter initialization using two passes over the data.

    Logic:
        1. Pass 1: Accumulate total counts to find the global mean per gene.
           Initialize Beta intercept as log(global_mean).
        2. Pass 2: Calculate Pearson residuals using global means to estimate
           dispersion (theta) via Method of Moments.
           Initialize Gamma intercept as log(theta).

    Args:
        loader: DataLoader yielding (y_batch, x_dict)
        device: torch.device (cpu, cuda, or mps)
        n_genes: Number of response columns (genes)
        p_mean: Number of predictors in the mean model
        p_disp: Number of predictors in the dispersion model

    Returns:
        beta_init: (p_mean × n_genes) tensor
        gamma_init: (p_disp × n_genes) tensor
    """
    # --- Pass 1: Global Mean ---
    sum_y = torch.zeros(n_genes, device=device)
    n_total = 0

    for y_batch, _ in loader:
        y_batch = y_batch.to(device)
        sum_y += y_batch.sum(dim=0)
        n_total += y_batch.shape[0]

    global_mean = sum_y / n_total
    beta_init = torch.zeros((p_mean, n_genes), device=device)
    beta_init[0, :] = torch.log(torch.clamp(global_mean, min=1e-2))

    # --- Pass 2: Global Dispersion (MoM) ---
    # θ̂ = (Σμ) / max(Σ(Y-μ)²/μ - (n-p), 0.1)
    sum_mu = torch.zeros(n_genes, device=device)
    sum_pearson = torch.zeros(n_genes, device=device)

    for y_batch, x_dict in loader:
        y_batch = y_batch.to(device)
        X = x_dict['mean'].to(device)
        mu_batch = torch.exp(X @ beta_init)

        sum_mu += mu_batch.sum(dim=0)
        sum_pearson += ((y_batch - mu_batch)**2 / mu_batch).sum(dim=0)

    # Degrees of freedom correction
    df = n_total - p_mean
    disp_init = sum_mu / torch.clamp(sum_pearson - df, min=0.1)

    gamma_init = torch.zeros((p_disp, n_genes), device=device)
    gamma_init[0, :] = torch.log(torch.clamp(disp_init, min=0.1))
    return beta_init, gamma_init


# ==============================================================================
# Batch Log-Likelihood
# ==============================================================================

def compute_batch_loglikelihood(y, mu, r):
    """
    Compute the negative binomial log-likelihood for a batch.

    Formula:
        ℓ = Σ [log Γ(Y+θ) - log Γ(θ) - log Γ(Y+1) + θ log θ + Y log μ - (Y+θ)log(μ+θ)]

    Args:
        y: Observed counts (n_batch × m_active)
        mu: Predicted means (n_batch × m_active)
        r: Dispersion parameters (n_batch × m_active)

    Returns:
        Total log-likelihood per response (m_active,)
    """
    # Note: log Γ(Y+1) is omitted if only used for relative change checks
    # between iterations on the same batch.
    log_r = torch.log(r)
    ll = (
        torch.lgamma(y + r) - torch.lgamma(r)
        + r * log_r + y * torch.log(mu)
        - (y + r) * torch.log(mu + r)
    )
    return torch.sum(ll, dim=0)


# ==============================================================================
# Stochastic IRLS Step
# ==============================================================================

def step_stochastic_irls(y, X, Z, beta, gamma, eta=0.8, tol=1e-4):
    """
    Perform a single damped Newton-Raphson update on a minibatch.

    Logic:
        1. Compute log-likelihood with current coefficients.
        2. Perform one IRLS step for Mean (Beta) and Dispersion (Gamma).
        3. Re-compute log-likelihood to determine convergence.
        4. Return updated coefficients and boolean convergence mask.

    Args:
        y: Count batch (n × m)
        X: Mean design matrix (n × p)
        Z: Dispersion design matrix (n × q)
        beta: Current mean coefficients (p × m)
        gamma: Current dispersion coefficients (q × m)
        eta: Damping factor (learning rate), 1.0 is pure Newton step.
        tol: Relative log-likelihood change threshold for convergence.

    Returns:
        beta_next: Updated mean coefficients (p × m)
        gamma_next: Updated dispersion coefficients (q × m)
        converged: Boolean mask of converged responses (m,)
    """
    # --- 1. Baseline Likelihood ---
    mu_old = torch.exp(X @ beta)
    r_old = torch.exp(Z @ gamma)
    ll_old = compute_batch_loglikelihood(y, mu_old, r_old)

    # --- 2. Update Mean (Beta) ---
    # Working weights W = μ/(1 + μ/θ)
    beta_target = update_mean_coefficients(X, y, beta, r_old)
    beta_next = (1 - eta) * beta + eta * beta_target

    # --- 3. Update Dispersion (Gamma) ---
    # Update depends on the latest mean estimates
    mu_next = torch.exp(X @ beta_next)
    gamma_target = update_dispersion_coefficients(Z, y, mu_next, gamma)
    gamma_target = torch.clamp(gamma_target, min=-10.0, max=10.0)
    gamma_next = (1 - eta) * gamma + eta * gamma_target

    # --- 4. Convergence Check ---
    r_next = torch.exp(Z @ gamma_next)
    ll_next = compute_batch_loglikelihood(y, mu_next, r_next)

    # Relative improvement in the objective function
    rel_change = torch.abs(ll_next - ll_old) / (torch.abs(ll_old) + 1e-10)
    converged = rel_change <= tol

    return beta_next, gamma_next, converged, rel_change