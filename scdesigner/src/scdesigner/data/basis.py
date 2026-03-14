import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import svd


def tps_kernel(A, B, eps):
    D = cdist(A, B)
    D = np.maximum(D, eps)
    K = D**2 * np.log(D)
    return K


def rbf_kernel(A, B, length_scale):
    D = cdist(A, B)
    return np.exp(-0.5 * (D / length_scale) ** 2)


def tps_basis(coords, df=15, n_landmarks=200, eps=1e-6, max_penalty_ratio=1e3,
              standardize=False):
    """Low-rank TPS basis via Nystrom approximation.

    Parameters
    ----------
    standardize : bool
        If True, rescale columns to unit variance. Use True for penalized
        fits (so penalty and coefficients are on the same scale) and False
        for unpenalized fits (where the natural column scaling provides
        implicit regularization of wiggly components).
    """
    n = coords.shape[0]
    idx = np.random.choice(n, size=min(n_landmarks, n), replace=False)
    landmarks = coords[idx]

    K_mm = tps_kernel(landmarks, landmarks, eps)
    K_nm = tps_kernel(coords, landmarks, eps)

    P = np.column_stack([np.ones(len(idx)), landmarks])
    Q, _ = np.linalg.qr(P, mode="reduced")
    proj = np.eye(len(idx)) - Q @ Q.T
    K_mm = proj @ K_mm @ proj

    U_m, s_m, _ = svd(K_mm, full_matrices=False)
    keep = s_m > eps
    U_m, s_m = U_m[:, keep], s_m[keep]

    basis = K_nm @ U_m / s_m
    basis = basis[:, :df]
    s_trunc = s_m[:df]

    if standardize:
        col_sd = np.std(basis, axis=0, keepdims=True)
        col_sd = np.maximum(col_sd, eps)
        basis = basis / col_sd

    penalty_diag = np.minimum(s_trunc[0] / s_trunc, max_penalty_ratio)
    return basis, penalty_diag


def gp_basis(coords, df=15, n_landmarks=200, length_scale=1.0, eps=1e-6,
             max_penalty_ratio=1e3, standardize=False):
    """Low-rank GP basis via Nystrom approximation.

    Parameters
    ----------
    standardize : bool
        If True, rescale columns to unit variance. Use True for penalized
        fits (so penalty and coefficients are on the same scale) and False
        for unpenalized fits (where the natural column scaling provides
        implicit regularization of wiggly components).
    """
    n = coords.shape[0]
    idx = np.random.choice(n, size=min(n_landmarks, n), replace=False)
    landmarks = coords[idx]

    K_mm = rbf_kernel(landmarks, landmarks, length_scale)
    K_nm = rbf_kernel(coords, landmarks, length_scale)

    K_mm += eps * np.eye(len(idx))
    U_m, s_m, _ = svd(K_mm, full_matrices=False)

    basis = K_nm @ U_m / s_m
    basis = basis[:, :df]
    s_trunc = s_m[:df]

    if standardize:
        col_sd = np.std(basis, axis=0, keepdims=True)
        col_sd = np.maximum(col_sd, eps)
        basis = basis / col_sd

    penalty_diag = np.minimum(s_trunc[0] / s_trunc, max_penalty_ratio)
    return basis, penalty_diag


def add_spatial_basis(adata, method="tps", df=15, prefix="sp_basis_", **kwargs):
    """Add 2D spatial basis columns to adata.obs."""
    coords = np.column_stack([
        adata.obs["spatial1"].values,
        adata.obs["spatial2"].values,
    ])

    if method == "tps":
        basis, penalty_diag = tps_basis(coords, df=df, **kwargs)
    elif method == "gp":
        basis, penalty_diag = gp_basis(coords, df=df, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    basis_cols = [f"{prefix}{i}" for i in range(basis.shape[1])]
    for i, col in enumerate(basis_cols):
        adata.obs[col] = basis[:, i]
    return adata, basis_cols, penalty_diag


def basis_formula(basis_cols, extra_terms=None):
    """Build a formula string referencing pre-computed basis columns."""
    parts = ["~ 0"] if extra_terms else ["~ 1"]
    if extra_terms:
        parts.extend(extra_terms)
    parts.extend(basis_cols)
    return " + ".join(parts)