import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import svd


def tps_kernel(A, B, eps):
    """Thin-plate spline kernel: K[i,j] = ||A_i - B_j||^2 log ||A_i - B_j||.

    Parameters
    ----------
    A : ndarray of shape (n, d)
        First set of points.
    B : ndarray of shape (m, d)
        Second set of points.
    eps : float
        Epsilon parameter, preventing log(0).

    Returns
    -------
    K : ndarray of shape (n, m)

    Examples
    --------
    >>> K = tps_kernel(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), eps=1e-6)
    >>> K  # doctest: +SKIP
    """
    D = cdist(A, B)
    D = np.maximum(D, eps)
    K = D**2 * np.log(D)
    return K


def rbf_kernel(A, B, length_scale):
    """Radial basis function kernel: K[i,j] = exp(-||A_i - B_j||^2 / (2 * length_scale^2)).

    Parameters
    ----------
    A : ndarray of shape (n, d)
        First set of points.
    B : ndarray of shape (m, d)
        Second set of points.
    length_scale : float
        Controls the spatial range of correlation; larger values give smoother functions.

    Returns
    -------
    K : ndarray of shape (n, m)

    Examples
    --------
    >>> K = rbf_kernel(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), length_scale=1.0)
    >>> float(K[0, 0])  # doctest: +SKIP
    """
    D = cdist(A, B)
    return np.exp(-0.5 * (D / length_scale) ** 2)


def tps_basis(coords, df=15, n_landmarks=200, eps=1e-6, max_penalty_ratio=1e3,
              standardize=False):
    """Low-rank TPS basis using a Nystrom approximation.

    Selects `n_landmarks` points, projects out the null space of the TPS
    penalty (intercept + linear terms), and returns the leading `df` eigenvectors
    of the projected landmark kernel as spatial basis columns.

    Parameters
    ----------
    coords : ndarray of shape (n, 2)
        2D spatial coordinates, one row per observation.
    df : int
        Number of basis columns to retain (i.e. rank of the approximation).
    n_landmarks : int
        Number of landmark points used in the Nystrom approximation. Larger
        values improve accuracy at higher computational cost.
    eps : float
        Epsilon parameter for the TPS kernel and eigenvalue threshold for
        discarding near-zero singular values.
    max_penalty_ratio : float
        Cap on penalty_diag[j] = s_0 / s_j. Prevents extreme down-weighting of
        high-frequency components when eigenvalues decay sharply.
    standardize : bool
        If True, rescale columns to unit variance. Use True for penalized
        fits (so penalty and coefficients are on the same scale) and False
        for unpenalized fits (where the natural column scaling provides
        implicit regularization of wiggly components).

    Returns
    -------
    basis : ndarray of shape (n, df)
        Spatial basis matrix.
    penalty_diag : ndarray of shape (df,)
        Per-column penalty weights proportional to s_0 / s_j, where s_j is
        the j-th eigenvalue of the projected landmark kernel.

    Examples
    --------
    >>> coords = np.random.uniform(size=(100, 2))
    >>> basis, penalty = tps_basis(coords, df=10)
    >>> basis.shape, penalty.shape  # doctest: +SKIP
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
    """Low-rank GP basis using a Nystrom approximation with an RBF kernel.

    Selects `n_landmarks` points, regularizes the landmark kernel with a
    diagonal `eps` shift, and returns the leading `df` eigenvectors as spatial
    basis columns.

    Parameters
    ----------
    coords : ndarray of shape (n, 2)
        2D spatial coordinates, one row per observation.
    df : int
        Number of basis columns to retain (i.e. rank of the approximation).
    n_landmarks : int
        Number of landmark points used in the Nystrom approximation. Larger
        values improve accuracy at higher computational cost.
    length_scale : float
        RBF kernel length scale. Smaller values capture shorter-range spatial
        structure; larger values produce smoother functions.
    eps : float
        Diagonal regularization added to the landmark kernel before SVD.
    max_penalty_ratio : float
        Cap on penalty_diag[j] = s_0 / s_j. Prevents extreme down-weighting of
        high-frequency components when eigenvalues decay sharply.
    standardize : bool
        If True, rescale columns to unit variance. Use True for penalized
        fits (so penalty and coefficients are on the same scale) and False
        for unpenalized fits (where the natural column scaling provides
        implicit regularization of high frequency components).

    Returns
    -------
    basis : ndarray of shape (n, df)
        Spatial basis matrix.
    penalty_diag : ndarray of shape (df,)
        Per-column penalty weights proportional to s_0 / s_j, where s_j is
        the j-th eigenvalue of the (regularized) landmark kernel.

    Examples
    --------
    >>> coords = np.random.uniform(size=(100, 2))
    >>> basis, penalty = gp_basis(coords, df=10, length_scale=0.3)
    >>> basis.shape, penalty.shape  # doctest: +SKIP
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
    """Add 2D spatial basis columns to adata.obs.

    Reads ``spatial1`` and ``spatial2`` from ``adata.obs``, constructs a
    low-rank spatial basis, and appends each column as a new obs variable.

    Parameters
    ----------
    adata : AnnData
        Annotated data object. Must contain ``adata.obs["spatial1"]`` and
        ``adata.obs["spatial2"]``.
    method : {"tps", "gp"}
        Basis construction method. ``"tps"`` uses a thin-plate spline kernel;
        ``"gp"`` uses an RBF kernel.
    df : int
        Number of basis columns to add (passed to the underlying basis function).
    prefix : str
        Column name prefix; columns are named ``{prefix}0``, ``{prefix}1``, ...
    **kwargs
        Additional keyword arguments forwarded to :func:`tps_basis` or
        :func:`gp_basis` (e.g. ``n_landmarks``, ``length_scale``,
        ``standardize``).

    Returns
    -------
    adata : AnnData
        Input object with basis columns appended to ``adata.obs`` in-place.
    basis_cols : list of str
        Names of the newly added columns.
    penalty_diag : ndarray of shape (df,)
        Per-column penalty weights from the chosen basis function.

    Examples
    --------
    >>> import anndata, pandas as pd
    >>> obs = pd.DataFrame({"spatial1": np.random.uniform(size=50),
    ...                     "spatial2": np.random.uniform(size=50)})
    >>> adata = anndata.AnnData(obs=obs)
    >>> adata, cols, penalty = add_spatial_basis(adata, method="tps", df=5)
    """
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
    """Build a formula string referencing pre-computed basis columns.

    When ``extra_terms`` is provided the intercept is suppressed (``~ 0``) so
    that ``extra_terms`` can supply their own reference levels; otherwise a
    standard intercept is included (``~ 1``).

    Parameters
    ----------
    basis_cols : list of str
        Names of the basis columns (e.g. from :func:`add_spatial_basis`).
    extra_terms : list of str or None
        Additional covariate terms inserted after the intercept/zero term and
        before the basis columns. ``None`` adds no extra terms.

    Returns
    -------
    formula : str
        Formula string that can be used in formulaic.

    Examples
    --------
    >>> basis_formula(["sp_basis_0", "sp_basis_1"])
    '~ 1 + sp_basis_0 + sp_basis_1'
    >>> basis_formula(["sp_basis_0"], extra_terms=["condition"])
    '~ 0 + condition + sp_basis_0'
    """
    parts = ["~ 0"] if extra_terms else ["~ 1"]
    if extra_terms:
        parts.extend(extra_terms)
    parts.extend(basis_cols)
    return " + ".join(parts)