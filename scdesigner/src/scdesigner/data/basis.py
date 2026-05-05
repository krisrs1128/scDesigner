from abc import ABC, abstractmethod
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


class SpatialBasis(ABC):
    """Low-rank 2D spatial basis with stateful fit / transform.

    Subclasses define ``_kernel`` and ``_regularize`` (applied to the
    landmark kernel before SVD). ``fit`` picks landmarks, stores the
    projection ``U_m[:, :df] / s_m[:df]``, training column std (if
    ``standardize``), and the penalty diagonal ``s_0 / s_j`` (capped).
    ``transform`` reuses that state on new coordinates.

    Parameters
    ----------
    df : int
        Basis columns to retain.
    n_landmarks : int
        Landmark count for the Nystrom approximation.
    eps : float
        Numerical floor for kernel and column-std.
    max_penalty_ratio : float
        Cap on ``penalty_diag[j]``.
    standardize : bool
        Divide columns by training std.

    Attributes
    ----------
    landmarks_, projection_, col_sd_, penalty_diag
        Set by :meth:`fit`. Shapes ``(m, 2)``, ``(m, df)``, ``(df,)``, ``(df,)``.
    """

    def __init__(self, df=15, n_landmarks=200, eps=1e-6,
                 max_penalty_ratio=1e3, standardize=False):
        self.df = df
        self.n_landmarks = n_landmarks
        self.eps = eps
        self.max_penalty_ratio = max_penalty_ratio
        self.standardize = standardize

        self.landmarks_ = None
        self.projection_ = None
        self.col_sd_ = None
        self.penalty_diag = None

    @abstractmethod
    def _kernel(self, A, B):
        """Kernel matrix between rows of ``A`` and rows of ``B``."""

    @abstractmethod
    def _regularize(self, K_mm, landmarks):
        """Transform the landmark kernel before SVD."""

    def _truncate_singular(self, U_m, s_m):
        """Drop near-zero singular values before truncating to ``df``."""
        return U_m, s_m

    def fit(self, coords):
        coords = np.asarray(coords)
        n = coords.shape[0]
        idx = np.random.choice(n, size=min(self.n_landmarks, n), replace=False)
        landmarks = coords[idx]

        K_mm = self._kernel(landmarks, landmarks)
        K_mm = self._regularize(K_mm, landmarks)
        U_m, s_m, _ = svd(K_mm, full_matrices=False)
        U_m, s_m = self._truncate_singular(U_m, s_m)

        df = self.df
        self.landmarks_ = landmarks
        self.projection_ = U_m[:, :df] / s_m[:df]

        s_trunc = s_m[:df]
        self.penalty_diag = np.minimum(
            s_trunc[0] / s_trunc, self.max_penalty_ratio
        )

        if self.standardize:
            basis = self._kernel(coords, landmarks) @ self.projection_
            col_sd = basis.std(axis=0)
            self.col_sd_ = np.maximum(col_sd, self.eps)
        return self

    def transform(self, coords):
        if self.landmarks_ is None:
            raise RuntimeError("SpatialBasis.transform called before fit.")
        coords = np.asarray(coords)
        basis = self._kernel(coords, self.landmarks_) @ self.projection_
        if self.standardize:
            basis = basis / self.col_sd_
        return basis

    def fit_transform(self, coords):
        return self.fit(coords).transform(coords)


class TPSBasis(SpatialBasis):
    """Thin-plate spline basis (Nystrom).

    Projects out the TPS penalty null space (intercept + linear terms over
    landmarks) before SVD; drops singular values below ``eps``.
    """

    def _kernel(self, A, B):
        return tps_kernel(A, B, self.eps)

    def _regularize(self, K_mm, landmarks):
        P = np.column_stack([np.ones(len(landmarks)), landmarks])
        Q, _ = np.linalg.qr(P, mode="reduced")
        proj = np.eye(len(landmarks)) - Q @ Q.T
        return proj @ K_mm @ proj

    def _truncate_singular(self, U_m, s_m):
        keep = s_m > self.eps
        return U_m[:, keep], s_m[keep]


class GPBasis(SpatialBasis):
    """Gaussian-process basis with an RBF kernel (Nystrom).

    Regularizes the landmark kernel with ``eps * I`` before SVD.

    Parameters
    ----------
    length_scale : float
        RBF length scale. Smaller = shorter-range structure.
    **kwargs
        Forwarded to :class:`SpatialBasis`.
    """

    def __init__(self, length_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.length_scale = length_scale

    def _kernel(self, A, B):
        return rbf_kernel(A, B, self.length_scale)

    def _regularize(self, K_mm, landmarks):
        return K_mm + self.eps * np.eye(len(landmarks))


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