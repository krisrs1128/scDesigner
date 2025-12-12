from ..data.formula import standardize_formula
from ..data.loader import _to_numpy
from ..base.simulator import Simulator
from anndata import AnnData
from formulaic import model_matrix
from scipy.stats import gamma
from typing import Union, Dict
import numpy as np
import pandas as pd
import torch

################################################################################
## Functions for estimating PNMF regression
################################################################################

# computes PNMF weight and score, ncol specify the number of clusters
def pnmf(log_data, nbase=3, **kwargs):  # data is np array, log transformed read data
    """
    Estimate PNMF components from log-transformed expression counts.

    Parameters
    ----------
    log_data : np.ndarray
        Log-transformed expression matrix (genes × cells).
    nbase : int, optional
        Number of latent PNMF bases to extract.
    **kwargs
        Additional arguments forwarded to :func:`pnmf_eucdist`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing the learned PNMF weight matrix ``W`` and score matrix
        ``S`` (pseudo-basis loadings for each cell).
    """
    U = left_singular(log_data, nbase)
    W = pnmf_eucdist(log_data, U, **kwargs)
    W = W / np.linalg.norm(W, ord=2)
    S = W.T @ log_data
    return W, S


def gamma_regression_array(
    x: np.array, y: np.array, lr: float = 0.1, epochs: int = 40
) -> dict:
    """
    Fit gamma regression coefficients in a batched regression context.

    Parameters
    ----------
    x : np.ndarray
        Design matrix for covariates (cells × covariates).
    y : np.ndarray
        Target matrix (cells × latent features) derived from PNMF scores.
    lr : float, optional
        Learning rate for the Adam optimizer.
    epochs : int, optional
        Number of training epochs.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing estimated ``"a"``, ``"loc"``, and ``"beta"``
        regression coefficients shaped (covariates, outcomes).
    """
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    n_features, n_outcomes = x.shape[1], y.shape[1]
    a = torch.zeros(n_features * n_outcomes, requires_grad=True)
    loc = torch.zeros(n_features * n_outcomes, requires_grad=True)
    beta = torch.zeros(n_features * n_outcomes, requires_grad=True)
    optimizer = torch.optim.Adam([a, loc, beta], lr=lr)

    for i in range(epochs):
        optimizer.zero_grad()
        loss = negative_gamma_log_likelihood(a, beta, loc, x, y)
        loss.backward()
        optimizer.step()

    a, loc, beta = _to_numpy(a, loc, beta)
    a = a.reshape(n_features, n_outcomes)
    loc = loc.reshape(n_features, n_outcomes)
    beta = beta.reshape(n_features, n_outcomes)
    return {"a": a, "loc": loc, "beta": beta}


def class_generator(score, n_clusters=3):
    """
    Cluster PNMF scores and return discrete class labels. (This function is not used in the current implementation.)

    Parameters
    ----------
    score : np.ndarray
        PNMF scores (latent factors) of shape (n_features, n_cells).
    n_clusters : int, optional
        Number of target clusters for grouping the scores.

    Returns
    -------
    np.ndarray
        Array of cluster labels of length ``n_cells``.
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters, random_state=0)  # Specify the number of clusters
    kmeans.fit(score.T)
    labels = kmeans.labels_
    return labels


###############################################################################
## Helpers for deriving PNMF
###############################################################################


def pnmf_eucdist(X, W_init, maxIter=500, threshold=1e-4, tol=1e-10, verbose=False, **kwargs):
    """
    Optimize PNMF weights via Euclidean distance minimization.

    Parameters
    ----------
    X : np.ndarray
        Input expression matrix (genes × cells).
    W_init : np.ndarray
        Initial estimate of the weight matrix.
    maxIter : int, optional
        Maximum number of iterations.
    threshold : float, optional
        Convergence threshold on relative weight change.
    tol : float, optional
        Numeric tolerance for truncating small entries.
    verbose : bool, optional
        If True, print progress every 10 iterations.
    **kwargs
        Reserved for future options.

    Returns
    -------
    np.ndarray
        Normalized PNMF weight matrix with positive entries.
    """
    # initialization
    W = W_init  # initial W is the PCA of X
    XX = X @ X.T

    # iterations
    for iter in range(maxIter):
        if verbose and (iter + 1) % 10 == 0:
            print("%d iterations used." % (iter + 1))
        W_old = W

        XXW = XX @ W
        SclFactor = np.dot(W, W.T @ XXW) + np.dot(XXW, W.T @ W)

        # QuotientLB
        SclFactor = MatFindlb(SclFactor, tol)
        SclFactor = XXW / SclFactor
        W = W * SclFactor  # somehow W *= SclFactor doesn't work?

        norm_W = np.linalg.norm(W)
        W /= norm_W
        W = MatFind(W, tol)

        diffW = np.linalg.norm(W_old - W) / np.linalg.norm(W_old)
        if diffW < threshold:
            break

    return W


# left singular vector of X
def left_singular(X, k):
    """
    Extract the top `k` left singular vectors of matrix `X` for initialization.
    """
    from scipy.sparse.linalg import svds
    U, _, _ = svds(X, k=k)
    return np.abs(U)


def MatFindlb(A, lb):
    """
    Clamp matrix A's entries to be greater than or equal to a lower bound `lb`.
    """
    B = np.ones(A.shape) * lb
    Alb = np.where(A < lb, B, A)
    return Alb


def MatFind(A, ZeroThres):
    """
    Zero out values below a threshold.
    """
    B = np.zeros(A.shape)
    Atrunc = np.where(A < ZeroThres, B, A)
    return Atrunc


###############################################################################
## Helpers for training PNMF regression
###############################################################################


def shifted_gamma_pdf(x, alpha, beta, loc):
    """
    Compute penalized negative log probability density function for shifted gamma. 
    A huge penalty is applied to values below the location parameter to ensure 
    the parameters fall within the support of the gamma distribution.

    Parameters
    ----------
    x : torch.Tensor or array-like
        Observed values.
    alpha : torch.Tensor
        Shape parameters.
    beta : torch.Tensor
        Rate parameters.
    loc : torch.Tensor
        Location parameters (shift).

    Returns
    -------
    torch.Tensor
        Mean negative log likelihood over ``x``.
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    mask = x < loc
    y_clamped = torch.clamp(x - loc, min=1e-12)

    log_pdf = (
        alpha * torch.log(beta)
        - torch.lgamma(alpha)
        + (alpha - 1) * torch.log(y_clamped)
        - beta * y_clamped
    )
    loss = -torch.mean(log_pdf[~mask])
    n_invalid = mask.sum()
    if n_invalid > 0:  # force samples to be greater than loc
        loss = loss + 1e10 * n_invalid.float()
    return loss


def negative_gamma_log_likelihood(log_a, log_beta, loc, X, y):
    """
    Compute the (negative) log likelihood over a gamma regression layer.

    Parameters
    ----------
    log_a : torch.Tensor
        Log-scale shape coefficients of shape (covariates × outcomes).
    log_beta : torch.Tensor
        Log-scale rate coefficients of shape (covariates × outcomes).
    loc : torch.Tensor
        Location coefficients of shape (covariates × outcomes).
    X : torch.Tensor
        Design matrix (cells × covariates).
    y : torch.Tensor
        Observed PNMF scores (cells × outcomes).

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the mean negative log-likelihood.
    """
    n_features = X.shape[1]
    n_outcomes = y.shape[1]

    a = torch.exp(log_a.reshape(n_features, n_outcomes))
    beta = torch.exp(log_beta.reshape(n_features, n_outcomes))
    loc = loc.reshape(n_features, n_outcomes)
    return shifted_gamma_pdf(y, X @ a, X @ beta, X @ loc)

def format_gamma_parameters(
    parameters: dict,
    W_index: list,
    coef_index: list,
) -> dict:
    """
    Format gamma regression parameters as DataFrames for downstream use.

    Parameters
    ----------
    parameters : dict
        Dictionary containing ``"a"``, ``"loc"``, ``"beta"``, and ``"W"`` arrays.
    W_index : list
        Row labels for the PNMF weights (typically gene names).
    coef_index : list
        Row labels for the regression coefficients (typically covariate names).

    Returns
    -------
    dict
        Updated dictionary with DataFrames stored under the original keys.
    """
    parameters["a"] = pd.DataFrame(parameters["a"], index=coef_index)
    parameters["loc"] = pd.DataFrame(parameters["loc"], index=coef_index)
    parameters["beta"] = pd.DataFrame(parameters["beta"], index=coef_index)
    parameters["W"] = pd.DataFrame(parameters["W"], index=W_index)
    return parameters


################################################################################
## Associated PNMF Objects
################################################################################

class PositiveNMF(Simulator):
    """
    Positive nonnegative matrix factorization (PNMF) simulator with gamma regression.

    This simulator fits a low-rank positive factorization on log-transformed
    expression data and then models the resulting latent scores using a
    covariate-dependent shifted gamma distribution. Sampling proceeds by
    drawing gamma latent scores and mapping them back to the gene space via the
    learned PNMF weights.

    Parameters
    ----------
    formula : dict or str
        Mean-model formula for the gamma regression. If a string is provided,
        it is interpreted as the mean formula and stored under the key
        ``"mean"``. The formula is evaluated against ``adata.obs`` via
        :func:`formulaic.model_matrix`.
    **kwargs
        Keyword arguments forwarded to :func:`pnmf` (e.g. ``nbase``, ``maxIter``).

    Attributes
    ----------
    formula : dict
        Standardized formula dictionary containing at least the ``"mean"`` key.
    parameters : dict or None
        Fitted parameters after calling :meth:`fit`. Keys include:

        * ``"a"`` (:class:`pandas.DataFrame`): gamma shape regression coefficients.
        * ``"loc"`` (:class:`pandas.DataFrame`): gamma location regression coefficients.
        * ``"beta"`` (:class:`pandas.DataFrame`): gamma rate regression coefficients.
        * ``"W"`` (:class:`pandas.DataFrame`): PNMF weight matrix mapping latent
          scores to genes.
    n_outcomes : int
        Number of simulated outcomes (cells) in the training data.
    columns : pandas.Index
        Column names of the design matrix produced from ``formula["mean"]``.

    Examples
    --------
    Fit a PNMF simulator, inspect fitted parameters, and generate samples:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from anndata import AnnData
    >>> from scdesigner.simulators import PositiveNMF
    >>>
    >>> rng = np.random.default_rng(0)
    >>> X = rng.poisson(lam=2.0, size=(50, 20)).astype(float)  # (cells × genes)
    >>> obs = pd.DataFrame({"condition": rng.choice(["A", "B"], size=50)})
    >>> adata = AnnData(X=X, obs=obs)
    >>> adata.var_names = [f"g{i}" for i in range(adata.n_vars)]
    >>>
    >>> sim = PositiveNMF("~ 1 + condition", nbase=3, maxIter=50)
    >>> sim.fit(adata, lr=0.1)
    >>> isinstance(sim.parameters, dict)
    True
    >>>
    >>> # Predict gamma parameters for new observations
    >>> new_obs = pd.DataFrame({"condition": ["A", "B", "A"]})
    >>> pred = sim.predict(new_obs)
    >>> sorted(pred.keys())
    ['a', 'beta', 'loc']
    >>>
    >>> # Sample a new dataset with the same genes
    >>> adata_sim = sim.sample(new_obs)
    >>> adata_sim.n_obs == 3 and adata_sim.n_vars == adata.n_vars
    True
    """
    def __init__(self, formula: Union[Dict, str], **kwargs):
        """
        Parameters
        ----------
        formula : dict or str
            Formula describing the mean model for the gamma regression.
        **kwargs
            Keyword arguments passed through to :func:`pnmf`.
        """
        self.formula = standardize_formula(formula, allowed_keys=['mean'])
        self.parameters = None
        self._hyperparams = kwargs


    def setup_data(self, adata: AnnData, **kwargs):
        self.log_data = np.log1p(adata.X).T #(genes x cells)
        self.n_outcomes = self.log_data.shape[1]
        self._template = adata
        self.x = model_matrix(self.formula["mean"], adata.obs)
        self.columns = self.x.columns
        self.x = np.asarray(self.x)


    def fit(self, adata: AnnData, lr: float=0.1):
        """
        Fit the PNMF marginals on the provided AnnData.

        Parameters
        ----------
        adata : AnnData
            Dataset used to estimate PNMF weights and gamma coefficients.
        lr : float, optional
            Learning rate for the gamma regression solver.
        """
        self.setup_data(adata)
        W, S = pnmf(self.log_data, **self._hyperparams)
        parameters = gamma_regression_array(self.x, S.T, lr)
        parameters["W"] = W
        self.parameters = format_gamma_parameters(
            parameters, list(self._template.var_names), list(self.columns)
        )


    def predict(self, obs=None, **kwargs):
        """
        Predict gamma regression parameters for new observations.

        Parameters
        ----------
        obs : pandas.DataFrame, optional
            Observation metadata used to construct the design matrix. Defaults
            to the training ``AnnData`` observations.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with ``"a"``, ``"loc"``, and ``"beta"`` arrays for each
            target feature and observation.
        """
        if obs is None:
            obs = self._template.obs

        x = model_matrix(self.formula["mean"], obs)
        a, loc, beta = (
            x @ np.exp(self.parameters["a"]),
            x @ self.parameters["loc"],
            x @ np.exp(self.parameters["beta"]),
        )
        return {"a": a, "loc": loc, "beta": beta}


    def sample(self, obs=None):
        """
        Generate expression samples from the fitted model.

        Parameters
        ----------
        obs : pandas.DataFrame, optional
            Metadata for the observations to simulate. Defaults to the training
            ``AnnData`` annotations.

        Returns
        -------
        AnnData
            Simulated :class:`AnnData` matrix containing generated expression
            counts on the original feature ordering.
        """
        if obs is None:
            obs = self._template.obs
        W = self.parameters["W"]
        parameters = self.predict(obs)
        a, loc, beta = parameters["a"], parameters["loc"], parameters["beta"]
        sim_score = gamma(a, loc, 1 / beta).rvs()
        samples = np.exp(W @ sim_score.T).T

        # thresholding samples
        floor = np.floor(samples)
        samples = floor + np.where(samples - floor < 0.9, 0, 1) - 1
        samples = np.where(samples < 0, 0, samples)

        result = AnnData(X=samples, obs=obs)
        result.var_names = self._template.var_names
        return result