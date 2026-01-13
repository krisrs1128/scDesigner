from ..utils.kwargs import DEFAULT_ALLOWED_KWARGS, _filter_kwargs
from ..data.loader import adata_loader, get_device
from anndata import AnnData
from typing import Union, Dict, Optional, Tuple
import pandas as pd
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Marginal(ABC):
    """
    A Feature-wise Marginal Model

    This is a class for handling feature-wise (e.g., gene-level) modeling that
    ignores any correlation between features. For example, it can be used to
    model the relationship between experimental design features, such as cell
    type or treatment, and the parameters of a collection of negative binomial
    models (one for each gene).

    These marginals can be plugged into a copula to build the complete scDesign3
    simulator. Methods are expected for estimating model parameters, evaluating
    likelihoods on new samples, and generating new samples given conditioning
    information. Since these marginal models are intended to be used within a
    copula, they should also provide utilities for evaluating quantiles and
    computing cumulative distribution functions.

    Parameters
    ----------
    formula : dict or str
        A dictionary or string specifying the relationship between the columns
        of an input data frame (adata.obs, adata.var, or similar attributes) and
        the parameters of the marginal model. If only a string is provided,
        then the means are allowed to depend on the design parameters, while all
        other parameters are treated as fixed. If a dictionary is provided,
        each key should correspond to a parameter. The string values should be
        in a format that can be parsed by the formulaic package.  For example,
        '~ x' will ensure that the parameter varies linearly with X.

    Attributes
    ----------
    formula : dict or str
        A dictionary or string specifying the relationship between the columns
        of an input data frame (adata.obs, adata.var, or similar attributes) and
        the parameters of the marginal model. If only a string is provided,
        then the means are allowed to depend on the design parameters, while all
        other parameters are treated as fixed. If a dictionary is provided,
        each key should correspond to a parameter. The string values should be
        in a format that can be parsed by the formulaic package.  For example,
        '~ x' will ensure that the parameter varies linearly with X.

    feature_dims : dict
        A dictionary containing the number of predictors associated with each
        distributional parameter.  Note that this number is repeated for every
        feature (e.g., gene) in the marginal model.  This information is often
        useful for computing the complexity of the estimated model.

    loader : torch.utils.data.DataLoader
        A torch DataLoader object that returns batches of data for use during
        training. This loader is constructed internally within the
        setup_optimizer method. Enumerating this loader returns a tuple: the
        first element contains a tensor of feature measurements (y), and the
        second element is a dictionary of tensors containing predictors to use
        for each parameter (x, for each parameter theta(x)). This design is
        useful because the design matrices may differ between parameters of the
        marginal model, y | x ~ F_(theta(x))(y)

    n_outcomes : int
        The number of features modeled by this marginal model. For example,
        this corresponds to the number of genes being simulated.

    predict : nn.Module
        A torch.nn.Module storing the relationship between predictors for each
        parameter and the predicted feature-wise outcomes. This module is expected
        to take the second element of the tuple defined by each batch and then
        predict a tensor with the same shape as the first element of the batch
        tuple.

    predictor_names : dict of list of str
        A dictionary whose keys are the parameter names associated with this
        marginal model. The values for each key are the names of predictors in
        the design matrix implied by the associated formula. Note that these
        names may have been expanded from the original formula specification.
        For example, if cell_type is included in the formula, then the predictor
        names will include the unique levels of cell_type as separate columns in
        the design matrix and therefore as separate elements in this list.

    parameters : dict of pandas.DataFrame
        A dictionary whose keys are the parameter names associated with this
        marginal model. The values for each key are pandas DataFrames storing
        the fitted parameter values. The rows of each DataFrame are the
        experimental features specified by the associated formula object (the
        rownames are the same as those in predictor_names). The columns are
        features that are being predicted.

    device : torch.device
        The device on which the prediction module is stored.  This is
        automatically determined when calling the .fit method.

    Examples
    --------
    >>> class DummyModel(Marginal):
    ...     def fit(self):
    ...         pass
    ...
    ...     def likelihood(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]]):
    ...         pass
    ...
    ...     def uniformize(self, y: torch.Tensor, x: Dict[str, torch.Tensor]):
    ...         pass
    ...
    ...     def invert(self, u: torch.Tensor, x: Dict[str, torch.Tensor]):
    ...         pass
    ...
    ...     def setup_optimizer(self):
    ...         pass
    ...
    >>> model = DummyModel("~ cell_type")
    >>> model.fit()
    """
    def __init__(self, formula: Union[Dict, str], device: Optional[torch.device]=None):
        self.formula = formula
        self.feature_dims = None
        self.loader = None
        self.n_outcomes = None
        self.predict = None
        self.predictor_names = None
        self.parameters = None
        self.device = get_device(device)

    def setup_data(self, adata: AnnData, batch_size: int = 1024, **kwargs):
        """Set up the dataloader for the AnnData object.

        The simulator class definition doesn’t actually require any particular
        template dataset. This is helpful for reasoning about simulators
        abstractly, but when we actually want to estimate parameters, we need a
        template. This method takes an input template dataset and adds
        attributes to the simulator object for later estimation and sampling
        steps.

        Parameters
        ----------
        adata : AnnData
            This is the object on which we want to estimate the simulator. This
            serves as the template for all downstream fitting.
        batch_size : int
            The number of sample to return on each call of the data loader.
            Defaults to 1024.
        **kwargs : Any
            Other keyword arguments passed to data loader construction. Any
            argument recognized by the PyTorch DataLoader can be passed in here.

        Returns
        -------
        """
        # keep a reference to the AnnData for later use (e.g., var_names)
        self.adata = adata
        self.loader = adata_loader(adata, self.formula, batch_size=batch_size, **kwargs)
        X_batch, obs_batch = next(iter(self.loader))
        self.n_outcomes = X_batch.shape[1]
        self.feature_dims = {k: v.shape[1] for k, v in obs_batch.items()}
        self.predictor_names = self.loader.dataset.predictor_names

    def fit(self, max_epochs: int = 50, verbose: bool = True, **kwargs):
        """Fit the marginal predictor using vanilla PyTorch training loop.

        This method runs stochastic gradient optimization using the template
        dataset defined by the setup_data method. The specific optimizer used
        can be modified with the setup_optimizer method and defaults to Adam.

        Note that, unlike `fit` in class `Simulator`, this method does not allow
        the template dataset as input. This requires `.setup_data()` to be
        called first. We want to give finer-grained control over the data
        loading and optimization in this class relative to the specific
        `Simulator` implementations, which are designed to be easy to run with
        as few steps as possible.

        Parameters
        ----------
        max_epochs : int
            The maximum number of epochs. This is the number of times we feed
            through our cells in the dataset.
        verbose : bool
            Should we print intermediate training outputs?

        Returns
        -------
        None
            This method doesn't return anything but modifies the self.parameters
            attribute with the trained model parameters.
        """
        if self.predict is None:
            self.setup_optimizer(**kwargs)

        for epoch in range(max_epochs):
            epoch_loss, n_batches = 0.0, 0

            for batch in self.loader:
                y, x = batch
                if y.device != self.device:
                    y = y.to(self.device)
                    x = {k: v.to(self.device) for k, v in x.items()}

                self.predict.optimizer.zero_grad()
                loss = self.predict.loss_fn((y, x))
                loss.backward()
                self.predict.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            if verbose:
                print(f"Epoch {epoch}/{max_epochs}, Loss: {avg_loss:.4f}", end='\r')
        self.parameters = self.format_parameters()

    def format_parameters(self):
        """Convert fitted coefficient tensors into pandas DataFrames.

        Returns:
            dict: mapping from parameter name -> pandas.DataFrame with rows
                corresponding to predictor column names (from
                `self.predictor_names[param]`) and columns corresponding to
                `self.adata.var_names` (gene names). The values are moved to
                CPU and converted to numpy floats.
        """
        var_names = list(self.adata.var_names)

        dfs = {}
        for param, tensor in self.predict.coefs.items():
            coef_np = tensor.detach().cpu().numpy()
            row_names = list(self.predictor_names[param])
            dfs[param] = pd.DataFrame(coef_np, index=row_names, columns=var_names)
        return dfs

    def num_params(self):
        """Return the number of parameters.

        Count the number of parameters in the marginal simulator. Usually this
        is just the number of predictors times the number of genes, because we
        use a linear model. However, in specific implementations, it’s possible
        to use more flexible models, in which case the number of parameters
        would increase.
        """
        if self.predict is None:
            return 0
        return sum(p.numel() for p in self.predict.parameters() if p.requires_grad)

    @abstractmethod
    def setup_optimizer(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def likelihood(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute the log-likelihood for a batch.

        The likelihood is used for maximum likelihood estimation. It is also
        used when computing AIC and BIC scores, which are important when
        choosing an appropriate model complexity.

        Parameters
        ----------
        batch : tuple of (torch.Tensor, dict of str -> torch.Tensor)
            A tuple of gene expression (y) and experimental factors (x) used to
            evaluate the model likelihood. The first element of the tuple is a
            cells x genes tensor. The second is a a dictionary of tensors, with
            one key/value pair per parameter.  These tensors are the
            conditioning information to pass to the .predict() function of this
            distribution class. They are the numerical design matrices implied
            by the initializing formulas.

        Returns
        -------
        torch.Tensor
            A scalar containing the log-likelihood of the batch under the
            current model parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def invert(self, u: torch.Tensor, x: Dict[str, torch.Tensor]):
        """Invert pseudoobservations.

        Return a quantile from the distribution. This handles the link between
        the marginal and the copula model. The idea is that the copula will
        generate pseudo-observations on the unit cube. All the values will be
        between zero and one. By calling invert, we transform these zero-to-one
        valued pseudo-observations into observations in the original data space;
        values between zero and one can be thought of like quantiles.

        Parameters
        ----------
        u : torch.Tensor
            Scalars between [0, 1] that specify the quantile level we want.
        x : Dict of (str -> torch.Tensor)
            A dictionary of tensors, with one key/value pair per parameter.
            These tensors are the conditioning information to pass to the
            .predict() function of this distribution class. They are the
            numerical design matrices implied by the initializing formulas.

        Returns
        -------
        z : torch.Tensor
            A tensor with dimension dim(u) x num_genes. Each row gives the
            requested quantile for the marginal distributions across all genes.
        """
        raise NotImplementedError

    @abstractmethod
    def uniformize(self, y: torch.Tensor, x: Dict[str, torch.Tensor]):
        """Uniformize using learned CDF.

        Apply a quantile/CDF transformation to observations y, accounting for
        conditioning variables x. This step is used in training the copula
        model. Since the copula needs to operate on the unit cube, we need to
        transform the original data onto the unit cube. This can be done by
        applying a CDF transformation.

        Parameters
        ----------
        y: torch.Tensor
            A cells x genes tensor with gene expression levels across all cells.
        x : Dict of (str -> torch.Tensor)
            A dictionary of tensors, with one key/value pair per parameter.
            These tensors are the conditioning information to pass to the
            .predict() function of this distribution class. They are the
            numerical design matrices implied by the initializing formulas.
        """
        raise NotImplementedError


class GLMPredictor(nn.Module):
    """GLM-style predictor with arbitrary named parameters.

    Args:
        n_outcomes: number of model outputs (e.g. genes)
        feature_dims: mapping from param name -> number of covariate features
        link_fns: optional mapping from param name -> callable(link) applied to linear predictor

    The module will create one coefficient matrix per named parameter with shape
    (n_features_for_param, n_outcomes) and expose them as Parameters under
    `self.coefs[param_name]`.
    """
    def __init__(
        self,
        n_outcomes: int,
        feature_dims: Dict[str, int],
        link_fns: Dict[str, callable] = None,
        loss_fn: Optional[callable] = None,
        optimizer_class: Optional[callable] = torch.optim.AdamW,
        optimizer_kwargs: Optional[Dict] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.n_outcomes = int(n_outcomes)
        self.feature_dims = dict(feature_dims)
        self.param_names = list(self.feature_dims.keys())

        self.link_fns = link_fns or {k: torch.exp for k in self.param_names}
        self.coefs = nn.ParameterDict()
        for key, dim in self.feature_dims.items():
            self.coefs[key] = nn.Parameter(torch.zeros(dim, self.n_outcomes))
        self.reset_parameters()

        self.loss_fn = loss_fn
        self.to(get_device(device))

        optimizer_kwargs = optimizer_kwargs or {}
        filtered_kwargs = _filter_kwargs(optimizer_kwargs, DEFAULT_ALLOWED_KWARGS['optimizer'])
        self.optimizer = optimizer_class(self.parameters(), **filtered_kwargs)

    def reset_parameters(self):
        for p in self.coefs.values():
            nn.init.normal_(p, mean=0.0, std=1e-4)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward Pass for Given Covariates

        obs_dict : Dict of (str -> torch.Tensor)
            A dictionary of tensors, with one key/value pair per parameter.
            These tensors are the conditioning information to pass to the
            .predict() function of this distribution class. They are the
            numerical design matrices implied by the initializing formulas.
        """
        out = {}
        for name in self.param_names:
            x_beta = obs_dict[name] @ self.coefs[name]
            link = self.link_fns.get(name, torch.exp)
            out[name] = link(x_beta)
        return out