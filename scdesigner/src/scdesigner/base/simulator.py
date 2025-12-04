from anndata import AnnData
from typing import Dict
from pandas import DataFrame
from abc import abstractmethod

class Simulator:
    """Simulation abstract class

    This abstract simulator class defines the minimal methods that must be
    exported by every simulator in scdesigner. These methods are:

        * `fit`: Given an anndata dataset object, estimate the model parameters.
        * `predict`: Given experimental and biological features x in the form of
           a cell-by-features pd.DataFrame, return the parameters theta(x) of
           interest.
        * `sample`: Given the same experimental/biological information x as
          `predict`, simulate hypothetical profiles associated wtih those
          samples.

    Example instantiations of this class are given in the module
    `scdesigner.base.simulators`.

    Examples
    --------
    >>> from scdesigner.datasets import pancreas
    >>> sim = Simulator()
    >>> sim.parameters
    >>>
    >>> # this is how a subclass would run, once its fit, predict, and sample
    >>> # methods are implemented.
    >>> sim.fit(pancreas) # doctest: +SKIP
    >>> sim.predict(pancreas.obs) # doctest: +SKIP
    >>> sim.sample(pancreas.obs) # doctest: +SKIP
    """
    def __init__(self):
        self.parameters = None

    @abstractmethod
    def fit(self, anndata: AnnData, **kwargs) -> None:
        """Fit the simulator

        Parameters
        ----------
        adata : AnnData
            This is the object on which we want to estimate the simulator. This
            serves as the template for all downstream fitting.
        """
        self.template = anndata
        raise NotImplementedError

    @abstractmethod
    def predict(self, obs: DataFrame=None, **kwargs) -> Dict:
        """Predict from an obs dataframe"""
        raise NotImplementedError

    @abstractmethod
    def sample(self, obs: DataFrame=None, **kwargs) -> AnnData:
        """Generate samples."""
        raise NotImplementedError