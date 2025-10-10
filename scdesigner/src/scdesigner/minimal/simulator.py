from anndata import AnnData
from typing import Dict
from pandas import DataFrame
from abc import abstractmethod

class Simulator:
    """Simulation abstract class"""

    def __init__(self):
        self.parameters = None

    @abstractmethod
    def fit(self, **kwargs) -> None:
        """Fit the simulator"""
        pass

    @abstractmethod
    def predict(self, obs: DataFrame=None, **kwargs) -> Dict:
        """Predict from an obs dataframe"""
        pass

    @abstractmethod
    def sample(self, obs: DataFrame=None, **kwargs) -> AnnData:
        """Generate samples."""
        pass