import anndata
import muon
from typing import Union


class Simulator:
    def __init__(self, data: Union[anndata.AnnData, muon.MuData], **kwargs):
        self.data = data

    def fit(self, **kwargs) -> dict:
        raise NotImplementedError("'fit' is not yet implemented for this simulator.")

    def sample(self, **kwargs) -> Union[anndata.AnnData, muon.MuData]:
        raise NotImplementedError("'sample' is not yet implemented for this simulator.")

    def nullify(self, **kwargs) -> None:
        raise NotImplementedError(
            "'nullify' is not yet implemented for this simulator."
        )

    def amplify(self, **kwargs) -> None:
        raise NotImplementedError(
            "'amplify' is not yet implemented for this simulator."
        )
