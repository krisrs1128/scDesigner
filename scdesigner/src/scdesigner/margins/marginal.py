import anndata as ad
import lightning as pl
from torch.utils.data import DataLoader
from .regressors import NBRegression
from ..formula import FormulaDataset

class MarginalModel:
    def __init__(self, formula, module, loader_opts={}):
        super().__init__()
        self.formula = formula
        self.module = module
        self.loader_opts = loader_opts

    def configure_loader(self, anndata):
        if self.loader_opts.get("batch_size") is None:
            self.loader_opts["batch_size"] = len(anndata)
        dataset = FormulaDataset(self.formula, anndata)
        return DataLoader(dataset, **self.loader_opts)

    def configure_module(self, anndata):
        ds = self.configure_loader(anndata)
        _, obs = next(iter(ds))
        n_input = {k: v.shape[-1] for k, v in obs.items()}
        self.module = self.module(n_input, anndata.var_names)

    def fit(self, anndata, max_epochs=5, **kwargs):
        self.configure_module(anndata)
        ds = self.configure_loader(anndata)
        pl.Trainer(max_epochs=max_epochs, **kwargs).fit(self.module, train_dataloaders=ds)

    def predict(self, obs):
        ds = self.configure_loader(ad.AnnData(obs=obs))
        preds = []
        for _, obs_ in ds:
            preds.append(self.module(obs_))

        return preds

    def parameters(self):
        pass

class NB(MarginalModel):
    def __init__(self, formula, loader_opts={}):
        super().__init__(formula, NBRegression, loader_opts)