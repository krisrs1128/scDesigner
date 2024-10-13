from .margins.marginal import args
from .copula import ScCopula
from .margins.reformulate import reformulate, match_marginal, nullify_formula
from . import join as scj
from .transform import amplify, dampen
from collections import defaultdict
import torch
from torch.optim import LBFGS
from torch.utils.data import DataLoader
import anndata as ad
import pandas as pd
import numpy as np

def retrieve_obs(N, obs, anndata):
    if obs is not None:
        return obs
    elif N is not None:
        ix = torch.randint(high=len(anndata), size=(N,))
        return anndata.obs.iloc[ix.tolist(), :]
    else:
        return anndata.obs

def merge_predictions(param_hat):
    merged = defaultdict(list)
    for genes, d in param_hat:
        for k, v in d.items():
            df = pd.DataFrame(v.numpy(), columns=genes)
            merged[k].append(df)

    return {k: pd.concat(v, axis=1) for k, v in merged.items()}

class Simulator:
    def __init__(self, margins, multivariate=None):
        super().__init__()
        self.margins = margins
        self.multivariate = multivariate
        self.anndata = None

    def fit(self, anndata, max_epochs=10):
        for margin in self.margins:
            y_names, submodel = margin
            if anndata.isbacked:
                train_subset = anndata[:, y_names]
            else:
                ix = np.sum(np.isnan(anndata[:, y_names].X), axis=1) == 0
                train_subset = anndata[ix, y_names]
            submodel.fit(train_subset, max_epochs)

        if self.multivariate is not None:
            self.multivariate.fit(self.margins, anndata)

        self.anndata = anndata

    def reformulate(self, genes, formula, max_epochs=10):
        def f(margin, genes):
            return reformulate(margin, genes, formula, self.anndata)

        self.margins = margin_apply(
            self.margins, genes, f, self.anndata, max_epochs=max_epochs
        )

    def predict(self, obs):
        param_hat = []
        for genes, margin in self.margins:
            param_hat.append([genes, margin.predict(obs)])
        return merge_predictions(param_hat)

    def sample(self, N=None, obs=None, split=False):
        new_obs = retrieve_obs(N, obs, self.anndata)
        if self.multivariate is None:
            var_names, counts = sample_marginals(self.margins, new_obs)
        else:
            var_names, counts = self.multivariate.sample(self.margins, new_obs)

        adata = ad.AnnData(
            np.concatenate(counts, axis=1), 
            new_obs, 
            var=self.anndata.var
        )

        adata.var_names = var_names
        if split:
            adata = scj.split_adata(adata)

        return adata

    def nullify(self, term, genes, max_epochs=10):
        def f(margin, genes):
            null_formula = nullify_formula(margin.formula, term)
            return reformulate(margin, genes, null_formula, self.anndata)

        self.margins = margin_apply(
            self.margins, genes, f, self.anndata, max_epochs=max_epochs
        )

    def amplify(self, factor, alter_features, alter_genes=None):
        self.margins = amplify(self, factor, alter_features, alter_genes).margins

    def dampen(self, factor, alter_features, alter_genes=None):
        self.margins = dampen(self, factor, alter_features, alter_genes).margins

    def join(self, sim2, mode="copula", **kwargs):
        if mode == "copula":
            return scj.join_copula(self, sim2, **kwargs)
        elif mode == "pamona":
            return scj.join_pamona(self, sim2, **kwargs)
        else:
            raise NotImplementedError(f"No join mode {mode}. Did you provide one of the supported modes: 'copula' or 'pamona'?")


def scdesigner(anndata, margins, delay=False, multivariate=ScCopula(), max_epochs=10, 
               chunk_size=1e4, **kwargs):
    if not isinstance(margins, list):
        margins = [(list(anndata.var_names), margins)]
    margins = fragment_margins(margins, chunk_size)

    for _, margin in margins:
        margin.loader_opts = safe_update(margin.loader_opts, args(DataLoader, **kwargs))
        margin.optimizer_opts = safe_update(
            margin.optimizer_opts, args(LBFGS, **kwargs)
        )

    simulator = Simulator(margins, multivariate)
    if not delay:
        simulator.fit(anndata, max_epochs)

    return simulator

###############################################################################
## Utilities
###############################################################################

def fragment_margins(margins, chunk_size=None):
    """
    Divide a margin with > chunk_size genes into many margins with only chunk_size genes.
    """
    if chunk_size is None:
        return margins

    new_margins = []
    for genes, margin in margins:
        new_margins += [
            (genes[i:(i + chunk_size)], margin)
            for i in range(0, len(genes), chunk_size)
        ]
    return new_margins


def safe_update(d1, d2):
    d1.update({k: v for k, v in d2.items() if k not in d1})
    return d1

def margin_apply(margins, genes, f, anndata, **kwargs):
    ix, matched = match_marginal(margins, genes)
    for i in sorted(ix, reverse=True):
        del margins[i]

    for _, margin in matched:
        new, unchanged = f(margin, genes)
        g1, g2 = (m.module.gene_names for m in (new, unchanged))
        new.fit(anndata[:, g1], **kwargs)
        margins += [(g1, new), (g2, unchanged)]
    return margins
        
def sample_marginals(margins, obs):
    var_names, counts = [], []
    for genes, margin in margins:
        var_names += list(genes)
        counts.append(margin.sample(obs).numpy())
    return var_names, counts
