from . import formula as fl
from anndata import AnnData
from formulaic import model_matrix
import numpy as np
import scipy
import torch
import torch.utils.data as td


def formula_group_loader(
    adata: AnnData, formula=None, grouping_variable=None, chunk_size=int(1e4), batch_size: int = None
):
    device = fl.check_device()
    if grouping_variable is None:
        adata.obs["_copula_group"] = "shared_group"
        grouping_variable = "_copula_group"

    if adata.isbacked:
        ds = FormulaGroupViewDataset(adata, formula, grouping_variable, chunk_size, device)
        dataloader = td.DataLoader(ds, batch_size=batch_size)
        ds.x_names = fl.model_matrix_names(adata, formula, ds.categories)
    else:
        # convert sparse to dense matrix
        y = adata.X
        if isinstance(y, scipy.sparse._csc.csc_matrix):
            y = y.todense()

        # wrap the entire data into a dataset
        x = model_matrix(formula, adata.obs)
        ds = td.StackDataset(
            td.TensorDataset(torch.tensor(np.array(x), dtype=torch.float32).to(device)),
            td.TensorDataset(torch.tensor(y, dtype=torch.float32).to(device)),
            ListDataset(adata.obs[grouping_variable])
        )
        ds.x_names = list(x.columns)
        dataloader = td.DataLoader(ds, batch_size=batch_size, collate_fn=stack_collate)

    return dataloader

class FormulaGroupViewDataset(td.Dataset):
    def __init__(self, view, formula=None, grouping_variable=None, chunk_size=int(1e4), device=None):
        super().__init__()
        self.device = device or fl.check_device()
        self.formula = formula
        self.categories = fl.column_levels(view.obs)
        self.grouping_variable = grouping_variable
        self.groups = list(self.categories[grouping_variable].dtype.categories)
        self.len = len(view)
        self.memberships = None
        self.view = view
        self.x = None
        self.y = None
        self.cur_range = range(0, min(self.len, chunk_size))

    def __len__(self):
        return self.len

    def __getitem__(self, ix):
        if self.x is None or ix not in self.cur_range:
            self.cur_range = range(ix, min(ix + len(self.cur_range), self.len))
            view_inmem = self.view[self.cur_range].to_memory()
            self.memberships = view_inmem.obs[self.grouping_variable]
            self.x = fl.safe_model_matrix(
                view_inmem.obs, self.formula, self.categories
            ).to(self.device)
            self.y = torch.from_numpy(view_inmem.X.toarray().astype(np.float32)).to(
                self.device
            )
        return self.x[ix - self.cur_range[0]], self.y[ix - self.cur_range[0]], self.memberships[ix - self.cur_range[0]]

class ListDataset(td.Dataset):
    """
    Simple DS to store groups
    """

    def __init__(self, list):
        self.list = list

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        return self.list[idx]


def stack_collate(batch):
    import pdb; pdb.set_trace()
    x = torch.stack([sample[0] for sample in batch])
    y = torch.stack([sample[1] for sample in batch])
    groups = tuple([sample[2] for sample in batch])
    return [x, y, groups]
