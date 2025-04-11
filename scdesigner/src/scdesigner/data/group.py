from ..data.formula import check_device
from anndata import AnnData
import numpy as np
import scipy
import torch
import torch.utils.data as td


def group_loader(
    adata: AnnData, grouping_variable=None, chunk_size=int(1e4), batch_size: int = None
):
    device = check_device()
    if adata.isbacked:
        ds = GroupViewDataset(adata, grouping_variable, chunk_size, device)
        dataloader = td.DataLoader(ds, batch_size=batch_size)
    else:
        # convert sparse to dense matrix
        y = adata.X
        if isinstance(y, scipy.sparse._csc.csc_matrix):
            y = y.todense()

        # wrap the entire data into a dataset
        ds = td.StackDataset(
            ListDataset(adata.obs[grouping_variable]),
            td.TensorDataset(torch.tensor(y, dtype=torch.float32).to(device)),
        )
        dataloader = td.DataLoader(ds, batch_size=batch_size, collate_fn=stack_collate)

    return dataloader


class GroupViewDataset(td.Dataset):
    def __init__(self, view, grouping_variable=None, chunk_size=int(1e4), device=None):
        super().__init__()
        self.device = device or check_device()
        self.grouping_variable = grouping_variable
        self.groups = unique_groups(view.obs, grouping_variable)
        self.len = len(view)
        self.memberships = None
        self.view = view
        self.y = None
        self.cur_range = range(0, min(self.len, chunk_size))

    def __len__(self):
        return self.len

    def __getitem__(self, ix):
        if self.memberships is None or ix not in self.cur_range:
            self.cur_range = range(ix, min(ix + len(self.cur_range), self.len))
            view_inmem = self.view[self.cur_range].to_memory()
            self.memberships = view_inmem.obs[self.grouping_variable]
            self.y = torch.from_numpy(view_inmem.X.toarray().astype(np.float32)).to(
                self.device
            )
        return self.memberships[ix - self.cur_range[0]], self.y[ix - self.cur_range[0]]


def unique_groups(obs, k):
    if obs[k].dtype.kind in {"O", "U"}:
        obs[k] = obs[k].astype("category")
    return list(obs[k].dtype.categories)


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
    groups = tuple([sample[0] for sample in batch])
    x = torch.stack([sample[1][0] for sample in batch])
    return [groups, x]
