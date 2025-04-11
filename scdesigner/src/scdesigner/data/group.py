from ..data.formula import check_device
import torch
import numpy as np
import torch.utils.data as td

def unique_groups(obs, k):
    if obs[k].dtype.kind in {'O', 'U'}:
        obs[k] = obs[k].astype("category")
    return list(obs[k].dtype.categories)

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
