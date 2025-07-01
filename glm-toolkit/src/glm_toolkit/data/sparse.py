import anndata
import torch
import torch.utils.data as td

class SparseMatrixLoader:
    def __init__(self, adata: anndata.AnnData, batch_size: int = None):
        ds = SparseMatrixDataset(adata, batch_size)
        self.loader = td.DataLoader(ds, batch_size=None)


class SparseMatrixDataset(td.IterableDataset):
    def __init__(self, anndata: anndata.AnnData, batch_size: int = None):
        self.n_rows = anndata.X.shape[0]
        if batch_size is None:
            batch_size = self.n_rows

        self.sparse_matrix = anndata.X
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, self.n_rows, self.batch_size):
            batch_indices = range(i, min(i + self.batch_size, self.n_rows))
            batch_rows = self.sparse_matrix[batch_indices, :]

            # Convert to sparse CSR tensor
            batch_indices_rows, batch_indices_cols = batch_rows.nonzero()
            batch_values = batch_rows.data

            batch_sparse_tensor = torch.sparse_coo_tensor(
                torch.tensor([batch_indices_rows, batch_indices_cols]),
                torch.tensor(batch_values, dtype=torch.float32),
                (len(batch_indices), self.sparse_matrix.shape[1]),
            ).to_sparse_csr()

            yield batch_sparse_tensor

    def __len__(self):
        return (self.n_rows + self.batch_size - 1) // self.batch_size

