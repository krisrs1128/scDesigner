import anndata
import scipy.sparse
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


class MementoEstimator:
    def __init__(self, q=0.1):
        try:
            import memento.simulate
        except ImportError:
            memento_error()
        self.q = q

    def estimate(self, loader: SparseMatrixLoader, **kwargs) -> dict:
        import memento.simulate as ms

        X = []
        for x in loader:
            X.append(torch_to_scipy(x))
        X = scipy.sparse.vstack(X)

        # estimate memento model
        x_param, _, Nc, _ = ms.extract_parameters(X, q=self.q, **kwargs)

        # postprocess parameters
        mean = x_param[0] * Nc.mean()
        variance = (x_param[1] + x_param[0] ** 2) * (Nc**2).mean() - x_param[
            0
        ] ** 2 * Nc.mean() ** 2
        norm_cov = ms.make_spd_matrix(x_param[0].shape[0])
        return {"mean": mean, "variance": variance, "norm_cov": norm_cov, "Nc": Nc}


def torch_to_scipy(x):
    """
    Convert sparse CSR tensor to sparse CSR scipy array.
    """
    ix = x.to_sparse().indices()
    return scipy.sparse.csr_matrix((x.values(), ix), x.shape)


def memento_error():
    raise ImportError(
        """
        The MementoEstimator class requires memento. Please install
        it with:

            pip install memento-de

        You can learn more about this dependency at

            Code: https://github.com/yelabucsf/scrna-parameter-estimation/tree/master
            Paper: https://www.cell.com/cell/fulltext/S0092-8674%2824%2901144-9
        """
    )
