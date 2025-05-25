import scipy.sparse
from .. import data

class MementoEstimator:
    def __init__(self, q=0.1):
        try:
            import memento.simulate
        except ImportError:
            memento_error()
        self.q = q

    def estimate(self, loader: data.SparseMatrixLoader, **kwargs) -> dict:
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
