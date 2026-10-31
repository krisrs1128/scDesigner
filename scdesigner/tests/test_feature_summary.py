"""Feature summaries agree with direct numerical reductions."""

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData

from scdesigner.data.feature_summary import summarize_features


@pytest.mark.parametrize("as_matrix", [np.array, sp.csr_matrix], ids=["dense", "csr"])
def test_feature_summary_matches_reference(as_matrix):
    values = np.array([[0., 1.], [0., 3.], [0., 2.], [0., 2.]])
    summary = summarize_features(AnnData(X=as_matrix(values)), include_max=True)
    np.testing.assert_array_equal(summary.mean, [0., 2.])
    np.testing.assert_array_equal(summary.maximum, [0., 3.])
