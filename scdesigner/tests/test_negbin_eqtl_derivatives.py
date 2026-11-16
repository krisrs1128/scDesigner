"""Check production likelihood derivatives against an independent autograd reference."""

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from scdesigner.distributions import NegBinEQTL


def _reference_derivatives(y, mean, dispersion):
    """Differentiate PyTorch's NB log probability with respect to log mean."""
    eta = mean.detach().double().log().requires_grad_()
    r = dispersion.detach().double()
    loglik = torch.distributions.NegativeBinomial(
        total_count=r, logits=eta - r.log()
    ).log_prob(y.double())
    terms = {}
    derivative = loglik
    for name in ("score", "observed", "third", "fourth"):
        derivative = torch.autograd.grad(derivative.sum(), eta, create_graph=True)[0]
        if name == "observed":
            derivative = -derivative  # Higher orders use the negative log likelihood.
        terms[name] = derivative.detach()
    return terms


@pytest.mark.parametrize("donor_idx", [
    pytest.param([0, 1, 2, 3, 4], id="per-cell"),
    pytest.param([0, 1, 0, 1, 1], id="summed-by-donor"),
])
def test_likelihood_derivatives_through_fourth_order(donor_idx):
    donor_idx = np.array(donor_idx)
    donors = [f"D{k}" for k in range(donor_idx.max() + 1)]
    y_array = np.array([[0, 3], [2, 1], [4, 0], [1, 5], [3, 2]], dtype=float)
    adata = AnnData(
        X=y_array,
        obs=pd.DataFrame(
            {"donor": [donors[k] for k in donor_idx]},
            index=[f"cell{i}" for i in range(len(donor_idx))],
        ),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    model = NegBinEQTL(
        "~ genotype + (1 | donor)",
        dosage=pd.DataFrame({"snp": np.arange(len(donors)) % 3}, index=donors),
        snp_map={"g0": ["snp"], "g1": ["snp"]},
        device="cpu",
        donor_blocked=False,
        sigma2_method="edgeworth",
    )
    model.setup_data(adata, batch_size=2)
    model.setup_optimizer()
    with torch.no_grad():
        model.predict.coefs["mean"].copy_(torch.tensor([[0.2, -0.3]]))
        model.predict.coefs["dispersion"].copy_(torch.log(torch.tensor([[1.5, 4.0]])))
        model.predict.U.copy_(torch.linspace(-0.4, 0.3, len(donors) * 2).reshape(-1, 2))
        model.predict.beta.fill_(0.2)

    # Accumulate over several batches, including noncontiguous cells from one donor.
    model.predict.zero_accumulators()
    predictions = []
    for y, x in model.loader:
        with torch.no_grad():
            predictions.append(model.predict(x))
            model.predict.loss_fn((y, x))

    reference = _reference_derivatives(
        torch.tensor(y_array),
        torch.cat([p["mean"] for p in predictions]),
        torch.cat([p["dispersion"] for p in predictions]),
    )
    for name, values in reference.items():
        # Use ordinary sums, independently of the production scatter operation.
        expected = torch.stack([
            values[donor_idx == k].sum(dim=0) for k in range(len(donors))
        ])
        torch.testing.assert_close(
            model.predict.accumulators[name].double(), expected, rtol=1e-5, atol=1e-6
        )
