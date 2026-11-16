"""Formula validation and donor/gene/SNP alignment for the eQTL marginal."""

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from scdesigner.distributions import NegBinEQTL


@pytest.fixture
def model_inputs():
    # Deliberately different donor, gene, and SNP orders in each input.
    dosage = pd.DataFrame(
        [[0, 1, 2], [2, 0, 1], [1, 2, 0], [2, 1, 1]],
        index=["D3", "D1", "D4", "D2"],
        columns=["snp3", "snp1", "snp2"],
    )
    return {
        "dosage": dosage,
        "snp_map": {"geneA": ["snp1", "snp2"], "geneB": ["snp3"]},
        "device": "cpu",
    }


@pytest.fixture
def adata():
    return AnnData(
        X=np.ones((5, 3)),
        obs=pd.DataFrame(
            {"indiv": ["D2", "D1", "D2", "D4", "D3"]},
            index=[f"cell{i}" for i in range(5)],
        ),
        var=pd.DataFrame(index=["geneC", "geneA", "geneB"]),
    )


@pytest.mark.parametrize("mean, dispersion", [
    pytest.param("~ cell_type + (1 | indiv)", "~ 1", id="missing-genotype"),
    pytest.param("~ genotype", "~ 1", id="missing-donor-intercept"),
    pytest.param(
        "~ genotype + (1 | indiv) + (1 | batch)", "~ 1", id="multiple-random-effects"
    ),
    pytest.param("~ genotype + (cell_type | indiv)", "~ 1", id="random-slope"),
    pytest.param(
        "~ genotype:cell_type + (1 | indiv)", "~ 1", id="interaction-without-main-effect"
    ),
    pytest.param(
        "~ genotype + bs(genotype, df=3) + (1 | indiv)", "~ 1", id="transformed-genotype"
    ),
    pytest.param("~ genotype + (1 | indiv)", "~ genotype", id="genotype-dispersion"),
    pytest.param("~ genotype + (1 | indiv)", "~ 1 + (1 | indiv)", id="random-dispersion"),
])
def test_invalid_formula_is_rejected(model_inputs, mean, dispersion):
    with pytest.raises(ValueError):
        NegBinEQTL({"mean": mean, "dispersion": dispersion}, **model_inputs)


def test_dosages_follow_donor_gene_and_snp_names(model_inputs, adata):
    model = NegBinEQTL("~ genotype + (1 | indiv)", **model_inputs)
    model.setup_data(adata, batch_size=3)

    # geneC has no SNPs, geneA has two, and geneB has one.
    expected = torch.tensor([
        [[0., 0.], [1., 2.], [0., 0.]],  # D3
        [[0., 0.], [0., 1.], [2., 0.]],  # D1
        [[0., 0.], [2., 0.], [1., 0.]],  # D4
        [[0., 0.], [1., 1.], [2., 0.]],  # D2
    ])
    assert model.donor_order == ["D3", "D1", "D4", "D2"]
    torch.testing.assert_close(model.D, expected)
    torch.testing.assert_close(
        model.mask, torch.tensor([[0., 0.], [1., 1.], [1., 0.]])
    )


def test_setup_rejects_unknown_donor(model_inputs, adata):
    model = NegBinEQTL("~ genotype + (1 | indiv)", **model_inputs)
    adata.obs.loc["cell0", "indiv"] = "UNKNOWN"
    with pytest.raises(ValueError):
        model.setup_data(adata, batch_size=3)
