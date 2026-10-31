"""Numerical contract for the shared coefficient penalty."""

import numpy as np
import pytest
import torch

from scdesigner.distributions.penalties import CoefficientPenalty, QuadraticPenalty


def test_penalty_value_and_gradient_match_named_coefficients():
    coefs = torch.tensor([[7., 8.], [1., 2.], [9., 10.], [3., 4.]],
                         dtype=torch.float64, requires_grad=True)
    penalty = CoefficientPenalty("mean", ["b", "a"], [2., 3.], QuadraticPenalty(lam=0.5))
    bound = penalty.bind({"mean": ["Intercept", "a", "covariate", "b"]},
                         {"mean": coefs}, np.array([0., 2.]))
    assert bound.value().item() == pytest.approx(1061.)
    bound.value().backward()
    torch.testing.assert_close(coefs.grad, torch.tensor(
        [[0., 0.], [300., 3.], [0., 0.], [600., 4.]], dtype=torch.float64))


@pytest.mark.parametrize("lam, expected", [(0.5, 32.5), (0., 0.)])
def test_equal_weight_and_disabled_penalty(lam, expected):
    coefs = torch.tensor([[1., 2.], [3., 4.]])
    penalty = CoefficientPenalty("mean", ["b", "a"], [2., 3.],
                                 QuadraticPenalty(lam=lam, weighting="none"))
    bound = penalty.bind({"mean": ["a", "b"]}, {"mean": coefs})
    assert bound.value().item() == pytest.approx(expected)
