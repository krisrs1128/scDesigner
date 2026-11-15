"""GPBasis length scales follow the coordinate scale unless set explicitly."""

import numpy as np

from scdesigner.data import GPBasis


def coordinates(scale=1.0, shift=0.0, n=60, seed=3):
    return np.random.default_rng(seed).uniform(0, 1, size=(n, 2)) * scale + shift


def fit(coords, **kwargs):
    np.random.seed(0)
    return GPBasis(df=4, n_landmarks=20, standardize=True, **kwargs).fit(coords)


def test_default_basis_does_not_depend_on_coordinate_units():
    unit = fit(coordinates())
    large = fit(coordinates(scale=1000.0, shift=500.0))
    np.testing.assert_allclose(
        large.transform(coordinates(1000.0, 500.0, n=10, seed=4)),
        unit.transform(coordinates(n=10, seed=4)),
        rtol=1e-6, atol=1e-8,
    )


def test_default_basis_reaches_held_out_cells_at_pixel_scale_coordinates():
    held_out = coordinates(scale=10000.0, n=15, seed=5)
    columns = fit(coordinates(scale=10000.0)).transform(held_out)
    assert (np.abs(columns).max(axis=1) > 1e-8).all()


def test_explicit_length_scale_is_used_as_given():
    held_out = coordinates(scale=10000.0, n=15, seed=5)
    columns = fit(coordinates(scale=10000.0), length_scale=1.0).transform(held_out)
    assert not columns.any()

