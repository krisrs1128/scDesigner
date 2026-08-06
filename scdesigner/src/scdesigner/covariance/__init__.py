"""Covariance estimators for copula dependence structures."""

from .base import CovarianceEstimator, as_covariance_estimator, register_estimator
from .moments import SampleCovariance
from .shrinkage import (
    LedoitWolf,
    OAS,
    ledoit_wolf_intensity,
    oas_intensity,
    shrink_covariance,
)

__all__ = [
    "as_covariance_estimator",
    "CovarianceEstimator",
    "LedoitWolf",
    "ledoit_wolf_intensity",
    "OAS",
    "oas_intensity",
    "register_estimator",
    "SampleCovariance",
    "shrink_covariance",
]
