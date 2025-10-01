from .scd3 import SCD3Simulator
from .negbin import NegBin
from .standard_covariance import StandardCovariance
from typing import Optional


class NegBinCopula(SCD3Simulator):
    def __init__(self,
                 mean_formula: Optional[str] = None,
                 dispersion_formula: Optional[str] = None,
                 copula_formula: Optional[str] = None) -> None:
        marginal = NegBin({"mean": mean_formula, "dispersion": dispersion_formula})
        covariance = StandardCovariance(copula_formula)
        super().__init__(marginal, covariance)