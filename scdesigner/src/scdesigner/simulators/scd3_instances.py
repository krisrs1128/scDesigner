# migrate the whole content to scd3.py

# from .scd3 import SCD3Simulator
# from .negbin import NegBin
# from .zero_inflated_negbin import ZeroInflatedNegBin
# from .gaussian import Gaussian
# from .standard_copula import StandardCopula
# from typing import Optional


# class NegBinCopula(SCD3Simulator):
#     def __init__(self,
#                  mean_formula: Optional[str] = None,
#                  dispersion_formula: Optional[str] = None,
#                  copula_formula: Optional[str] = None) -> None:
#         marginal = NegBin({"mean": mean_formula, "dispersion": dispersion_formula})
#         covariance = StandardCopula(copula_formula)
#         super().__init__(marginal, covariance)


# class ZeroInflatedNegBinCopula(SCD3Simulator):
#     def __init__(self,
#                  mean_formula: Optional[str] = None,
#                  dispersion_formula: Optional[str] = None,
#                  zero_inflation_formula: Optional[str] = None,
#                  copula_formula: Optional[str] = None) -> None:
#         marginal = ZeroInflatedNegBin({
#             "mean": mean_formula,
#             "dispersion": dispersion_formula,
#             "zero_inflation_formula": zero_inflation_formula
#         })
#         covariance = StandardCopula(copula_formula)
#         super().__init__(marginal, covariance)


# class BernoulliCopula(SCD3Simulator):
#     def __init__(self,
#                  mean_formula: Optional[str] = None,
#                  copula_formula: Optional[str] = None) -> None:
#         marginal = NegBin({"mean": mean_formula})
#         covariance = StandardCopula(copula_formula)
#         super().__init__(marginal, covariance)


# class GaussianCopula(SCD3Simulator):
#     def __init__(self,
#                  mean_formula: Optional[str] = None,
#                  sdev_formula: Optional[str] = None,
#                  copula_formula: Optional[str] = None) -> None:
#         marginal = Gaussian({"mean": mean_formula, "sdev": sdev_formula})
#         covariance = StandardCopula(copula_formula)
#         super().__init__(marginal, covariance)