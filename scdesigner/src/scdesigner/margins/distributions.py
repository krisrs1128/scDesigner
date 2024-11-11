from scipy.stats import nbinom
from typing import Optional, Union
import torch.distributions

class NegativeBinomial(torch.distributions.NegativeBinomial):
    """
    Code taken from:
    https://ts.gluon.ai/stable/_modules/gluonts/torch/distributions/negative_binomial.html#NegativeBinomial.cdf

    Negative binomial distribution with `total_count` and `probs` or `logits`
    parameters.

    Based on torch.distributions.NegativeBinomial, with added `cdf` and `icdf`
    methods.
    """
    def __init__(
        self,
        total_count: Union[float, torch.Tensor],
        probs: Optional[Union[float, torch.Tensor]] = None,
        logits: Optional[Union[float, torch.Tensor]] = None,
        validate_args=None,
    ):
        super().__init__(
            total_count=total_count,
            probs=probs,
            logits=logits,
            validate_args=validate_args,
        )

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            if not torch.all(value == value.floor()):
                raise ValueError("Cannot use an NB model if entries are not all integers. Did you accidentally log transform the data?")
            self._validate_sample(value)

        result = self.scipy_nbinom.cdf(value.numpy())
        return torch.tensor(result, device=value.device, dtype=value.dtype)


    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        result = self.scipy_nbinom.ppf(value.numpy())
        return torch.tensor(result, device=value.device, dtype=value.dtype)

    @property
    def scipy_nbinom(self):
        return nbinom(
            n=self.total_count.numpy(),
            p=1.0 - self.probs.numpy(),
        )