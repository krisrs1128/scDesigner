# GLM Toolkit

A Python package for Generalized Linear Models (GLM) providing various estimators, predictors, and samplers. This package is extracted from the scDesigner project and provides core GLM functionality.

## Features

- **Estimators**: Various GLM estimators including Negative Binomial, Poisson, Bernoulli, and Zero-Inflated models
- **Predictors**: Prediction functions for all supported models
- **Samplers**: Sampling functions for all supported models


## Usage

```python
from glm_toolkit.estimators import negbin_regression
from glm_toolkit.predictors import negbin_predict
from glm_toolkit.samplers import negbin_sample

# Let adata be an AnnData object with features in adata.obs and response in adata.X
formula = "feature_0 + feature_1 + feature_2 - 1"

# Fit model
params = negbin_regression(adata, formula)
# Predict parameters for each sample
local_params = negbin_predict(params, adata.obs, formula)
# Sample from the model
samples = negbin_sample(local_params, adata.obs)
```
