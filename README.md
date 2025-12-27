## scDesigner

**scDesigner** is a Python Module for modeling and simulating single-cell data with various models built on top of PyTorch.

### Features

- **Flexible models**: Poisson / Negative Binomial / Zero-inflated variants (and more).
- **Scalable simulation**: GPU-accelerated training and simulation with PyTorch.
- **User-friendly interface**: implemented with `scikit-learn`-like API.
- **Extensible design**: easy to extend to new models and simulators.

### Installation guide

#### Option A: Install the Python package (recommended)

Create a clean environment (optional but recommended):

```bash
conda create -n scdesigner python=3.11 -y
conda activate scdesigner
```

Install:

```bash
pip install scdesigner==0.0.6
```

#### Option B: Install from source (this repository)

This option is recommended if you want to access the latest features and bug fixes.

```bash
git clone https://github.com/krisrs1128/scDesigner.git
cd scDesigner/scdesigner
pip install -e .
```

### Quickstart

After installation, you can import the package in Python:

```python
import scdesigner
```

You may refer to `examples/quickstart.ipynb` for basic usage of the package.

### Repository Structure

- **Examples**: `examples/`
- **Python package source**: `scdesigner/src/scdesigner/`
- **R wrappers**: `R/scDesigner/`

### Contributing

Issues and pull requests are welcome. Please include a minimal reproducible example and relevant environment information.
