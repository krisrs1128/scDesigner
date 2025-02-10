def doc(text):
    def decorator(func):
        func.__doc__ = text
        return func
    return decorator

nullify = """
# fit a model
example_sce = anndata.read_h5ad(save_path)
x = formulaic.model_matrix("~ bs(pseudotime, degree=2)", example_sce.obs)
x = x.to_numpy()
y = example_sce.X.todense()
params = negative_binomial_copula(x, y)

# nullify pseudotime for first ten genes
outcomes = example_sce.var_names[:10]
mask = anndata_formula_mask(outcomes, ["pseudotime], "~ bs(pseudotime, degree=2)", example_sce)
null_params = nullify(params, "beta", mask)
"""