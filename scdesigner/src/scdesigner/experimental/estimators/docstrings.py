def doc(text):
    def decorator(func):
        func.__doc__ = text
        return func
    return decorator

negative_binomial_copula = """
A minimal NB copula model

# simulate data
n_samples, n_features, n_outcomes = 1000, 2, 4
x_sim = np.random.normal(size=(n_samples, n_features))
beta_sim = np.random.normal(size=(n_features, n_outcomes))
mu_sim = np.exp(x_sim @ beta_sim)
r_sim = np.random.uniform(.5, 1.5, n_outcomes)
y_sim = np.random.negative_binomial(r_sim, r_sim / (r_sim + mu_sim))
y_sim[:, 1] = y_sim[:, 0]

# estimate model
negative_binomial_copula(x_sim, y_sim)
"""

negative_binomial_regression =  """
A minimal NB regression model

# simulate data
n_samples, n_features, n_outcomes = 1000, 2, 4
x_sim = np.random.normal(size=(n_samples, n_features))
beta_sim = np.random.normal(size=(n_features, n_outcomes))
mu_sim = np.exp(x_sim @ beta_sim)
r_sim = np.random.uniform(.5, 1.5, n_outcomes)
y_sim = np.random.negative_binomial(r_sim, r_sim / (r_sim + mu_sim))

# estimate model
negative_binomial_regression(x_sim, y_sim)
"""