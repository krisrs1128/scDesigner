class Marginal:
    def __init__(self, formula):
        super().__init__()
        self.formula = formula

    def fit(self, Y, X=None, **kwargs):
        pass

    def sample(self, Y, X=None, **kargs):
        pass

    def logliklihood(self, Y, X=None):
        pass

    def quantile(self, Y, q):
        pass
