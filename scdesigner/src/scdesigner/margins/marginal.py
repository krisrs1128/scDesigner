
class Marginal():
    def __init__(self, formula):
        super().__init__()
        self.formula = formula

    def fit(self, Y, X=None, **kwargs):
        pass
    
    def sample(self, Y, X=None, **kargs):
        pass

    def logliklihood(self, Y, X=None):
        pass

class MarginalList():
    def __init__(self, marginals):
        super().__init__()
        self.marginals = marginals


    def fit(self, Y, X=None, index=None, **kwargs):
        if index is None:
            index = range(self.marginals)
        for ix in index:
            self.margins[ix].fit(X, Y, **kwargs)

    
    def sample(self, Y, X=None, **kargs):
        pass

    def logliklihood(self, Y, X=None):
        pass