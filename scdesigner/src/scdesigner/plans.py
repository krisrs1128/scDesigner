import rich

def shorten_names(features, max_features=5):
    if len(features) > max_features:
        features = features[:(int(max_features - 1))] + ["..."] + features[-1:]
    return f"""[{', '.join(features)}]"""

class SimulationPlan():
    def __init__(self, margins, copula=None):
        super().__init__()
        self.margins = margins
        self.copula = copula

    def __repr__(self):
        table = rich.table.Table(
            title="[bold magenta]Simulation Plan[/bold magenta]", 
            title_justify="left"
        )
        table.add_column("formula")
        table.add_column("distribution")
        table.add_column("features")

        i = 1
        for m in self.margins:
            features, margin = m
            tup = tuple(margin.to_df().iloc[0, :]) + (shorten_names(features),)
            table.add_row(*tup)
            i += 1

        rich.print(table)
        return ""

    def __str__(self):
        self.__repr__()
        return ""


def setup_simulator(anndata, margins):
    if not isinstance(margins, list):
        margins = [(list(anndata.var_names), margins)]

    plan = SimulationPlan(margins)
    return plan