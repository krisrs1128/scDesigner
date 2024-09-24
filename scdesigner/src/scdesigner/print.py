import rich
import rich.table


def shorten_names(features, max_features=5):
    if len(features) > max_features:
        features = features[: (int(max_features - 1))] + ["..."] + features[-1:]
    return f"""[{', '.join(features)}]"""


def print_simulator(margins, copula):
    table = rich.table.Table(
        title="[bold magenta]Simulation Plan[/bold magenta]", title_justify="left"
    )
    table.add_column("formula")
    table.add_column("distribution")
    table.add_column("features")

    i = 1
    for m in margins:
        features, margin = m
        tup = tuple(margin.to_df().iloc[0, :]) + (shorten_names(features),)
        table.add_row(*tup)
        i += 1

    rich.print(table)
    if copula is None:
        rich.print("Marginal models without copula.")

    return ""
