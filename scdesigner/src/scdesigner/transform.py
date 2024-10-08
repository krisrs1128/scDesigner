from copy import deepcopy


def amplify(simulator, factor, alter_features, alter_genes):
    sim = deepcopy(simulator)
    transform_parameter(lambda x: factor * x, sim, alter_features, alter_genes)
    return sim


def dampen(simulator, factor, alter_features, alter_genes):
    sim = deepcopy(simulator)
    transform_parameter(lambda x: x / factor, sim, alter_features, alter_genes)
    return sim


def substring_match(feature, alter_features):
    for f in alter_features:
        if f in feature:
            return True
    return False


def transform_parameter(f, simulator, alter_features, alter_genes=None, parameter="mu"):
    for genes, margin in simulator.margins:
        if alter_genes is None:
            alter_genes = genes

        for theta in margin.module.linear[parameter].parameters():
            features = margin.configure_loader(simulator.anndata).dataset.features[parameter]
            gene_ix = [i for i, v in enumerate(genes) if v in alter_genes]

            feature_ix = [i for i, v in enumerate(features) if substring_match(v, alter_features)]
            theta.data[feature_ix, gene_ix] = f(theta[feature_ix, gene_ix])