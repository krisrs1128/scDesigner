"""Benchmark harness for the `NegBinEQTL` donor-variance estimators.

`sigma2_method` selects how the EM step approximates the posterior second
moment `E[u_kj^2 | y]`:

    "laplace"    first-order Laplace marginal ML   (C in the write-up, default)
    "edgeworth"  second-order Edgeworth moments    (C2)

Every number on record for these comes from one gene (`ENSG00000237541`) in one
40-gene fit, where only 2 genes carry an empty donor. This module runs the same
comparison across four exported configurations, up to 977 genes where 492 do.

The notebooks under `examples/vignettes-eqtl-*.ipynb` call into here so the
harness lives in one place. Data comes from `validation/eqtl_glmmtmb/export_data.R`;
see :func:`load_config` for the command.

Two caveats that constrain every result produced here:

* **Genotypes are permuted** in all of these datasets, documented upstream.
  There is no true `beta` to recover. Accuracy can only mean `sigma^2` against
  the exact-quadrature referee, or null calibration `beta_hat ~= 0`.
* **`beta` and `zeta` are individually unidentified** under
  `mean = "~ C(cell_type)"` with `interaction = "~ 0 + C(cell_type)"`; the
  one-hot makes `sum_l V_il = 1`, so only `beta_j + zeta_jl` is pinned down.
  Read slopes through :func:`slopes`, never off `beta` alone.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
VALIDATION_DIR = REPO / "validation" / "eqtl_glmmtmb"
DATA_ROOT = HERE / "data" / "eqtl_bench"
RESULTS_DIR = DATA_ROOT / "results"

sys.path.insert(0, str(VALIDATION_DIR))

from common import cell_type_slopes, working_summary  # noqa: E402
from scdesigner.distributions import NegBinEQTL  # noqa: E402

METHODS = ("laplace", "edgeworth")

#: Exported configurations, in the order they are meant to be read.
CONFIGS = {
    "quickstart129": "moderate expression window, 129 genes",
    "quickstart977": "every non-corrupt gene, 977 genes",
    "trans": "trans-eQTL table, 27 genes over 5 SNPs",
    "selected": "the two genes from the power vignette",
}

#: Held fixed across every run so timings compare like with like.
MEAN_FORMULA = "~ C(cell_type)"
DISPERSION_FORMULA = "~ 1"
INTERACTION_FORMULA = "~ 0 + C(cell_type)"
DONOR_COL = "indiv"


def load_config(name: str, root: Path = DATA_ROOT):
    """Return `(adata, dosage, snp_map, gene_stats)` for one exported config.

    Raises if the export is missing. It is not regenerated here and there is no
    fallback to a different config: the R export is the only thing that defines
    which genes a run covers, so a silent substitution would change the
    experiment without changing the label on it.
    """
    if name not in CONFIGS:
        raise ValueError(f"unknown config {name!r}; expected one of {sorted(CONFIGS)}")

    directory = Path(root) / name
    required = ["counts.csv", "obs.csv", "dosage.csv", "snp_map.csv", "gene_stats.csv"]
    missing = [f for f in required if not (directory / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"config {name!r} is missing {missing} in {directory}.\n"
            f"Generate it with:\n"
            f"    cd {VALIDATION_DIR}\n"
            f"    Rscript export_data.R {directory} {name}"
        )

    from anndata import AnnData

    counts = pd.read_csv(directory / "counts.csv")
    obs = pd.read_csv(directory / "obs.csv")
    dosage = pd.read_csv(directory / "dosage.csv", index_col=0)
    pairs = pd.read_csv(directory / "snp_map.csv")
    gene_stats = pd.read_csv(directory / "gene_stats.csv")

    adata = AnnData(X=counts.values.astype(np.float32), obs=obs.reset_index(drop=True))
    adata.var_names = list(counts.columns)
    snp_map = {row.gene: [row.snp] for row in pairs.itertuples()}
    return adata, dosage, snp_map, gene_stats


@dataclass
class Run:
    """One fit: its configuration, its timings, and the model that produced them."""

    config: str
    method: str
    seed: int
    device: str
    epochs: int
    fit_seconds: float
    setup_seconds: float
    model: object = field(repr=False)

    @property
    def seconds_per_epoch(self) -> float:
        return self.fit_seconds / self.epochs


def fit_once(
    config: str,
    method: str,
    *,
    seed: int = 0,
    max_epochs: int = 400,
    device: str = "cpu",
    batch_size: int = 1024,
    data=None,
) -> Run:
    """Fit one (config, method, seed) and time it.

    Early stopping is disabled (`val_frac=0.0`, `min_epochs=max_epochs`) so every
    run does the same number of passes. Otherwise a method that converged sooner
    would look faster per fit while being slower per epoch, and the two
    questions -- cost per unit work, and work required -- would be entangled.

    `data` accepts an already-loaded `(adata, dosage, snp_map, gene_stats)` tuple
    so a sweep over methods and seeds does not re-read the CSVs each time.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")

    adata, dosage, snp_map, _ = data if data is not None else load_config(config)

    torch.manual_seed(seed)
    model = NegBinEQTL(
        {
            "mean": MEAN_FORMULA,
            "dispersion": DISPERSION_FORMULA,
            "interaction": INTERACTION_FORMULA,
        },
        donor_col=DONOR_COL,
        dosage=dosage,
        snp_map=snp_map,
        sigma2_method=method,
        device=device,
    )

    start = time.perf_counter()
    model.setup_data(adata, batch_size=batch_size)
    setup_seconds = time.perf_counter() - start

    start = time.perf_counter()
    model.fit(max_epochs=max_epochs, verbose=False, val_frac=0.0, min_epochs=max_epochs)
    fit_seconds = time.perf_counter() - start

    return Run(
        config=config,
        method=method,
        seed=seed,
        device=device,
        epochs=max_epochs,
        fit_seconds=fit_seconds,
        setup_seconds=setup_seconds,
        model=model,
    )


def slopes(model) -> pd.DataFrame:
    """Per-cell-type genotype slopes `beta_j + zeta_jl`, the identified combination."""
    return cell_type_slopes(model, None)


def dispersion(model) -> np.ndarray:
    """Per-gene NB size `r_j`. The dispersion formula is `~ 1` throughout."""
    return np.exp(model.predict.coefs["dispersion"].detach().cpu().numpy()[0])


def summarize(run: Run, gene_stats: pd.DataFrame) -> dict:
    """Collapse one run to a single tidy row of robustness and cost metrics."""
    model = run.model
    sigma2 = model.sigma2.cpu().numpy()
    gap = model.mode_gap.cpu().numpy()
    slope = slopes(model)
    low, high = model._sigma2_bounds

    genes = list(model.adata.var_names)
    stats = gene_stats.set_index("gene").loc[genes]
    sparse = (stats["n_empty_donors"] > 0).to_numpy()

    return {
        "config": run.config,
        "method": run.method,
        "seed": run.seed,
        "device": run.device,
        "n_genes": len(genes),
        "n_cells": int(model.adata.n_obs),
        "n_donors": int(model.predict.U.shape[0]),
        "n_sparse_genes": int(sparse.sum()),
        # cost
        "fit_seconds": run.fit_seconds,
        "setup_seconds": run.setup_seconds,
        "seconds_per_epoch": run.seconds_per_epoch,
        "epochs": run.epochs,
        # robustness
        "invalid_posterior_variances": model.invalid_posterior_variance_count,
        # Relative, not exact. `ENSG00000205021` converges to 9.9999 under
        # laplace and 10.0000 under edgeworth; an equality test counts that as a
        # robustness difference between the methods when both have simply run
        # into the bound.
        "n_at_sigma2_cap": int((sigma2 >= high * (1 - 1e-4)).sum()),
        "n_at_sigma2_floor": int((sigma2 <= low * (1 + 1e-4)).sum()),
        "n_nonfinite_sigma2": int((~np.isfinite(sigma2)).sum()),
        "n_nonfinite_slopes": int((~np.isfinite(slope.to_numpy())).sum()),
        "mode_gap_median": float(np.median(gap)),
        "mode_gap_max": float(gap.max()),
        # the precondition: both methods expand about the conditional mode, so a
        # gap that is large relative to sigma invalidates the run rather than
        # describing it
        "mode_gap_over_sigma_max": float((gap / np.sqrt(sigma2)).max()),
        # location
        "sigma2_median": float(np.median(sigma2)),
        "sigma2_median_sparse": float(np.median(sigma2[sparse])) if sparse.any() else np.nan,
        "sigma2_median_dense": float(np.median(sigma2[~sparse])) if (~sparse).any() else np.nan,
        # null calibration: genotypes are permuted, so the slopes should be ~0
        "slope_abs_median": float(np.median(np.abs(slope.to_numpy()))),
        "slope_abs_max": float(np.abs(slope.to_numpy()).max()),
    }


def sweep(
    config: str,
    *,
    seeds=(0, 1),
    methods=METHODS,
    max_epochs: int = 400,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Run every (method, seed) for one config; return the summary and the models.

    Two seeds by default. `docs/eqtl-findings.md` records the fit as
    non-deterministic (max |dU| 1.2e-02 between two fits in the same process),
    so a single-seed difference between methods cannot be told apart from that
    noise.
    """
    data = load_config(config)
    gene_stats = data[3]

    rows, models = [], {}
    for method in methods:
        for seed in seeds:
            if verbose:
                print(f"  fitting {config} / {method} / seed {seed} ...", flush=True)
            run = fit_once(
                config, method, seed=seed, max_epochs=max_epochs,
                device=device, data=data,
            )
            rows.append(summarize(run, gene_stats))
            models[(method, seed)] = run
            if verbose:
                print(
                    f"    {run.fit_seconds:.1f}s, "
                    f"sigma2 median {rows[-1]['sigma2_median']:.4f}, "
                    f"invalid {rows[-1]['invalid_posterior_variances']}",
                    flush=True,
                )
    return pd.DataFrame(rows), models


def save_results(frame: pd.DataFrame, config: str) -> Path:
    """Cache one config's summary so the comparison notebook need not refit."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{config}.csv"
    frame.to_csv(path, index=False)
    return path


def load_results(config: str | None = None) -> pd.DataFrame:
    """Read back cached summaries; all configs when `config` is None."""
    names = list(CONFIGS) if config is None else [config]
    frames = []
    for name in names:
        path = RESULTS_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"no cached results for {name!r} at {path}. "
                f"Run the {name} notebook first."
            )
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# The exact-quadrature referee
# --------------------------------------------------------------------------

def referee_genes(
    gene_stats: pd.DataFrame,
    genes,
    n_sparse: int = 40,
    n_control: int = 20,
    seed: int = 0,
):
    """Stratified gene subsample for the referee: sparse genes, plus controls.

    The referee costs roughly ten forward passes and materialises a `(K, J, Q)`
    node tensor with `Q = 3 * nodes_per_block`. At the default 2400 that is
    7,200 nodes per donor-gene pair, so all 492 sparse genes of `quickstart977`
    would need about 1.2 GB in float64 -- hence the cap on both strata, not just
    the control one.

    The stratification is not a convenience: genes with an empty donor are the
    entire reason the corrections exist, and genes without one are the control
    that says whether a difference is specific to that regime. Sparse genes are
    taken in decreasing order of empty-donor count so the cap keeps the hardest
    cases rather than a random slice of them.
    """
    stats = gene_stats.set_index("gene").loc[list(genes)]
    sparse_stats = stats[stats["n_empty_donors"] > 0].sort_values(
        "n_empty_donors", ascending=False
    )
    sparse = sparse_stats.index[:n_sparse].tolist()
    dense = stats.index[stats["n_empty_donors"] == 0].tolist()

    rng = np.random.default_rng(seed)
    if len(dense) > n_control:
        dense = list(rng.choice(dense, size=n_control, replace=False))
    return sparse, dense


def referee_sigma2(
    model,
    gene_subset,
    nodes_per_block: int = 2400,
    rtol: float = 1e-6,
    max_updates: int = 2000,
) -> pd.DataFrame:
    """Exact-quadrature `sigma^2` per gene, at the fitted predictor.

    Integrates `u` out numerically instead of expanding about its mode, so it
    referees both methods rather than being a third one. The fit is held fixed:
    only `sigma^2` moves, which is what makes the comparison attributable.

    Run `pytest validation/eqtl_glmmtmb/test_exact_ebnm.py -q -s` before trusting
    the output; `docs/eqtl-findings.md` states that as a precondition.

    **Why this does not call `sparse_donor_correction.exact_referee`.** That
    function stops on the *total log-marginal* and raises if the whole problem
    has not converged. Both choices are right for what it was written for -- one
    40-gene fit whose published numbers it reproduces -- and wrong here:

    * On low-expression datasets some genes carry almost no between-donor signal,
      so `sigma^2` heads for the floor and the EM map's slope approaches 1 at the
      boundary. Measured on `trans`: at update 300 the log-marginal was moving by
      3e-11 relative while `sigma^2` was still moving by 1.2e-3. A criterion on
      the log-marginal, at magnitude 2e6, cannot see that.
    * One such gene made the whole call raise, taking 26 perfectly converged
      genes down with it.

    So this loop uses the same primitives, stops on the parameter it actually
    reports, and returns a **per-gene** `converged` flag. Genes that did not
    converge are returned and flagged rather than dropped, because the count is
    itself a finding about the dataset. Callers should not average over them
    without saying so.
    """
    from ebnm import make_grid, normalize_scale
    from exact_ebnm import build_nodes, donor_loglik, exact_estep, exact_update
    from exact_gate0 import extract_fit

    genes = list(model.adata.var_names)
    index = [genes.index(g) for g in gene_subset]
    if not index:
        return pd.DataFrame(columns=["exact", "converged", "updates"])

    y, eta0, r, donor_idx = extract_fit(model)
    y, eta0, r, donor_idx = y.cpu(), eta0.cpu(), r.cpu(), donor_idx.cpu()
    U = model.predict.U.detach().double().cpu()
    sigma2_init = model.sigma2.detach().double().cpu()
    H, S = working_summary(model)
    H, S = H.double().cpu(), S.double().cpu()
    z = U + S / H.clamp(min=1e-12)

    columns = torch.tensor(index, dtype=torch.long)
    grid = make_grid(1, (0.25, 4.0), dtype=torch.float64)
    pi = torch.ones(1, dtype=torch.float64)
    sigma2, grid = normalize_scale(pi, sigma2_init[columns].clone(), grid)

    nodes, weights = build_nodes(
        z[:, columns], H[:, columns], sigma2, grid,
        n_peak=nodes_per_block, n_zero=nodes_per_block, n_cover=nodes_per_block,
    )
    # Independent of the prior and the component, so computed once and reused
    # across every update -- the efficiency the whole referee depends on.
    loglik = donor_loglik(
        y[:, columns], eta0[:, columns], r[columns], donor_idx, nodes, node_chunk=64
    )

    converged = torch.zeros(len(index), dtype=torch.bool)
    updates = torch.zeros(len(index), dtype=torch.long)
    for iteration in range(1, max_updates + 1):
        previous = sigma2
        _, sigma2, pi, grid, _ = exact_update(loglik, nodes, weights, sigma2, grid, pi)
        moved = (sigma2 - previous).abs() / previous.clamp(min=1e-12)
        newly = (~converged) & (moved < rtol)
        updates[newly] = iteration
        converged |= newly
        if converged.all():
            break
    updates[~converged] = max_updates

    gamma, component_mean, component_m2, _ = exact_estep(
        loglik, nodes, weights, sigma2, grid, pi
    )
    m2 = (gamma * component_m2).sum(dim=2)

    return pd.DataFrame(
        {
            "exact": m2.mean(dim=0).numpy(),
            "converged": converged.numpy(),
            "updates": updates.numpy(),
        },
        index=list(gene_subset),
    )


def accuracy_table(
    runs: dict,
    gene_stats: pd.DataFrame,
    *,
    n_sparse: int = 40,
    n_control: int = 20,
    nodes_per_block: int = 2400,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-gene fitted vs exact `sigma^2` for each method, on a stratified subset.

    `runs` maps method name to a :class:`Run`, one per method, same seed.

    **The referee is run once per method, at that method's own fitted `eta0` and
    `r`.** It integrates `u` out *given* the linear predictor, so its answer
    moves when the predictor does -- measured on `ENSG00000237541`, the exact
    value is 6.6063 at a `mode_polish=True` fit and 7.2174 at a
    `mode_polish=False` one, from the same data. Evaluating both methods against
    a single shared referee would therefore charge one of them for a linear
    predictor it never fitted. The question each row answers is the
    self-consistent one: given the model this method converged to, does its
    variance estimate agree with exact integration over `u`?
    """
    frames = []
    for method, run in runs.items():
        genes = list(run.model.adata.var_names)
        sparse, dense = referee_genes(
            gene_stats, genes, n_sparse=n_sparse, n_control=n_control, seed=seed
        )
        subset = sparse + dense
        exact = referee_sigma2(run.model, subset, nodes_per_block=nodes_per_block)

        sigma2 = pd.Series(run.model.sigma2.cpu().numpy(), index=genes)
        stats = gene_stats.set_index("gene").loc[subset]
        frames.append(pd.DataFrame({
            "config": run.config,
            "method": method,
            "seed": run.seed,
            "gene": subset,
            "n_empty_donors": stats["n_empty_donors"].to_numpy(),
            "mean_expr": stats["mean_expr"].to_numpy(),
            "fitted": sigma2.loc[subset].to_numpy(),
            "exact": exact.loc[subset, "exact"].to_numpy(),
            "referee_converged": exact.loc[subset, "converged"].to_numpy(),
        }))

    table = pd.concat(frames, ignore_index=True)
    table["error"] = table["fitted"] - table["exact"]
    table["sparse"] = table["n_empty_donors"] > 0
    return table
