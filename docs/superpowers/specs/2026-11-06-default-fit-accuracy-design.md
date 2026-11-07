# Default-fit accuracy study design

Status: draft for approval. The staged approach is agreed; the experimental choices below are proposed. No experiments or additional package-default changes have been made for this study.

## Intent

Find shared scDesigner fitting defaults that approach the scDesign3 accuracies in `../scDesigner_validation/analysis/revised_report.qmd`, without requiring users to tune each dataset. Explain which differences are attributable to optimization, initialization, regularization, or spatial basis capacity.

Accuracy takes priority over preserving the present speed advantage. Exact timing comparisons, reproducing cluster hardware, and rerunning scDesign3 are unnecessary. Record elapsed time and epoch counts only to identify impractical settings and break ties between similarly accurate configurations.

This is an experimental project: a reproducible runner and analysis, followed by evidence-based recommendations. A negative result is useful. Do not promise that shared NB likelihoods imply equivalent fitted models or simulation quality.

## Scope

- Treat the newly committed patience, coordinate-scaled GP length scale, and penalized-model initialization as intentional defaults to evaluate.
- Use staged ablations rather than an unrestricted joint hyperparameter search.
- Test whether the GP changes permit higher spatial capacity without unstable optimization.
- Investigate learning rate, stopping behavior, regularization, and initialization without implementing MGCV-style gene-specific smoothing selection.
- Prefer a common NB fitting policy. A spatial-recipe default may differ from the ordinary NB recipe, but no preset may branch on dataset identity or use a dataset-specific tuned value.
- Leave dataset formulas, response families, gene selection, and copula choices fixed to the benchmark. Conclusions concern fitting defaults conditional on those inputs, not a fully automatic choice of scientific model.

## Approaches

The chosen approach screens explicit factors on development datasets, locks shared presets, then checks transfer to other benchmark datasets. It provides interpretable comparisons and limits the number of expensive fits.

The alternative was a joint hyperparameter search. It could discover strong individual configurations but would make attribution and selection of reusable defaults harder. Per-dataset searches are excluded because they do not answer the user's question.

## Baselines and evidence

Package baseline: commit `69b30a2cd6acba83f7053fffa394e2b8de05e8fc`. The validation checkout inspected was `d0fcbd2db5e87c4390106728549cf57ecc2fabd0`; its vendor checkout was still `dc8d803b45c72450ee5cfa02680e18993ed0e2a3`. Package version strings alone do not identify the code used.

| Setting | Historical benchmark configuration | Committed package default |
| --- | --- | --- |
| Patience | 6 | 40 |
| Maximum epochs | 500 | 500 |
| Minimum epochs / relative loss tolerance | 10 / `1e-4` | 10 / `1e-4` |
| Learning rate, ordinary NB | `0.005` | `0.005` |
| Learning rate, spatial NB | `0.0005` | `0.005`, inherited from the predictor |
| Spatial mean basis | TPS, 30 columns requested | GP, 100 columns requested |
| GP length scale | Not applicable to the TPS benchmark | `0.17 * max(coordinate ranges)` |
| Spatial penalty strengths | `lam=lam_disp=1` | `lam=lam_disp=1` |
| Initialization | Historical implementation must be verified; the pre-change NB initializer used 10 Poisson IRLS iterations | 0 iterations with active penalties; 10 otherwise |
| Optimizer weight decay | Not explicit in YAML | AdamW's `0.01` unless overridden |

The validation model YAML still pins historical patience and spatial settings. A run using that YAML unchanged is not a test of current package defaults.

The report selects the latest completed attempt per dataset/model/device/seed, preferring completion over recency. Its current inputs contain 71 selected combinations, 55 completed, and 16 datasets with completed scDesign3 and NB-scDesigner results. All completed reference runs use seed 42. The report's framework medians pool devices and, for sciATAC, also include a scDesigner ZIP run. Furthermore, sciATAC's saved `scd3_nb` run actually sets `family_use=zip`. Preserve those original figures as context, but do not pool models or devices. Include sciATAC as a practical comparison to the requested reference, separately labeled as a different-family comparison; it cannot establish an NB optimization gap.

Examples from the report's pooled summaries:

| Dataset | scDesigner log-expression variance correlation | scDesign3 |
| --- | ---: | ---: |
| ovarian | 0.753 | 0.993 |
| acinar | 0.706 | 0.976 |
| embryo | 0.753 | 0.905 |

Embryo reaches epoch 500 with its lowest printed validation loss at that epoch. Ovarian and acinar compare TPS with 30 requested columns against scDesign3 GP smooths with `k=400`. These observations motivate experiments; they do not establish causes. Matching `df` and `k` does not make the bases or penalties identical.

Maintain distinct baseline labels:

- `historical-results`: immutable saved metrics and their selected attempt IDs.
- `historical-settings-replay`: historical settings explicitly applied to current code, including initialization. This is not an exact reconstruction of the old software.
- `package-defaults`: current package constructor/fit defaults, with benchmark scientific inputs and explicit reproducibility/device controls.

## Data and evaluation protocol

### Reference selection

Reproduce the selection in `revised_report.qmd`, including completion-before-recency and the full run key. Freeze the resulting manifest before fitting. Use completed `scd3_nb` runs as the reference; retain missing and failed combinations in a coverage table rather than treating them as zero accuracy.

All 16 selected scDesign3 runs currently have saved `train_data.h5ad` and `validation_data.h5ad`. These are the canonical outer training/test partitions. Do not reload a newer dataset release or regenerate HVGs from current configs. Check cell IDs, gene IDs/order, counts, and required covariates against the selected scDesigner artifacts before describing historical results as paired. If they differ, retain the historical metrics with a mismatch label and run new scDesigner fits on the canonical scDesign3 inputs.

Use saved configuration values, resolving their interpolations without substituting current dataset defaults. Record actual response families rather than inferring them from model names. Record hashes of the input artifacts and relevant configurations. A missing canonical artifact blocks an exact-reference comparison; do not silently substitute different data.

Recompute the primary metrics on the saved real/simulated reference artifacts as a metric-compatibility audit. Keep the original numbers unchanged. Unexplained differences beyond numerical tolerance block claims of direct comparability until the source or preprocessing mismatch is resolved.

### Development and transfer datasets

Proposed development panel:

- Spatial: ovarian and acinar for substantial deficits; prostate as a less-degraded spatial control.
- Count: embryo for unfinished convergence; cite and HVG_embryoatlas for variance or distributional deficits; pancreas and batch as controls.

Reserve granja_atac, ifnb, mobsc, mobsp, mouse_cortex, mouse_visual, sciatac, and seqfish for transfer checks after presets are locked. Baseline comparisons on these datasets may be displayed at final evaluation, but must not guide candidate selection.

Datasets without completed scDesign3 results remain optional stability checks, not evidence of matching scDesign3. The original figures have already been inspected, so this is a development/transfer separation within an existing benchmark, not a wholly untouched external validation study.

### Tuning without using the final test cells

Within each canonical outer training partition, reserve 20% of cells as an experiment tuning partition using split seed 314159. Candidate fitting, spatial-basis construction, copula fitting, and output-cap estimation see only the remaining 80%. The package retains its own 10% internal marginal-validation split for checkpoint selection, with `validation_seed=42` fixed across candidates.

Use the external tuning partition for candidate likelihood and simulation comparisons. Do not use the package's internal stopping set to assess full-simulator fidelity: the copula and other fitted state can use those internal cells.

After locking presets, refit on the complete canonical outer training partition and evaluate at the saved outer test covariates. Keep the package's internal validation behavior unchanged. The historical outer test metrics are comparison targets only; do not use new final-test scores to select presets. If confirmation leads to another round of development, label it exploratory and acknowledge reuse of the benchmark.

### Randomness and devices

Screen with fit seed 42. Confirm shortlisted presets on development data with fit seeds 42, 89, and 123 before locking them; use the same seeds for final fits. Separate fit RNG, internal-split seed, simulation RNG, and metric RNG in the manifest. For each fit, use simulation seeds 1001, 1002, and 1003 for finalist evaluation; seed 1001 suffices for screening. Hold the metric/PCA/oracle seed at the reference value 42 so simulation repetitions do not also change the evaluation procedure.

Keep data partitions fixed across these repetitions. They measure fitting and sampling variability, not uncertainty over new train/test splits. The saved scDesign3 reference is one realization and has no estimated sampling uncertainty.

Screen on one explicitly selected available device and record it; do not silently fall back to another device. Retain device labels in every comparison. A small CPU/CUDA accuracy check is optional when both are available, not a hardware benchmark or a requirement for finishing the study.

## Experiments

Use the same batch size (1024), response representation, formulas, and copula settings throughout screening. Keep `val_frac=0.1`, `min_epochs=10`, standardization, sample caps, and numerical floors fixed unless a named experiment explicitly changes them. Pin initialization during every comparison: setting `lam=0` must not implicitly enable Poisson IRLS.

### Baseline audit

Run `package-defaults` and `historical-settings-replay` on the development tuning partitions. Later evaluate them alongside locked candidates on the canonical outer test partitions. Record resolved defaults from actual estimator/optimizer state, not only YAML. Preserve the saved historical results separately.

Verify that the installed `scdesigner` is this package checkout, not the validation repository's older vendor installation. Record commit, import path, source-diff hash, and versions in every run.

### Optimization screen

At the package-default basis and initialization, screen:

| Factor | Count NB | Spatial NB |
| --- | --- | --- |
| Learning rate | `0.001, 0.005, 0.01` | `0.0005, 0.001, 0.005` |
| Patience | `6, 40` | `6, 40` |
| Maximum epochs | 2000 | 2000 |
| Relative loss tolerance | `1e-4` | `1e-4` |

The 500-epoch package baseline distinguishes extra training from other changes. Retain at most two shared training policies per model family using the selection rule below. On those policies, compare `loss_tol=1e-5` against `1e-4`; do not simultaneously change patience.

If a shortlisted policy hits its budget and its last 100 epochs improve validation NLL by more than `1e-4` relative to the value 100 epochs earlier, test a 4000-epoch version on the whole development panel for that family. Otherwise retain the budget and label unresolved convergence when appropriate. Do not introduce per-dataset epoch limits.

### GP length scale, initialization, and capacity

On the spatial development panel, use the leading shared training policy and cross:

- Length scale: explicit `length_scale=1.0` versus `length_scale=None, length_fraction=0.17`.
- Initialization: explicit `max_iter=10` versus `max_iter=0`.
- Requested mean basis columns: `30, 100, 200, 400`.

Hold `n_landmarks=400` across this factorial comparison so changing capacity does not also change landmark count. Include a GP-100 anchor with 200 landmarks to connect back to the package default. Reuse the same landmark selection for matched factors and a fixed seed; record actual column count when observations or rank limit the request. A truncated request is not evidence about the requested larger model.

Keep the benchmark dispersion formula and spatial dispersion setting unchanged. The legacy absolute scale is a diagnostic control, not a candidate general default. Inspect basis behavior at held-out coordinates, not only at landmarks.

For the leading capacity/initialization combination, compare `length_fraction=0.08, 0.17, 0.34`. The scaling rule must be identical across datasets; do not choose a separate fraction per tissue. Check the leading structural choices under the second retained learning policy when available, to avoid declaring a basis unstable solely because of one learning rate.

### Regularization screen

On the leading structures, compare `lam=0, 0.1, 1, 10` and `weight_decay=0, 0.01`. For ordinary NB, compare only weight decay; do not add a new penalty. Pin `max_iter` even for `lam=0`.

Keep dispersion structure and `lam_disp` fixed unless a spatial dispersion term is actually present. Do not report an inactive `lam_disp` change as an experiment. Increasing dispersion flexibility is outside this initial study because it changes the scientific specification being compared.

The explicit penalty is added to a summed minibatch NLL. Therefore its effective strength depends on batch size, including a small final batch. Record realized batch sizes and the penalty/NLL ratio; do not change batching during this screen or silently reinterpret `lam` as a normalized smoothing parameter.

Recheck the resulting combined preset against its parent configurations across the development panel. Do not infer that individually beneficial changes must remain beneficial together. An exhausted screen that fails to find broadly useful settings is a result, not a reason to start tuning each dataset.

## Diagnostics and metrics

For each fit, retain the existing per-epoch history, selected checkpoint, stop reason, and whether the epoch budget was exhausted. Distinguish the minimum validation loss in the history from the checkpoint accepted by `loss_tol`; the current implementation uses that tolerance for both patience reset and checkpoint saving.

Evaluate unpenalized NB NLL on the external tuning/test partition in batches, using the fitted design. Save mean NLL per cell-gene entry and per-gene values. Stratify diagnostics by training expression/zero fraction and by available cell labels. These likelihood comparisons are between scDesigner configurations: the saved scDesign3 runs did not retain marginal models, so no scDesign3 NLL comparison is promised.

For spatial fits, record coordinate extent, resolved length scale, requested/realized basis width, basis spectrum or conditioning summaries, penalty-diagonal summaries, and basis magnitude on held-out cells. For all fits, check non-finite coefficients/predictions/losses, extreme fitted means or NB sizes, and the fraction of simulated entries affected by the observed-maximum cap. Initialization diagnostics must include the starting mean/dispersion ranges. Do not hide a failed fit by clipping its metrics or selecting a more fortunate seed.

Reuse the validation metric implementations unchanged. Primary simulation metrics are:

- `mean_log_expression_correlation`
- `var_log_expression_correlation`
- `gene_zero_fraction_correlation`
- `gene_correlation_correlation`
- `log_library_size_ks`
- `cell_zero_fraction_ks`

Also report `lisi_pca` alongside `lisi_pca_real_real`. The real-real value is context, not a hard performance ceiling; very low values weaken the interpretation of mixing scores. Avoid counting `mean_correlation` twice: it is the same log-mean statistic in this implementation. UMAP and SimBench can accompany finalists but do not drive screening.

Preserve the metric implementation's zero-cell filtering for comparability, and separately report the number of zero cells removed. Undefined correlations remain undefined and receive an explanation; a candidate-specific non-finite metric cannot improve its aggregate score by disappearing.

## Selecting shared defaults

These are proposed analysis rules, not thresholds supplied by the user.

First discard candidates with fit failures or non-finite predictions. Relative to `package-defaults`, flag material regressions on the development tuning data as a correlation decrease greater than 0.02 or a KS increase greater than 0.03. An already undefined baseline correlation is not made into a numeric target. If the package baseline itself fails, retain that dataset in coverage and treat a successful candidate as a reliability improvement; do not invent a baseline NLL or metric value.

Prefer candidates without material regressions on any development dataset's evaluable primary metric. Within that set, select the lowest mean within-dataset rank across the primary simulation metrics (rank 1 is best, tied values receive average ranks), weighting datasets equally. Use a common evaluable metric mask across candidates; candidate-specific missing metrics fail eligibility rather than improving rank. Break tied aggregate ranks by median relative improvement in unpenalized tuning NLL over finite package baselines, then lower capacity and fewer epochs. Likelihood is a diagnostic and tie-breaker; the simulation accuracies in the report determine the recommendation. A candidate with regressions may remain a diagnostic comparison, but cannot become the recommended default under this rule. If no candidate improves broadly without regressions, retain current defaults and report the limitation.

Apply the rule across repeated fits/simulations when narrowing to finalists, using median metric values and displaying the individual repetitions. Lock one common fitting policy if it passes for both families; otherwise allow one ordinary-NB and one spatial-NB preset. Presets may contain a fixed coordinate-relative length-scale rule and a fixed landmark/rank policy, but no dataset-name lookup or manual per-dataset choice.

For final comparisons, mark a dataset/metric as comparable to the saved scDesign3 result when the median deficit is at most 0.02 for correlations or 0.03 for KS. Show the actual deficits and repetition ranges next to these descriptive tolerances. Use 0.05 as a descriptive PCA-iLISI deficit tolerance, with its oracle caveat. These tolerances are not statistical equivalence tests. Report coverage explicitly rather than declaring success from one composite average, and list datasets with remaining substantial gaps.

A useful recommendation must improve the problematic settings without materially degrading the controls or failing transfer checks. If final evaluation exposes exceptions, report them; do not replace the shared preset with a per-dataset winner. Better likelihood with unchanged dependence metrics should motivate a separate copula investigation, not further blind tuning of marginal defaults.

## Code and artifact boundaries

Keep the experiment on this package branch in a new `validation/defaults_study/` directory. The neighboring validation repository and historical outputs are read-only inputs. This avoids mixing the experiment with existing uncommitted notebook/config work there.

Proposed responsibilities:

| File or area | Responsibility |
| --- | --- |
| `validation/defaults_study/reference.py` | Report-compatible attempt selection, canonical artifact/config checks, frozen reference manifest |
| `validation/defaults_study/experiments.py` | Explicit baseline definitions and bounded stage matrices; resolved settings and seed policy |
| `validation/defaults_study/run.py` | Read saved partitions, call existing fit/sample/metric code, persist successful and failed attempts, resume by exact configuration identity |
| `validation/defaults_study/diagnostics.py` | Batched NLL and fitted-state/basis diagnostics |
| `validation/defaults_study/analysis.py` | Paired summaries, candidate-selection rules, coverage and regression flags |
| `validation/defaults_study/report.qmd` | Experimental results and defaults recommendation, separate from the original report |
| `validation/defaults_study/tests/` | Synthetic tests of selection, partitions, matrices, diagnostics, and aggregation |
| `validation/defaults_study/README.md` | Environment/import checks and commands for each stage |

Reuse `src/model.py`, `src/metrics.py`, and the splitting helper from `../scDesigner_validation/`; do not copy the model or metric implementations. Pass the validation repository path explicitly, verify required interfaces, and record its commit and relevant source diff. Do not make the new experiment depend on the uncommitted `src/interactive.py` notebook helper.

Store generated artifacts under a separately ignored experiment output directory. Each run records its stage, preset/config hash, source provenance, data hashes, seeds, device, status, and failure reason. Persist resolved config, epoch history, diagnostics, scalar metrics, per-gene NLL, and coarse elapsed time. Save simulated data for baselines/finalists; large screening outputs can be omitted. Resume only when input, source, and configuration identities match. Never select retries by metric quality.

No likelihood, optimizer algorithm, or production default changes are required to start the study. Any minimal observation hook needed for diagnostics must leave fitted behavior unchanged and have a regression test. Recommend later production changes only after evidence is available; approval of this design does not preselect their values.

## Verification requirements

The implementation plan must include tests for:

- Completed-versus-incomplete retries, missing/non-finite metrics, CPU/CUDA separation, scDesigner ZIP exclusion from NB summaries, and sciATAC's different-family scDesign3 reference.
- Metric compatibility with saved artifacts, fixed metric randomness across simulation repetitions, and handling of a failed package baseline without fabricating values.
- Saved-data identity checks and disjoint tuning/test partitions; no fitting of basis, copula, or output cap on external tuning/test cells.
- Actual package-default resolution and rejection of an unintended vendor-package import.
- Explicit initialization across `lam=0`, realized-rank limits, common landmarks in matched comparisons, and unused dispersion-penalty settings.
- Batched NLL agreement with direct NB likelihood on a small fixture, with finite and degenerate genes.
- Equal dataset weighting, metric direction, repeated-seed summaries, candidate-specific missing metrics, and enforcement of one preset per family.
- A small end-to-end synthetic fit and resumable persistence without reading the large benchmark datasets.

Scientific verification consists of the frozen-reference audit, the ablation results, and the final locked-preset comparisons. Unit tests cannot establish improved biological simulation accuracy.

## Proposed choices and remaining implementation decisions

The development/transfer split, factor grids, repetition counts, descriptive tolerances, and file placement are design proposals for approval. The original report and user priorities are the requirements; those proposals must not be presented as user-specified values.

The implementer may choose the local Python environment, table storage format, plot layout, and sequential execution mechanism. Prefer the existing validation environment and dependencies. A scheduler integration, automated hyperparameter-search service, exact timing study, new copula method, or new adaptive smoothing algorithm is not part of this project.
