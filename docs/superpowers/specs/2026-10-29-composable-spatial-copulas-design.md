# Composable spatial copulas

Status: architecture approved; delivery split into three separately reviewed merge requests. MR 1 planning is authorized; implementation awaits review of its plan.

## Intent

Support a spatially varying negative-binomial mean with dispersion constant across coordinates, and provide the analogous spatial Bernoulli model. Separate spatial basis construction and coefficient penalization from distribution-specific likelihoods and predictors. Compute per-feature summaries without materializing the full response matrix.

The user selected explicit spatial terms plus a `spatial_copula` composition function, with predefined simulators supplying convenient defaults. Inverse-mean weighting is acceptable for both NB and Bernoulli features. Backward compatibility is not required.

## Goals and boundaries

- Specify formulas independently for each marginal parameter. A parameter without a spatial term receives no spatial basis or spatial penalty.
- Keep `SpatialNegBinCopula` and introduce `SpatialBernoulliCopula` as thin predefined constructors using the composition function.
- Support coefficient penalties independently of spatial modeling, including a penalized Bernoulli marginal.
- Respect `chunk_size` for feature summaries and preserve chunked access throughout spatial fitting on backed AnnData.
- Retain fitted basis state for prediction, sampling, and external-data likelihood evaluation.
- Do not introduce a hierarchy of distribution-specific spatial or penalized implementations.
- Do not promise constant memory in the number of observations or features: observation metadata, spatial basis matrices, model coefficients, and copula covariance matrices still occupy memory. Prediction and sampling continue returning complete outputs.
- Do not redesign basis algorithms, covariance estimators, NB initialization, or optimizer defaults. Do not generalize the recipe to every marginal family in this change.

Use the existing Python >=3.10 stack (AnnData, NumPy/SciPy, Formulaic, PyTorch); no new dependencies are needed.

## Delivery and review milestones

Implement and merge these in order. Each merge request has its own implementation plan, code review, and explicit human merge approval. Start the next MR from the merged predecessor; do not implement all milestones on one branch. The sections below describe the final design; this milestone breakdown determines when each requirement applies.

### MR 1: Composition and the NB recipe

Extract distribution-independent penalties, introduce `SpatialTerm` and `spatial_copula`, and replace NB convenience implementations with thin factories. Support independent parameter formulas, spatial mean with constant dispersion, fitted-basis reuse, and refitting without shared state. Preserve NB initialization and existing penalty scaling. Add task-level reviews after penalty extraction/training integration and after spatial composition, followed by a whole-MR review.

This MR supports the new composed spatial recipe on in-memory dense/sparse AnnData. It may retain full-response materialization for summaries and the temporary AnnData wrapper used for augmented observations. Keep these operations isolated for replacement in MR 2; clearly reject backed input in new paths that cannot yet handle it. Existing nonspatial paths without the new components retain their current data support. Do not change loader chunk selection, copula `top_k` reductions, or Bernoulli behavior here.

### MR 2: Chunked response access

Replace summary materialization with bounded row reads; preserve backed storage while supplying augmented observations; honor explicit chunking for in-memory inputs. Verify the entire response-access path, including initialization, copula `top_k`, and NB capping. Add instrumented bounded-read tests for dense/sparse storage and partial final chunks. Remove MR 1's temporary backed-input restrictions. Observation metadata, basis matrices, and returned outputs remain outside the bounded-response-read guarantee.

### MR 3: Bernoulli recipes

Add `PenalizedBernoulli` and `SpatialBernoulliCopula` through MR 1's components and MR 2's data handling. Apply inverse-prevalence weighting, binary-input validation, and a stable Bernoulli likelihood. Review that no distribution-specific training loop or penalty implementation was duplicated, and test saturated probabilities and constant binary features.

Only MR 1 receives a detailed implementation plan now. Plan subsequent MRs after predecessor review/merge, using the interfaces that actually ship.

## Approaches considered

### Explicit components plus recipe function: selected

A spatial term couples a stateful basis with a penalty specification. A recipe adds those terms to an existing marginal and constructs the simulator. Penalties attach to training without replacing the distribution's predictor. Predefined constructors call the same recipe.

This needs small explicit integration points for observation transformation and training penalties, but leaves likelihoods, initialization, and link functions with their existing marginal classes.

### Nested model wrappers

`Spatial(Penalized(NegBin(...)))` is expressive, but the current mutable marginal interface would require coordination of formulas, loaders, predictor state, and fitted parameters across wrappers. This was not selected because it risks the entanglement the user wants to avoid.

### Family-selecting factories alone

A factory selecting NB or Bernoulli makes standard recipes easy to construct but does not itself separate distribution-specific training from penalties. Factories are useful as the outer convenience layer, not the mechanism for sharing all behavior.

## Public construction

The following is the proposed API, not currently available code:

```python
mean_term = SpatialTerm(
    basis=GPBasis(df=15, standardize=True),
    penalty=QuadraticPenalty(lam=1.0, weighting="inverse_mean"),
)

sim = spatial_copula(
    marginal=NegBin({"mean": "~ 1", "dispersion": "~ 1"}),
    terms={"mean": mean_term},
    copula=StandardCopula("~ 1"),
    spatial_cols=("spatial1", "spatial2"),
    cap_at_observed_max=True,
)
```

Replacing the marginal with `Bernoulli({"mean": "~ 1"})` and disabling count capping constructs the Bernoulli recipe. Omitting `copula` creates `StandardCopula("~ 1")`. The generic recipe defaults capping to false; it rejects enabling count capping for Bernoulli.

Base formulas retain their intercept and covariate semantics. Spatial basis columns are appended only for keys in `terms`; adding spatial terms must not silently remove an intercept. For example, `{"mean": "~ cell_type", "dispersion": "~ batch"}` can receive a spatial term only on its mean. Unknown term keys are rejected.

### Predefined simulators

Proposed defaults and convenient entry points:

```python
# Spatial mean; intercept-only dispersion.
nb = SpatialNegBinCopula()

# Spatial mean with additional covariates; nonspatial batch-specific dispersion.
nb = SpatialNegBinCopula(
    mean_formula="~ cell_type",
    dispersion_formula="~ batch",
)

# Spatial effects on both parameters: TPS bases of rank 15 (mean) and 5 (dispersion).
nb = SpatialNegBinCopula(basis="tps", mean_df=15, disp_df=5)

binary = SpatialBernoulliCopula(mean_formula="~ cell_type")
```

The predefined constructors take scalar settings and build their `SpatialTerm`s internally; they do not accept `SpatialTerm` objects. Per-parameter bases, unpenalized terms, other weightings, and custom basis classes use `spatial_copula` directly.

NB arguments: `mean_formula`, `dispersion_formula`, and `copula_formula` (each default `"~ 1"`); `mean_df=15`; `disp_df=None`, where `None` gives the dispersion no spatial term; `basis="gp"` (`"gp"` or `"tps"`), shared by both terms; `standardize=True`; `lam=1.0` and `lam_disp=1.0`, the strengths of inverse-mean quadratic penalties; `spatial_cols=("spatial1", "spatial2")`; `cap_at_observed_max=True`; `estimator` and `top_k` for the standard copula; `device`; and `**basis_kwargs` (for example `n_landmarks`, `length_scale`) passed to both bases. Bernoulli mirrors the mean-side arguments and has no capping argument.

These PascalCase names are thin factory functions returning configured `SCD3Simulator` instances. They preserve familiar construction syntax without another class hierarchy. Class identity, legacy argument names, and legacy import paths need not be preserved.

`PenalizedNegBin` and new `PenalizedBernoulli` are likewise convenience factories over the ordinary marginals plus coefficient-penalty specifications. They do not override `setup_optimizer`. Ordinary `NegBin` and `Bernoulli` also accept penalty components directly, so these convenience names are optional.

## Components and responsibilities

### Spatial terms and observation transformation

`SpatialTerm` contains a basis and an optional `QuadraticPenalty`. A term without a penalty adds unpenalized basis columns. The recipe owns a parameter-keyed collection of terms and fits each basis once per simulator fit.

Keep original base formulas separately from effective formulas so refitting never appends the same columns twice. Use parameter-qualified generated names such as `sp_mean_0` and `sp_dispersion_0`. Reject collisions with input observation columns rather than overwriting data.

At prediction, sampling, and external-data complexity evaluation, use each fitted basis's `transform`; never refit landmarks or standardization on new coordinates. Missing/nonfinite coordinate values fail clearly. Refitting clears old basis, penalty, and summary state. Prediction before fitting fails clearly.

`spatial_copula` accepts unfitted NB or Bernoulli marginal instances, copulas, and term specifications, and takes independent copies of their configuration. Reusing a component across recipes must not share fitted state. Reject already-fitted inputs rather than silently transferring their fitted state.

### Distribution-independent coefficient penalties

A `QuadraticPenalty` specifies `lam` and weighting (`"inverse_mean"` or `"none"`). Binding it to a marginal parameter requires an ordered list of design-column names and a matching nonnegative diagonal. A nonspatial user can supply this binding explicitly; a spatial term supplies its generated names and fitted basis diagonal automatically.

For each bound penalty, evaluate:

`P = lam * sum_j w_j * sum_k diagonal[k] * coefficient[column[k], j]**2`

For inverse weighting, `w_j = 1 / max(feature_mean[j], 0.01)`. Bernoulli means are feature prevalences. Equal weighting uses `w_j = 1`. An all-zero feature therefore has finite weight 100. Only selected columns are penalized; intercepts and unrelated covariates are untouched.

Resolve coefficient rows by exact design-column names, not by assuming that spatial columns occupy the final rows. Reject missing/duplicate names, unknown parameters, diagonal-length mismatches, and nonfinite/negative strengths or diagonal entries. Permit `lam=0`.

The marginal training loop adds bound penalties to the existing predictor loss. Binding happens after the marginal creates its own predictor; the penalty does not rebuild it, change its link functions, initialize its coefficients, or own an optimizer. Validate that the predictor exposes the requested coefficient matrices. Without penalties, the training path is unchanged. Construct penalty tensors on the coefficient device and dtype.

Preserve the current objective convention: summed negative log-likelihood plus the full penalty on each minibatch. `batch_size` therefore retains its current influence on effective regularization; `chunk_size` controls I/O and does not rescale penalties. Changing penalty normalization is outside this work. Early stopping and reported likelihood/AIC/BIC remain based on unpenalized likelihood, as today.

### Bernoulli behavior

Use the existing Bernoulli parameter name `"mean"` and sigmoid link. Penalized and spatial Bernoulli models reuse its initialization, uniformization, and inversion. Validate binary response values while reading chunks/batches; reject nonbinary or nonfinite values rather than silently binarizing them. Use a numerically stable logits-based likelihood so saturated predictions and all-zero/all-one features do not produce `0 * log(0)` NaNs. Simulated values remain in `{0, 1}`.

### Feature summaries and chunked access

Provide one response-only feature-summary utility shared by adaptive penalties and observed-maximum capping. Iterate contiguous row slices of at most `chunk_size`, accumulating float64 sums, row count, and optional maxima on CPU. Divide total sums by the total row count; do not average chunk means. Include the short final chunk and sparse implicit zeros. Reject empty datasets and nonfinite response values.

Support dense/sparse in-memory AnnData and dense/sparse backed AnnData. Slice rows before calling `toarray()`, `np.array()`, or `to_memory()`; never materialize the response matrix before slicing. Densification within a bounded chunk is allowed, including a single chunk when the dataset is smaller than `chunk_size`. Summary state is O(number of features); response working storage is O(chunk_size * number of features).

Use the supplied positive integer `chunk_size`; when absent, summaries use 5000 rows, matching the loader's existing backed default. Compute requested means/maxima together, once per preparation pass. The scan is independent of training shuffle, validation splitting, `batch_size`, and `drop_last`, and counts every input row once. Inverse weights and caps use all observations supplied to `fit`, including the validation subset, matching existing behavior. Skip maxima when capping is disabled.

An explicit `chunk_size` also selects chunked loading for in-memory AnnData. With no explicit size, retain preloading for in-memory data and the 5000-row default for backed data. Thread the same choice through marginal fitting, NB initialization, and copula fitting.

Do not reconstruct `AnnData(X=adata.X, obs=augmented_obs, ...)` for spatial fitting. Add an explicit observation-table override to loader/setup interfaces so original backed response storage is retained while design matrices use augmented observations. Check observation length and index order before accepting an override. Design validation must use the same augmented table. This argument must survive `StandardCopula.setup_data` keyword filtering.

The simulator retains the original dataset as its template and its fitted observation transform separately. Calls without explicit observations use the original template observations through that transform; calls with new observations transform those instead. The caller's AnnData and observation table are not mutated.

`StandardCopula._gene_partitions` currently uses `adata.X.sum(axis=0)`. Replace this response-wide reduction with the shared chunked summary utility so `top_k` also works on backed inputs. A separate summary pass here is acceptable; cross-component caching is not required.

## Integration boundaries

Keep spatial recipe/convenience construction in a focused simulator module, spatial observation transformation separate from simulator orchestration, penalty definitions independent of distribution modules, and feature summaries beside the data loader. Extend `SCD3Simulator` with optional observation-transform and output-capping components; no distribution-specific branching belongs in its fit/predict/sample routines. Nonspatial simulators retain their existing behavior.

`Marginal` needs only penalty configuration, binding, and loss addition. `NegBin` and `Bernoulli` retain their predictor setup. Remove the separate NB penalty implementation rather than keeping two implementations. Update package exports and directly affected documentation/examples to the new public constructors. No deprecation layer is needed.

The implementation plan may choose exact internal filenames, binding types, and method signatures within these boundaries. Avoid a general plugin registry, dynamic class generation, or an arbitrary transformation pipeline.

## Acceptance criteria

- The default spatial NB recipe has distinct mean predictions across appropriate coordinates but identical dispersion predictions when only coordinates change. Its dispersion predictor has one intercept coefficient per feature and no spatial penalty.
- Explicit spatial dispersion, nonspatial dispersion covariates, and different mean/dispersion basis ranks work independently.
- Predefined constructors and equivalent explicit recipes produce the same configured behavior under controlled random seeds.
- Standalone penalized Bernoulli fitting works without spatial data; adding penalties preserves the sigmoid link and produces finite losses/gradients for saturated logits and constant binary features.
- Penalty values and gradients match hand calculations, including interleaved selected/unselected columns and inverse weighting at the 0.01 floor.
- Chunked summaries match dense references for dense/sparse and backed/in-memory inputs, with unequal chunk lengths, a single feature, all-zero features, and chunk sizes larger than the dataset.
- Instrumented reads show no response-wide materialization. Summary scans cover all rows exactly once despite shuffled or dropping training batches.
- A backed spatial fit, including `top_k`, initialization, and optional NB capping, retains bounded response reads and does not replace the backed dataset with an in-memory copy.
- Fresh-coordinate prediction, sampling, and external-data complexity evaluation reuse fitted bases without mutating user data. Refitting and separate recipes do not share stale state.
- Existing nonspatial fitting and early-stopping tests continue passing. CPU tests are mandatory; device-sensitive tests run when CUDA/MPS is available.

## Accepted design defaults

These were assistant proposals accepted with the overall design:

- Make spatial mean plus constant dispersion the NB convenience default.
- Keep familiar predefined constructor names as thin factory functions, rather than preserving separate class identities.
- Preserve current per-minibatch penalty scaling and full-fit-data adaptive weights.
- Limit the memory guarantee to response access; spatial basis construction and returned outputs remain in memory.
