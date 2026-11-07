"""Design matrices evaluated on the complete adata.obs data

If we don't evaluate the design matrix once on the entire alt data, we risk
"transform state" (e.g., spline knots or categorical levels) which is internally
defined by formulaic when setting a formula not agreeing across chunks.
"""

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional

import numpy as np
import pandas as pd
from formulaic import Formula, ModelSpec
from formulaic.parser.types import Factor
from formulaic.transforms import TRANSFORMS

if TYPE_CHECKING:
    from .spatial import SpatialDesign

# Calls that can be used in chunk formulas but which don't have state
CHUNKED_CALLS = {"C", "log"}


def clip_extrapolation(transform):
    """Clip spline inputs beyond the training data range

    By default, the formulaic splines will raise an error if they see an input
    value beyond the training data range. This can cause problems in our
    training / validation splits, where pseudotime in validation splits lies
    beyond the training range. This class overrides that formulaic default.
    """
    def wrapper(*args, extrapolation="clip", **kwargs):
        return transform(*args, extrapolation=extrapolation, **kwargs)

    wrapper.__dict__.update(vars(transform))
    return wrapper


SPLINES = {name: clip_extrapolation(TRANSFORMS[name]) for name in ("bs", "cc", "cr", "cs")}


@dataclass
class FormulaDesign:
    """Fitted state for one formula.

    Attributes
    ----------
    spec : formulaic.ModelSpec
        Formula state, with ``na_action="raise"``.
    levels : dict of str to pandas.CategoricalDtype
        Fit-time levels of categorical and ``C(column)`` columns.
    predictor_names : list of str
        Columns retained after reference-level selection.
    """

    spec: ModelSpec
    levels: dict[str, pd.CategoricalDtype]
    predictor_names: list[str]

    def matrix(self, obs: pd.DataFrame) -> pd.DataFrame:
        """Evaluate using fit-time levels and retain the selected columns."""
        matrix = self.spec.get_model_matrix(code_levels(obs, self.levels))
        return matrix[self.predictor_names]


@dataclass
class Design:
    """Fitted formulas and (optional) spatial transform.

    ``formulas`` maps each parameter key to :class:`FormulaDesign`.
    ``transform`` appends spatial basis columns before evaluating formulas.
    """

    formulas: dict[str, FormulaDesign]
    transform: Optional["SpatialDesign"] = None

    @property
    def predictor_names(self) -> dict[str, list[str]]:
        """Relevant predictor columns for each formula key."""
        return {key: design.predictor_names for key, design in self.formulas.items()}

    def matrices(self, obs: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Design matrix for each key in formulas."""
        if self.transform is not None:
            obs = self.transform.transform(obs)
        return {key: design.matrix(obs) for key, design in self.formulas.items()}


def fit_design(
    obs: pd.DataFrame,
    formula: Mapping[str, str],
    *,
    partition_keys: frozenset[str] | set[str] = frozenset(),
    transform: Optional["SpatialDesign"] = None,
    chunked: bool = False,
) -> Design:
    """Learn design-matrix rules from the complete observation table.

    This function lets us learn categorical levels and transform state (e.g.,
    spline knots) that can be reused across chunks and queries.

    Parameters
    ----------
    obs : pandas.DataFrame
        Full training covariate table, before any formula has been applied.
    formula : mapping of str to str
        Named Formulaic formulas, e.g. ``{"mean": "~ age + cell_type"}``.
        Each key is associated with a separate output matrix.
    partition_keys : set of str, optional
        Formula keys exempt from reference-column removal. To retain all
        cell-type indicators, use ``"~ 0 + cell_type"`` and include its key
        here. Beware that this option does not remove the formula's intercept.
    transform : SpatialDesign, optional
        Already-fitted spatial transform, applied before each matrix evaluation.
        By this point, formulas must reference the already-transformed columns.
    chunked : bool, default False
        If True, learn state from full columns but build only a one-row matrix
        to determine the relevant structure. This avoids creating the full
        training design matrix. Allowed factors are column references,
        ``C(column)``, ``log(column)`` and state-aware Formulaic transforms of
        one column with constant keywords, e.g. ``bs(column, df=...)`` or
        ``cr(column, df=...)``. If False, build the full matrix while fitting.

    Returns
    -------
    Design
        Fitted rules that can be applied to new chunks.
        ``design.matrices(query)[key]`` evaluates one formula on a covariate
        table (before transforms have been applied). This returns a DataFrame
        with the over the ``query`` rows and has columns
        ``design.predictor_names[key]``. You can think of this as a function
        factory that creates these design-matrix-creating functions.

    Notes
    -----
    String and categorical levels are ordered by decreasing frequency, with
    the most common as reference. Numeric ``C(column)`` levels are sorted.

    Examples
    --------
    >>> obs = pd.DataFrame({"cell_type": ["B", "B", "T"]})
    >>> design = fit_design(obs, {"mean": "~ cell_type"}, chunked=True)
    >>> design.predictor_names["mean"]
    ['Intercept', 'cell_type[T.T]']
    >>> design.matrices(obs.iloc[[2, 0]])["mean"].to_numpy().tolist()
    [[1.0, 1.0], [1.0, 0.0]]
    """
    rows = obs.iloc[:1] if chunked else obs
    if transform is not None:
        rows = transform.transform(rows)

    column_levels = {}
    formulas = {
        key: fit_formula(
            obs, formula_text, rows=rows, column_levels=column_levels,
            partition=key in partition_keys, chunked=chunked,
        )
        for key, formula_text in formula.items()
    }
    return Design(formulas, transform)


def fit_formula(
    obs: pd.DataFrame,
    formula_text: str,
    *,
    rows: pd.DataFrame,
    column_levels: dict[str, pd.CategoricalDtype],
    partition: bool = False,
    chunked: bool = False,
) -> FormulaDesign:
    """Fit a formula and determine its predictor columns

    When ``chunked=True``, categorical levels and state are computed using the
    ``obs`` data frame. Formulaic evaluates the formula on the first row to
    determine column names and categorical encoding structure. Otherwise, it
    evaluates the formula on every row.
    """
    columns, calls = analyze_formula(formula_text)
    if chunked:
        check_chunked(formula_text, calls)

    # evaluate levels in categorical data
    columns &= set(obs.columns)
    categorical_columns = {column for column in columns if is_categorical(obs[column])}
    categorical_columns |= {
        call.column for call in calls.values() if call is not None and call.function == "C"
    } & columns
    for column in categorical_columns - column_levels.keys():
        column_levels[column] = categorical_levels(obs[column])
    levels = {column: column_levels[column] for column in sorted(categorical_columns)}

    # Seed transform state from full columns; the remaining structure needs one row.
    state = transform_state(obs, calls) if chunked else {}
    spec = ModelSpec(formula=formula_text, na_action="raise", transform_state=state)
    matrix = spec.get_model_matrix(code_levels(rows, levels))
    dropped = set() if partition else reference_columns(obs, levels)
    predictor_names = [name for name in matrix.columns if name not in dropped]
    return FormulaDesign(matrix.model_spec, levels, predictor_names)


@dataclass(frozen=True)
class FormulaCall:
    """A simple ``function(column, **literals)`` expression."""

    function: str
    column: str
    keywords: dict


def analyze_formula(
    formula_text: str,
) -> tuple[set[str], dict[str, Optional[FormulaCall]]]:
    """Collect referenced names and classify Python expressions.

    This is used to extract function names from formulas when creating
    FormulaCalls. If the syntax is not recognized by parse_call, the associated
    python expression gets mapped to None.

    We need this analysis so that we can compute full .obs data level summary
    statistics, ensuring that calls on batches are consistent with each other.

    For example,
        bs(age, df=4)
    gets read into
        FormulaCall(function="bs", column="age", keywords={"df": 4})
    which then lets us find the knots to be used in each batch. Similar
    operations are available for C(), to find the underlying levels.
    """
    names = set()
    calls = {}
    factors = [
        factor
        for term in Formula(formula_text)
        for factor in term.factors
    ]

    for factor in factors:
        expression = factor.expr

        # A direct column reference, such as age.
        if factor.eval_method is Factor.EvalMethod.LOOKUP:
            names.add(expression)
            continue

        # Constants need no analysis; repeated expressions are already recorded.
        if factor.eval_method is not Factor.EvalMethod.PYTHON:
            continue
        if expression in calls:
            continue

        # A Python expression, such as C(batch) or log(age + 1).
        tree = ast.parse(expression, mode="eval")
        referenced_names = {
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        names.update(referenced_names)
        calls[expression] = parse_call(tree.body)

    return names, calls


def parse_call(node: ast.AST) -> Optional[FormulaCall]:
    """Parse terms in``function(column, keyword=value, ...)``.

    This returns the function name, column name, and keyword values as a
    ``FormulaCall``.  The function and column must be bare names, with exactly
    one positional argument It returns ``None`` for any other expression.
    """
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and all(isinstance(keyword.value, ast.Constant) for keyword in node.keywords)
    ):
        return None
    return FormulaCall(
        function=node.func.id,
        column=node.args[0].id,
        keywords={keyword.arg: keyword.value.value for keyword in node.keywords},
    )


def check_chunked(formula_text: str, calls: Mapping[str, Optional[FormulaCall]]) -> None:
    """Reject unallowed formula terms in chunked data."""
    for expr, call in calls.items():
        supported = call is not None and (
            call.function in CHUNKED_CALLS or is_stateful(call.function)
        )
        if not supported:
            raise ValueError(
                f"Chunked fitting does not support {expr!r} in {formula_text!r}. Chunked "
                "formulas may use column references, categorical columns, C(column), "
                "log(column) or state-aware Formulaic transforms of a column with ",
                "eg. bs(column, df=...) or cr(column, df=...). For more complex",
                "transformations, add them as covariates in .obs."
            )


def is_stateful(function: str) -> bool:
    """Whether ``function`` is a Formulaic transform that is aware of state."""
    return getattr(TRANSFORMS.get(function), "__is_stateful_transform__", False)


def transform_state(obs: pd.DataFrame, calls: Mapping[str, Optional[FormulaCall]]) -> dict:
    """Formulaic state for a given formulaic call, based on a column of ``obs``.

    This uses the fact that Formulaic has a ``__state`` argument. We don't
    actually use the formulaic evaluation, only the filled in transform.
    """
    state = {}
    for expr, call in calls.items():
        if call is not None and is_stateful(call.function):
            state[expr] = {}
            TRANSFORMS[call.function](obs[call.column], **call.keywords, _state=state[expr])
    return state


def is_categorical(values: pd.Series) -> bool:
    return isinstance(values.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(values.dtype)


def categorical_levels(values: pd.Series) -> pd.CategoricalDtype:
    """Frequency-ordered levels of categorical columns; sorted levels otherwise."""
    if is_categorical(values):
        return pd.CategoricalDtype(values.value_counts().index)
    return pd.CategoricalDtype(np.sort(values.dropna().unique()))


def code_levels(obs: pd.DataFrame, levels: Mapping[str, pd.CategoricalDtype]) -> pd.DataFrame:
    """Cast columns to fit-time levels; unseen values become missing."""
    if not levels:
        return obs
    return obs.astype(levels)


def reference_columns(obs: pd.DataFrame, levels: Mapping[str, pd.CategoricalDtype]) -> set[str]:
    """Reference columns to omit from regression designs without an intercept.

    With an intercept, Formulaic already drops the reference and uses ``[T.]``
    column names. Partition designs skip this selection to keep all indicators.
    """
    return {
        f"{column}[{dtype.categories[0]}]"
        for column, dtype in levels.items() if is_categorical(obs[column])
    }
