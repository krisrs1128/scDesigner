from typing import Union
import re
import warnings

def standardize_formula(formula: Union[str, dict], allowed_keys = None):
    # The first element of allowed_keys should be the name of default parameter
    if allowed_keys is None:
        raise ValueError("Internal error: allowed_keys must be specified")
    formula = {allowed_keys[0]: formula} if isinstance(formula, str) else formula

    formula_keys = set(formula.keys())
    allowed_keys = set(allowed_keys)

    if not formula_keys & allowed_keys:
        raise ValueError(f"formula must have at least one of the following keys: {allowed_keys}")

    if extra_keys := formula_keys - allowed_keys:
        warnings.warn(
            f"Invalid formulas in dictionary will not be used: {extra_keys}",
            UserWarning,
        )

    formula.update({k: '~ 1' for k in allowed_keys - formula_keys})
    return formula


def partition_formula(formula: str) -> str:
    """Modify a formula to partition observations.

    We remove the intercept unless the formula has no other terms, so that ``"~
    cell_type"`` is a proper indicator across cell types (and there is no
    intercept-like column equal to 1 across all cells).

    Parameters
    ----------
    formula : str
        A copula group formula.

    Returns
    -------
    str
        The normalized formula.

    Examples
    --------
    >>> partition_formula("~ cell_type")
    '~ -1 + cell_type'
    """
    rhs = formula.split("~", 1)[-1]
    terms = []
    for raw_term in rhs.split("+"):

        # strip out irrelevant whitespaces
        term = re.sub(r"-\s*[01]\b", "", raw_term).strip()
        if term and term not in {"0", "1"}:
            terms.append(term)
    if not terms:
        return "~ 1"
    return "~ -1 + " + " + ".join(terms)
