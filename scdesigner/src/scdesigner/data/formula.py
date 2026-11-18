from typing import Optional, Union
import ast
from formulaic import Formula
from formulaic.parser.types import Factor
import re
import warnings

OFFSET_PREFIX = "offset:"


def offset_key(parameter: str, prefix: str = OFFSET_PREFIX) -> str:
    return prefix + parameter


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


def split_offset_terms(formula: str) -> tuple[str, Optional[str]]:
    """Separate an ``offset(...)`` term from a formula.

    The offsets need to be treated separately in the initialization and
    optimization, so we split them off from the main formula.

    Parameters
    ----------
    formula : str
        A formula potentially with an ``offset(expression)`` term.

    Returns
    -------
    fixed : str
        The formula without the offset. If there wasn't an offset in the first
        place, the formula is returned as is.
    offset : str or None
        A formula with the offset or ``None`` if there is no offset. Puts
        expressions that need to be evaluated in to brackets {}.

    Examples
    --------
    >>> split_offset_terms("~ x + offset(log(lib))")
    ('~ x', '~ 0 + {log(lib)}')
    >>> split_offset_terms("~ 0 + offset(lib)")
    ('~ 0', '~ 0 + `lib`')
    """
    rhs = formula.split("~", 1)[-1]

    # find the offset prefix
    kept, offsets = [], []
    for summand in split_top_level(rhs, "+"):
        if summand.startswith("offset(") and closing_paren(summand[6:]) == len(summand) - 7:
            offsets.append(summand[7:-1].strip())
        else:
            kept.append(summand)

    # some checks
    if len(offsets) > 1:
        raise ValueError("At most one offset() term is allowed per formula; combine offsets in a column of .obs")
    if not offsets:
        return formula, None

    # reconstructed tuple
    quoted = f"`{offsets[0]}`" if offsets[0].isidentifier() else "{" + offsets[0] + "}"
    return "~ " + (" + ".join(kept) or "1"), "~ 0 + " + quoted


def split_random_terms(formula: str) -> tuple[str, list[tuple[str, str]]]:
    """Separate lme4-style random-effect terms from a formula.

    We parse random-effect terms formatted like ``(1 | donor)``, since they
    aren't supported natively in formulaic.

    Parameters
    ----------
    formula : str
        A formula whose right-hand side may include random-effect terms.

    Returns
    -------
    fixed : str
        The formula without random-effect terms, or ``"~ 1"`` if the formula
        only included random effects.
    random : list of tuple of str
        ``(lhs, group)`` pairs in order of appearance, e.g. ``("1", "donor")``.

    Examples
    --------
    >>> split_random_terms("~ genotype * cell_type + (1 | donor)")
    ('~ genotype * cell_type', [('1', 'donor')])
    """
    rhs = formula.split("~", 1)[-1]
    fixed, random = [], []
    for summand in split_top_level(rhs, "+"):
        is_group = summand.startswith("(") and closing_paren(summand) == len(summand) - 1
        sides = split_top_level(summand[1:-1], "|") if is_group else [summand]
        if len(sides) > 1:
            random.append((sides[0], sides[1]))
        elif summand:
            fixed.append(summand)
    return "~ " + (" + ".join(fixed) or "1"), random


def split_top_level(text: str, separator: str) -> list[str]:
    """Split on ``separator`` outside of brackets

   This also removes surrounding whitespace.
    """
    parts, depth, start = [], 0, 0
    for i, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def closing_paren(text: str) -> int:
    """Index of the bracket closing the one at ``text[0]``, or -1."""
    depth = 0
    for i, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return i
    return -1
