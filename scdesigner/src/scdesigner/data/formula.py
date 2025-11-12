from typing import Union
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