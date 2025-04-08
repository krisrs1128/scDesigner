import numpy as np
from copy import deepcopy
from typing import Union


def substitute(
    params: dict, id: str, new_value: np.array, mask: Union[np.array, None] = None
) -> dict:
    if mask is None:
        mask = np.ones(params[id].shape)

    result = deepcopy(params)
    result[id][mask] = new_value
    return result
