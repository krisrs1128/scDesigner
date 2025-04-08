import numpy as np
from typing import Union
from copy import deepcopy


def amplify(
    params: dict, id: str, mask: Union[np.array, None] = None, factor: float = 1
) -> dict:
    if mask is None:
        mask = np.ones(params[id].shape)

    result = deepcopy(params)
    result[id].values[mask] = factor * result[id].values[mask]
    return result
