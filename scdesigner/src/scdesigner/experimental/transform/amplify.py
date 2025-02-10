import numpy as np
from typing import Union


def amplify(
    params: dict, id: str, factor: float = 1, mask: Union[np.array, None] = None
) -> dict:
    if mask is None:
        mask = np.ones(params[id].shape)

    params[id][mask == 1] *= float
    return params
