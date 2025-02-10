import numpy as np
from typing import Union

def nullify(params: dict, id: str, mask: Union[np.array, None]=None) -> dict:
    if mask is None:
        mask = np.ones(params[id].shape)

    params[id][mask == 1] = 0
    return params