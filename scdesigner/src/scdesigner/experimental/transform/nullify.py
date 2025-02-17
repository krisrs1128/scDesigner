from typing import Union
from copy import deepcopy
import numpy as np
from . import docstrings as ds


@ds.doc(ds.nullify)
def nullify(params: dict, id: str, mask: Union[np.array, None] = None) -> dict:
    if mask is None:
        mask = np.ones(params[id].shape)

    result = deepcopy(params)
    result[id][mask] = 0
    return result
