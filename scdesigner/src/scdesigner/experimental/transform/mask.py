import numpy as np
import re


def str_match(string: str, string_list: list) -> bool:
    for l in string_list:
        if l in string:
            return True
    return False


def data_frame_mask(df, row_pattern=".", col_pattern=".") -> np.array:
    mask = np.zeros(df.shape, dtype=bool)
    if isinstance(col_pattern, str):
        col_pattern = [col_pattern]
    if isinstance(row_pattern, str):
        row_pattern = [row_pattern]

    # check for columns that match at least one pattern
    col_matches = np.zeros(df.shape[1], dtype=bool)
    for col in df.columns:
        if any(re.search(pat, col) for pat in col_pattern):
            col_matches[df.columns.get_loc(col)] = True

    # same with rows
    row_matches = np.zeros(df.shape[0], dtype=bool)
    for idx in df.index:
        if any(re.search(pat, str(idx)) for pat in row_pattern):
            row_matches[df.index.get_loc(idx)] = True

    # set mask to the intersection of row and column matches
    mask = np.outer(row_matches, col_matches)
    return mask
