import copy

import numpy as np

A = np.array([[1 / 2, 1 / 2], [-1, 1]])


def compute_orct3(bayer):
    bayer_number_of_rows = bayer.shape[0]
    bayer_number_of_columns = bayer.shape[1]
    final_block = copy.deepcopy(bayer)

    for row_index in range(0, bayer_number_of_rows, 2):
        for column_index in range(0, bayer_number_of_columns - 3, 2):
            y1 = bayer[row_index][column_index]
            y2 = bayer[row_index][column_index + 2]
            dy = y1 - y2
            wy = y2 + np.floor(dy / 2)
            final_block[row_index][column_index] = dy
            final_block[row_index][column_index + 2] = wy

    return final_block
