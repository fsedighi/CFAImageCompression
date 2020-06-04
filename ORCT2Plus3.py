import numpy as np


def compute_orct2plus3(bayer):
    bayer_number_of_rows = bayer.shape[0]
    bayer_number_of_columns = bayer.shape[1]
    final_block = bayer
    row_index = 0
    column_index = 0
    while column_index < bayer_number_of_columns:
        while row_index < bayer_number_of_rows - 1:
            w1 = bayer[row_index][column_index]
            w2 = bayer[row_index + 1][column_index]
            d = w1 - w2
            w = w2 + np.floor(d/2)
            final_block[row_index][column_index] = d
            final_block[row_index + 1][column_index] = w
            row_index += 1
        row_index = 0
        column_index += 2
    return final_block

