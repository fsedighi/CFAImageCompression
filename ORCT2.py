import numpy as np

A = np.array([1 / 2, 1 / 2], [-1, 1])


def compute_orct2(bayer):
    bayer_number_of_rows = bayer.shape[0]
    bayer_number_of_columns = bayer.shape[1]
    final_block = np.zeros((bayer_number_of_rows, bayer_number_of_columns))
    row_index = 0
    column_index = 0
    while column_index < bayer_number_of_columns:
        while row_index < bayer_number_of_rows:
            wr = bayer[row_index][column_index]
            wb = bayer[row_index + 1][column_index]
            y = A[1][0] * wr + A[1][1] * wb
            dw = A[0][0] * wr + A[0][1] * wb
            final_block[row_index][column_index] = y
            final_block[row_index + 1][column_index] = dw
            row_index += 2
        column_index += 2
    return final_block

