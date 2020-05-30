import numpy as np

A = np.array([[1 / 2, 1 / 2], [-1, 1]])
# Ar = np.array([[1 / 2, 1 / 2], [-1, 1]])
# Ab = np.array([[1 / 2, 1 / 2], [-1, 1]])


def compute_orct1(bayer):
    bayer_number_of_rows = bayer.shape[0]
    bayer_number_of_columns = bayer.shape[1]
    final_block = np.zeros((bayer_number_of_rows, bayer_number_of_columns))
    row_index = 0
    column_index = 0
    while row_index <= bayer_number_of_rows:
        while column_index < bayer_number_of_columns:
            rb = bayer[row_index][column_index]
            g = bayer[row_index][column_index + 1]
            drb = A[1][0] * g + A[1][1] * rb
            wrb = A[0][0] * g + A[0][1] * rb
            final_block[row_index][column_index] = wrb
            final_block[row_index][column_index + 1] = drb
            column_index += 2
        row_index += 1
    return final_block
