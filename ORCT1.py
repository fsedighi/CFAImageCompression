import numpy as np

A = np.array([[1 / 2, 1 / 2], [-1, 1]])


def compute_orct1(bayer):
    bayer_number_of_rows = bayer.shape[0]
    bayer_number_of_columns = bayer.shape[1]
    final_block = np.zeros((bayer_number_of_rows, bayer_number_of_columns))

    for row_index in range(bayer_number_of_rows):
        for column_index in range(0, bayer_number_of_columns - 1, 2):
            rb = bayer[row_index][column_index]
            g = bayer[row_index][column_index + 1]
            # drb = A[1][0] * g + A[1][1] * rb
            # wrb = A[0][0] * g + A[0][1] * rb
            drb = rb - g
            wrb = g + np.floor(drb / 2)
            final_block[row_index][column_index] = wrb
            final_block[row_index][column_index + 1] = drb

    return final_block
