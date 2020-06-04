import numpy as np

A = np.array([[1 / 2, 1 / 2], [-1, 1]])


def compute_orct2(bayer):
    bayer_number_of_rows = bayer.shape[0]
    bayer_number_of_columns = bayer.shape[1]
    final_block = np.zeros((bayer_number_of_rows, bayer_number_of_columns))

    for column_index in range(0, bayer_number_of_columns, 2):
        for row_index in range(0, bayer_number_of_rows - 1, 2):
            wr = bayer[row_index][column_index]
            wb = bayer[row_index + 1][column_index]
            # dw = A[1][0] * wr + A[1][1] * wb
            # y = A[0][0] * wr + A[0][1] * wb
            dw = wr - wb
            y = wb + np.floor(dw / 2)
            final_block[row_index][column_index] = y
            final_block[row_index + 1][column_index] = dw

    return final_block
