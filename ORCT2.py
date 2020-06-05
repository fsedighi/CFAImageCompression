import copy
import copy
import numpy as np

A = np.array([[1 / 2, 1 / 2], [-1, 1]])


def compute_orct2(bayer):
    bayer_number_of_rows = bayer.shape[0]
    bayer_number_of_columns = bayer.shape[1]
    final_block = copy.deepcopy(bayer)

    for column_index in range(1, bayer_number_of_columns, 2):
        for row_index in range(0, bayer_number_of_rows, 2):
            wr = bayer[row_index][column_index]
            wb = bayer[row_index + 1][column_index]

            converted = A @ np.asarray([wr, wb])

            final_block[row_index][column_index] = converted[0]
            final_block[row_index + 1][column_index] = converted[1]

    return final_block
