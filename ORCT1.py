import numpy as np

A = np.array([[1 / 2, 1 / 2], [-1, 1]])


def compute_orct1(bayer):
    bayer_number_of_rows = bayer.shape[0]
    bayer_number_of_columns = bayer.shape[1]
    final_block = np.zeros((bayer_number_of_rows, bayer_number_of_columns))

    for row_index in range(0, bayer_number_of_rows, 2):
        for column_index in range(0, bayer_number_of_columns, 2):
            gr = bayer[row_index][column_index]
            r = bayer[row_index][column_index + 1]

            b = bayer[row_index + 1][column_index]
            gb = bayer[row_index + 1][column_index + 1]
            converted_dr_wr = A @ np.asarray([gr, r])
            converted_wb_db = A @ np.asarray([gb, b])

            final_block[row_index][column_index + 1] = converted_dr_wr[0]
            final_block[row_index][column_index] = converted_dr_wr[1]

            final_block[row_index + 1][column_index + 1] = converted_wb_db[0]
            final_block[row_index + 1][column_index] = converted_wb_db[1]

    return final_block
