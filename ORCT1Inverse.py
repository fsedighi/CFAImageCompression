import numpy as np

A = np.array([[1 / 2, 1 / 2], [-1, 1]])


def compute_orct1inverse(bayer):
    bayer_number_of_rows = bayer.shape[0]
    bayer_number_of_columns = bayer.shape[1]
    final_block = np.zeros((bayer_number_of_rows, bayer_number_of_columns))

    for row_index in range(0, bayer_number_of_rows, 2):
        for column_index in range(0, bayer_number_of_columns, 2):
            wr = bayer[row_index][column_index]
            dr = bayer[row_index][column_index + 1]

            wb = bayer[row_index + 1][column_index]
            db = bayer[row_index + 1][column_index + 1]
            converted_gr_r = np.linalg.pinv(A) @ np.asarray([wr, dr]).transpose()
            converted_gb_b = A @ np.asarray([wb, db]).transpose()

            final_block[row_index][column_index + 1] = converted_gr_r[1]
            final_block[row_index+ 1][column_index] = converted_gr_r[0]

            final_block[row_index + 1][column_index + 1] = converted_gb_b[1]
            final_block[row_index][column_index] = converted_gb_b[0]

    return final_block
