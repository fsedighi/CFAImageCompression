import copy

import numpy as np

global A
A = np.array([[1 / 2, 1 / 2], [-1, 1]])


def compute_orct1inverse(bayer, Alocal=None, precisionFloatingPoint=0):
    if Alocal is not None:
        global A
        A = Alocal
    bayer_number_of_rows = bayer.shape[0]
    bayer_number_of_columns = bayer.shape[1]
    final_block = np.zeros((bayer_number_of_rows, bayer_number_of_columns))
    # final_block = copy.deepcopy(bayer)
    inverseA = np.linalg.pinv(A)
    for row_index in range(0, bayer_number_of_rows, 2):
        for column_index in range(0, bayer_number_of_columns - 1, 2):
            wr = bayer[row_index][column_index]
            dr = bayer[row_index][column_index + 1]
            index_r = dr % 2

            wb = bayer[row_index + 1][column_index]
            db = bayer[row_index + 1][column_index + 1]
            index_b = db % 2

            if index_r == 1:
                wr += 0.5
            if index_b == 1:
                wb += 0.5

            converted_gr_r = inverseA @ np.asarray([wr, dr]).transpose()
            converted_gb_b = inverseA @ np.asarray([wb, db]).transpose()

            final_block[row_index][column_index + 1] = np.round(converted_gr_r[1], precisionFloatingPoint)
            final_block[row_index][column_index] = np.round(converted_gr_r[0], precisionFloatingPoint)

            final_block[row_index + 1][column_index] = np.round(converted_gb_b[1], precisionFloatingPoint)
            final_block[row_index + 1][column_index + 1] = np.round(converted_gb_b[0], precisionFloatingPoint)

    return final_block
