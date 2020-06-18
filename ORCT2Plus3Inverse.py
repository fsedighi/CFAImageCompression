import copy

import numpy as np

global A
A = np.array([[1 / 2, 1 / 2], [-1, 1]])


def compute_orct2plus3inverse(bayer, Alocal=None, precisionFloatingPoint=0):
    if Alocal is not None:
        global A
        A = Alocal
    bayer_number_of_rows = bayer.shape[0]
    bayer_number_of_columns = bayer.shape[1]
    final_block = copy.deepcopy(bayer)
    inverseA = np.linalg.pinv(A)
    for column_index in range(0, bayer_number_of_columns, 2):
        for row_index in range(bayer_number_of_rows - 1, 0, -1):

            w1 = final_block[row_index][column_index]
            w2 = final_block[row_index - 1][column_index]
            converted_w1_w2 = inverseA @ np.array([w1, w2]).transpose()

            final_block[row_index][column_index] = np.round(converted_w1_w2[0], precisionFloatingPoint)
            final_block[row_index - 1][column_index] = np.round(converted_w1_w2[1], precisionFloatingPoint)

    return final_block
