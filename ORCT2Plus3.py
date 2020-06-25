import copy

import numpy as np

global A
A = np.array([[1 / 2, 1 / 2], [-1, 1]])


def compute_orct2plus3(bayer, Alocal=None, precisionFloatingPoint=0):
    if Alocal is not None:
        global A
        A = Alocal
    bayer_number_of_rows = bayer.shape[0]
    bayer_number_of_columns = bayer.shape[1]
    final_block = copy.deepcopy(bayer)

    for column_index in range(0, bayer_number_of_columns, 2):
        for row_index in range(bayer_number_of_rows - 1):
            index = bayer_number_of_rows % 2
            w1 = final_block[row_index][column_index]
            w2 = final_block[row_index + 1][column_index]
            converted_w1_w2 = A @ np.array([w2, w1]).transpose()
            if index == 0:
                final_block[row_index][column_index] = np.floor(converted_w1_w2[1])
                final_block[row_index + 1][column_index] = np.floor(converted_w1_w2[0])
            elif index == 1:
                final_block[row_index][column_index] = np.ceil(converted_w1_w2[1])
                final_block[row_index + 1][column_index] = np.ceil(converted_w1_w2[0])

    return final_block
