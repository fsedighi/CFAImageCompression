import copy
import copy
import numpy as np

global A
A = np.array([[1 / 2, 1 / 2], [-1, 1]])


def compute_orct2inverse(bayer, Alocal=None, precisionFloatingPoint=0):
    if Alocal is not None:
        global A
        A = Alocal
    bayer_number_of_rows = bayer.shape[0]
    bayer_number_of_columns = bayer.shape[1]
    final_block = copy.deepcopy(bayer)
    # final_block = np.zeros((bayer_number_of_rows, bayer_number_of_columns))
    inverseA = np.linalg.pinv(A)
    for column_index in range(0, bayer_number_of_columns, 2):
        for row_index in range(0, bayer_number_of_rows, 2):
            y = bayer[row_index][column_index]
            dw = bayer[row_index + 1][column_index]
            index = dw % 2

            if index == 1:
                y += 0.5
            else:
                pass
            converted_wb_wr = inverseA @ np.asarray([y, dw]).transpose()

            final_block[row_index][column_index] = converted_wb_wr[1]
            final_block[row_index + 1][column_index] = np.ceil(converted_wb_wr[0])

    return final_block
