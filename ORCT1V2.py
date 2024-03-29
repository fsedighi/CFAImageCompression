import numpy as np

global A
A = np.array([[1 / 2, 1 / 2], [-1, 1]])


def compute_orct1V2(bayer, Alocal=None, precisionFloatingPoint=0):
    if Alocal is not None:
        global A
        A = Alocal
    bayer_number_of_rows = bayer.shape[0]
    bayer_number_of_columns = bayer.shape[1]
    final_block = np.zeros((bayer_number_of_rows, bayer_number_of_columns))

    for row_index in range(0, bayer_number_of_rows - 1, 2):
        for column_index in range(0, bayer_number_of_columns - 1, 2):
            gr = bayer[row_index][column_index]
            r = bayer[row_index][column_index + 1]

            b = bayer[row_index + 1][column_index]
            gb = bayer[row_index + 1][column_index + 1]
            converted_wr_dr = A @ np.asarray([gr, r]).transpose()
            converted_wb_db = A @ np.asarray([gb, b]).transpose()

            final_block[row_index][column_index + 1] = converted_wr_dr[1]
            final_block[row_index][column_index] = np.floor(converted_wr_dr[0])

            final_block[row_index + 1][column_index + 1] = converted_wb_db[1]
            final_block[row_index + 1][column_index] = np.floor(converted_wb_db[0])
    return final_block
