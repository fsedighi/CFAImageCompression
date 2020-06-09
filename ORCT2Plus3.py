import numpy as np

A = np.array([[1 / 2, 1 / 2], [-1, 1]])

def compute_orct2plus3(bayer):
    bayer_number_of_rows = bayer.shape[0]
    bayer_number_of_columns = bayer.shape[1]
    final_block = bayer
    row_index = 0
    column_index = 0
    # while column_index < bayer_number_of_columns:
    #     while row_index < bayer_number_of_rows - 1:
    #         w1 = bayer[row_index][column_index]
    #         w2 = bayer[row_index + 1][column_index]
    #         d = w1 - w2
    #         w = w2 + np.floor(d/2)
    #         final_block[row_index][column_index] = d
    #         final_block[row_index + 1][column_index] = w
    #         row_index += 1
    #     row_index = 0
    #     column_index += 2
    for column_index in range(0, bayer_number_of_columns, 2):
        for row_index in range(bayer_number_of_rows - 1):
            w1 = bayer[row_index][column_index]
            w2 = bayer[row_index + 1][column_index]
            converted_w1_w2 = A @ np.array([w2, w1]).transpose()
            final_block[row_index][column_index] = converted_w1_w2[1]
            final_block[row_index + 1][column_index] = converted_w1_w2[0]

    return final_block

