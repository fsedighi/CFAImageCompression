import unittest

import numpy as np
from ORCT1 import compute_orct1
import cv2


class TestORCT(unittest.TestCase):

    def test_orct1(self):
        bayer = cv2.imread("../Data/image.bmp")
        bayer = np.sum(bayer, axis=2).astype('float64')
        compute_orct1(bayer)
