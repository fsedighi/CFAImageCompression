import unittest

import numpy as np
from ORCT1 import compute_orct1
import cv2

from ORCT2 import compute_orct2
from ORCT3 import compute_orct3
from Utils.CompressionEvaluation import CompressionEvaluation


class TestORCT(unittest.TestCase):

    def test_orct1(self):
        bayer = cv2.imread("../Data/image.bmp")
        bayer = np.sum(bayer, axis=2).astype('float64')
        orct1Filtered = compute_orct1(bayer)
        pass

    def test_orct2(self):
        bayer = cv2.imread("../Data/image.bmp")
        bayer = np.sum(bayer, axis=2).astype('float64')
        orct2Filtered = compute_orct2(bayer)
        pass

    def test_orct3(self):
        bayer = cv2.imread("../Data/image.bmp")
        bayer = np.sum(bayer, axis=2).astype('float64')
        orct3Filtered = compute_orct3(bayer)
        pass

    def test_orct123(self):
        bayer = cv2.imread("../Data/image.bmp")
        bayer = np.sum(bayer, axis=2).astype('float64')
        filtered = compute_orct3(compute_orct2(compute_orct1(bayer)))
        compressionEvaluation = CompressionEvaluation()
        compressionEvaluation.evaluate(bayer, "before ocrt")
        compressionEvaluation.evaluate(filtered, "after ocrt")
        pass
