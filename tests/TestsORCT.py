import unittest

import numpy as np
from ORCT1 import compute_orct1
import cv2

from ORCT2 import compute_orct2
from ORCT3 import compute_orct3
from Utils.CompressionEvaluation import CompressionEvaluation
from Utils.DataUtils import DataUtils


class TestORCT(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.datasetUtils = DataUtils()
        self.compressionEvaluation = CompressionEvaluation()

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
        bayer = self.datasetUtils.readCFAImages()

        twoComplement = self.datasetUtils.twoComplementMatrix(bayer)
        twoComplement = twoComplement.astype("float32")

        filtered = compute_orct3(compute_orct2(compute_orct1(twoComplement)))

        filtered = (filtered + 255) / 2

        self.compressionEvaluation.evaluate(bayer, "before ocrt")
        self.compressionEvaluation.evaluate(filtered, "after ocrt")
        pass

    def test_orct12(self):
        bayer=self.datasetUtils.readCFAImages()

        twoComplement = self.datasetUtils.twoComplementMatrix(bayer)
        twoComplement = twoComplement.astype("float32")

        filtered = compute_orct2(compute_orct1(twoComplement))

        filtered = (filtered + 255) / 2

        self.compressionEvaluation.evaluate(bayer, "before ocrt")
        self.compressionEvaluation.evaluate(filtered, "after ocrt")
        pass

    def test_ocrtWithDataset(self):
        rgbImages = self.datasetUtils.loadKodakDataset()
        cfaImages, image_size = self.datasetUtils.convertDatasetToCFA(rgbImages)
        bayer = cfaImages[0, :, :]

        twoComplement = self.datasetUtils.twoComplementMatrix(bayer)
        twoComplement = twoComplement.astype("float32")

        filtered = compute_orct2(compute_orct1(twoComplement))

        filtered = (filtered + 255) / 2

        self.compressionEvaluation.evaluate(bayer, "before ocrt")
        self.compressionEvaluation.evaluate(filtered, "after ocrt")
        pass
