import unittest

import numpy as np
from ORCT1 import compute_orct1
import cv2

from ORCT1Inverse import compute_orct1inverse
from ORCT2 import compute_orct2
from ORCT2Inverse import compute_orct2inverse
from ORCT2Plus3 import compute_orct2plus3
from ORCT3 import compute_orct3
from Utils.Evaluation import Evaluation
from Utils.DataUtils import DataUtils


class TestORCT(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.datasetUtils = DataUtils()
        self.evaluation = Evaluation()

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

        self.evaluation.evaluate(filtered, bayer)
        pass

    def test_orct123Plus(self):
        bayer = self.datasetUtils.readCFAImages()

        twoComplement = self.datasetUtils.twoComplementMatrix(bayer)
        twoComplement = twoComplement.astype("float32")

        filtered = compute_orct2plus3(compute_orct1(twoComplement))

        filtered = (filtered + 255) / 2

        self.evaluation.evaluate(filtered, bayer)

        pass

    def test_orct12(self):
        bayer = self.datasetUtils.readCFAImages()

        twoComplement = self.datasetUtils.twoComplementMatrix(bayer)
        twoComplement = twoComplement.astype("float32")

        filtered = compute_orct2(compute_orct1(twoComplement))

        filtered = (filtered + 255) / 2

        def inverseFunction(data):
            data = data * 2 - 255
            data = compute_orct2inverse(data)
            data = compute_orct1inverse(data)
            return data

        sampleFunctionReverse = inverseFunction
        self.evaluation.evaluate(filtered, bayer, sampleFunctionReverse)
        pass

    def test_ocrtWithDataset(self):
        rgbImages = self.datasetUtils.loadKodakDataset()
        cfaImages, image_size = self.datasetUtils.convertDatasetToCFA(rgbImages)
        bayer = cfaImages[2, :, :]

        twoComplement = self.datasetUtils.twoComplementMatrix(bayer)
        twoComplement = twoComplement.astype("float32")

        filtered = compute_orct2plus3(compute_orct2(compute_orct1(twoComplement)))

        filtered = (filtered + 255) / 2

        self.evaluation.evaluate(filtered, bayer)
        pass
