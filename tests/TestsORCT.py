import unittest

import numpy as np
from ORCT1 import compute_orct1
import cv2

from ORCT1Inverse import compute_orct1inverse
from ORCT2 import compute_orct2
from ORCT2Inverse import compute_orct2inverse
from ORCT2Plus3 import compute_orct2plus3
from ORCT2Plus3Inverse import compute_orct2plus3inverse
import pandas as pd
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

    def test_orct123Plus(self):
        bayer = self.datasetUtils.readCFAImages()

        twoComplement = self.datasetUtils.twoComplementMatrix(bayer)
        twoComplement = twoComplement.astype("float32")

        filtered = compute_orct2plus3(compute_orct1(twoComplement))

        filtered = np.floor((filtered + 255) / 2)

        def inverseFunction(data):
            data = data.astype('float32') * 2 - 255
            data = compute_orct2plus3inverse(data)
            data = compute_orct1inverse(data)
            return np.round(data)

        sampleFunctionReverse = inverseFunction
        self.evaluation.evaluate(filtered, twoComplement, sampleFunctionReverse)

        pass

    def test_orct12(self):
        bayer = self.datasetUtils.readCFAImages()

        twoComplement = self.datasetUtils.twoComplementMatrix(bayer)
        twoComplement = twoComplement.astype("float32")

        filtered = compute_orct2(compute_orct1(twoComplement))

        filtered = np.round((filtered + 255) / 2)

        def inverseFunction(data):
            data = data.astype('float32') * 2 - 255
            data = compute_orct2inverse(data)
            data = compute_orct1inverse(data)
            return data

        sampleFunctionReverse = inverseFunction
        self.evaluation.evaluate(filtered, twoComplement, sampleFunctionReverse)
        pass

    def test_ocrtShahedWithDataset(self):
        rgbImages = self.datasetUtils.loadNikonDataset("D40")
        cfaImages, image_size = self.datasetUtils.convertDatasetToCFA(rgbImages)
        psnrs = []
        ssims = []
        jpeg2000CompressionRatioAfters = []

        def inverseFunction(data):
            data = data.astype('float32') * 2 - 255
            data = compute_orct2inverse(data)
            data = compute_orct1inverse(data)
            return data

        sampleFunctionReverse = inverseFunction

        for bayer in cfaImages:
            twoComplement = self.datasetUtils.twoComplementMatrix(bayer)
            twoComplement = twoComplement.astype("float32")

            filtered = compute_orct2(compute_orct1(twoComplement))

            filtered = np.round((filtered + 255) / 2)

            psnr, ssim, jpeg2000CompressionRatioAfter, jpeg2000CompressionRatioBefore = self.evaluation.evaluate(filtered, bayer, sampleFunctionReverse)
            psnrs.append(psnr)
            ssims.append(ssim)
            jpeg2000CompressionRatioAfters.append(jpeg2000CompressionRatioAfter)
        pd.DataFrame({"psnr": psnrs, "ssim": ssims, "jpeg200CompressionRatio": jpeg2000CompressionRatioAfters}).to_excel("resultsShahedMethod.xlsx")

    def test_ocrtNewMethodWithDataset(self):
        rgbImages = self.datasetUtils.loadKodakDataset()
        cfaImages, image_size = self.datasetUtils.convertDatasetToCFA(rgbImages)
        psnrs = []
        ssims = []
        jpeg2000CompressionRatioAfters = []

        def inverseFunction(data):
            data = data.astype('float32') * 2 - 255
            data = compute_orct2plus3inverse(data)
            data = compute_orct1inverse(data)
            return data

        sampleFunctionReverse = inverseFunction

        for bayer in cfaImages:
            twoComplement = self.datasetUtils.twoComplementMatrix(bayer)
            twoComplement = twoComplement.astype("float32")

            filtered = compute_orct2plus3(compute_orct1(twoComplement))

            filtered = (filtered + 255) / 2

            psnr, ssim, jpeg2000CompressionRatioAfter, jpeg2000CompressionRatioBefore = self.evaluation.evaluate(filtered, bayer, sampleFunctionReverse)
            psnrs.append(psnr)
            ssims.append(ssim)
            jpeg2000CompressionRatioAfters.append(jpeg2000CompressionRatioAfter)
        pd.DataFrame({"psnr": psnrs, "ssim": ssims, "jpeg200CompressionRatio": jpeg2000CompressionRatioAfters}).to_excel("resultsNewMethod.xlsx")

    def test_simpleORCT(self):
        bayer = np.array([[145, 77, 142, 73], [76, 67, 72, 62], [127, 67, 125, 65], [65, 54, 65, 57],
                          [145, 75, 142, 73], [46, 61, 72, 62], [117, 47, 105, 65], [87, 31, 53, 17]])
        bayer = bayer.astype("float32")
        data = compute_orct2plus3(bayer)
        data2 = compute_orct2plus3inverse(data)
        pass
