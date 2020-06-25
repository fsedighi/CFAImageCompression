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
        self.precisionFloatingPoint = 0

    def test_orct1(self):
        bayer = cv2.imread("../Data/image.bmp")
        bayer = np.sum(bayer, axis=2).astype('float64')
        orct1Filtered = compute_orct1(bayer, precisionFloatingPoint=self.precisionFloatingPoint)
        pass

    def test_orct2(self):
        bayer = cv2.imread("../Data/image.bmp")
        bayer = np.sum(bayer, axis=2).astype('float64')
        orct2Filtered = compute_orct2(bayer, precisionFloatingPoint=self.precisionFloatingPoint)
        pass

    def test_orct12(self):
        bayer = self.datasetUtils.readCFAImages()

        bayer = bayer.astype("float64")

        orct1Res = compute_orct1(bayer, precisionFloatingPoint=self.precisionFloatingPoint)
        filtered = compute_orct2(orct1Res, precisionFloatingPoint=self.precisionFloatingPoint)
        # filtered = (filtered)/2

        filtered = (filtered + 256) / 2

        def inverseFunction(data):
            data = data.astype('float32') * 2 - 256
            data = compute_orct2inverse(data, precisionFloatingPoint=self.precisionFloatingPoint)
            data = compute_orct1inverse(data, precisionFloatingPoint=self.precisionFloatingPoint)
            return data

        sampleFunctionReverse = inverseFunction
        self.evaluation.evaluate(filtered, bayer, sampleFunctionReverse, precisionFloatingPoint=self.precisionFloatingPoint)
        pass

    def test_orct123Plus(self):
        bayer = self.datasetUtils.readCFAImages()

        bayer = bayer.astype("float32")

        filtered = compute_orct2plus3(compute_orct1(bayer, precisionFloatingPoint=self.precisionFloatingPoint), precisionFloatingPoint=self.precisionFloatingPoint)

        x = np.ones(filtered.shape)
        x[1::2, ::2] = -1
        x[::2, 1::2] = -1
        filtered = np.multiply(x, filtered)
        mask = np.multiply(filtered > -3, filtered < 3)
        filtered[mask] = np.abs(filtered[mask])
        negativemask = filtered < 0

        # filtered = (filtered + 128)

        def inverseFunction(data):
            data = np.multiply(x, data).astype('float32')
            mask = np.multiply(data > -3, data < 3)
            data[mask] = np.abs(data[mask])
            data[negativemask] = -np.abs(data[negativemask])
            # data = data.astype('float32') - 128
            data = compute_orct2plus3inverse(data, precisionFloatingPoint=self.precisionFloatingPoint)
            data = compute_orct1inverse(data, precisionFloatingPoint=self.precisionFloatingPoint)
            return data

        sampleFunctionReverse = inverseFunction
        self.evaluation.evaluate(filtered, bayer, sampleFunctionReverse, precisionFloatingPoint=self.precisionFloatingPoint)

        pass

    def test_ocrtShahedWithDataset(self):
        rgbImages = self.datasetUtils.loadKodakDataset()
        cfaImages, image_size = self.datasetUtils.convertDatasetToCFA(rgbImages)
        psnrs = []
        ssims = []
        jpeg2000CompressionRatioAfters = []
        JpegLsCompressionRatios = []
        compressionRatioLZWs = []
        compressionRatiojpeg2000LossyAfters = []
        # filtered = (filtered + 128)

        def inverseFunction(data):
            data = data.astype('float32') * 2 - 256
            data = compute_orct2inverse(data, precisionFloatingPoint=self.precisionFloatingPoint)
            data = compute_orct1inverse(data, precisionFloatingPoint=self.precisionFloatingPoint)
            return data

        sampleFunctionReverse = inverseFunction

        for bayer in cfaImages:
            bayer = bayer.astype("float32")

            filtered = compute_orct2(compute_orct1(bayer, precisionFloatingPoint=self.precisionFloatingPoint), precisionFloatingPoint=self.precisionFloatingPoint)
            filtered = (filtered + 256) / 2

            psnr, ssim, jpeg2000CompressionRatioAfter, JpegLsCompressionRatio, compressionRatioLZWAfter, compressionRatiojpeg2000LossyAfter = self.evaluation.evaluate(filtered, bayer,
                                                                                                                                                                       sampleFunctionReverse,
                                                                                                                                                                       precisionFloatingPoint=self.precisionFloatingPoint)

            psnrs.append(psnr)
            ssims.append(ssim)
            jpeg2000CompressionRatioAfters.append(jpeg2000CompressionRatioAfter)
            JpegLsCompressionRatios.append(JpegLsCompressionRatio)
            compressionRatioLZWs.append(compressionRatioLZWAfter)
            compressionRatiojpeg2000LossyAfters.append(compressionRatiojpeg2000LossyAfter)
        pd.DataFrame({"psnr": psnrs, "ssim": ssims, "jpeg200CompressionRatio (bpp)": jpeg2000CompressionRatioAfters,
                      "JpegLsCompressionRatio": JpegLsCompressionRatios,
                      "compressionRatioLZW": compressionRatioLZWs,
                      "compressionRatiojpeg2000Lossy": compressionRatiojpeg2000LossyAfters}).to_excel("resultsShahedMethod.xlsx")

    def test_ocrtNewMethodWithDataset(self):
        rgbImages = self.datasetUtils.loadKodakDataset()
        cfaImages, image_size = self.datasetUtils.convertDatasetToCFA(rgbImages)
        psnrs = []
        ssims = []
        jpeg2000CompressionRatioAfters = []
        JpegLsCompressionRatios = []
        compressionRatioLZWs = []
        compressionRatiojpeg2000LossyAfters = []

        def inverseFunction(data):
            data = data.astype('float32') * 2 - 255
            data = compute_orct2plus3inverse(data, precisionFloatingPoint=self.precisionFloatingPoint)
            data = compute_orct1inverse(data, precisionFloatingPoint=self.precisionFloatingPoint)
            return data

        sampleFunctionReverse = inverseFunction

        for bayer in cfaImages:
            bayer = bayer.astype("float32")

            filtered = compute_orct2plus3(compute_orct1(bayer, precisionFloatingPoint=self.precisionFloatingPoint), precisionFloatingPoint=self.precisionFloatingPoint)

            filtered = np.round((filtered + 255) / 2, self.precisionFloatingPoint)

            psnr, ssim, jpeg2000CompressionRatioAfter, JpegLsCompressionRatio, compressionRatioLZWAfter, compressionRatiojpeg2000LossyAfter = self.evaluation.evaluate(filtered, bayer,
                                                                                                                                                                       sampleFunctionReverse,
                                                                                                                                                                       precisionFloatingPoint=self.precisionFloatingPoint)
            psnrs.append(psnr)
            ssims.append(ssim)
            jpeg2000CompressionRatioAfters.append(jpeg2000CompressionRatioAfter)
            JpegLsCompressionRatios.append(JpegLsCompressionRatio)
            compressionRatioLZWs.append(compressionRatioLZWAfter)
            compressionRatiojpeg2000LossyAfters.append(compressionRatiojpeg2000LossyAfter)
        pd.DataFrame({"psnr": psnrs, "ssim": ssims, "jpeg200CompressionRatio (bpp)": jpeg2000CompressionRatioAfters,
                      "JpegLsCompressionRatio": JpegLsCompressionRatios,
                      "compressionRatioLZW": compressionRatioLZWs,
                      "compressionRatiojpeg2000Lossy":compressionRatiojpeg2000LossyAfters}).to_excel("resultsNewMethod.xlsx")

    def test_simpleORCT(self):
        bayer = np.array([[145, 77, 142, 73], [76, 67, 72, 62], [127, 67, 125, 65], [65, 54, 65, 57],
                          [145, 75, 142, 73], [46, 61, 72, 62], [117, 47, 105, 65], [87, 31, 53, 17]])
        bayer = bayer.astype("float32")
        data = compute_orct2plus3(bayer)
        data2 = compute_orct2plus3inverse(data)
        pass
