import copy
import unittest

import numpy as np
from ORCT1 import compute_orct1
import cv2

from ORCT1Inverse import compute_orct1inverse
from ORCT1InverseV2 import compute_orct1inverseV2
from ORCT1V2 import compute_orct1V2
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

    def test_orct23PlusReversible(self):
        bayer = cv2.imread("../Data/image.bmp")
        bayer = np.sum(bayer, axis=2).astype('float64')
        orct23Filtered = compute_orct2plus3(bayer, precisionFloatingPoint=self.precisionFloatingPoint)
        orct23FilteredInversed = compute_orct2plus3inverse(orct23Filtered, precisionFloatingPoint=self.precisionFloatingPoint)
        print("PSNR: {}".format(self.evaluation.calculate_psnr(bayer, np.round(orct23FilteredInversed))))

    def test_orct1Reversible(self):
        bayer = cv2.imread("../Data/image.bmp")
        bayer = np.sum(bayer, axis=2).astype('float64')
        orct1Filtered = compute_orct1V2(bayer, precisionFloatingPoint=self.precisionFloatingPoint)
        orct1FilteredInversed = compute_orct1inverseV2(orct1Filtered, precisionFloatingPoint=self.precisionFloatingPoint)
        print("PSNR: {}".format(self.evaluation.calculate_psnr(bayer, np.round(orct1FilteredInversed))))

    def test_orct123Reversible(self):
        bayer = cv2.imread("../Data/image.bmp")
        bayer = np.sum(bayer, axis=2).astype('float64')
        orct1Filtered = compute_orct1V2(bayer, precisionFloatingPoint=self.precisionFloatingPoint)
        orct23Filtered = compute_orct2plus3(orct1Filtered, precisionFloatingPoint=self.precisionFloatingPoint)
        orct23FilteredNormalized = copy.deepcopy(orct23Filtered)
        orct23FilteredNormalized[orct23FilteredNormalized == 0] = -256
        orct23FilteredNormalized = (orct23FilteredNormalized + 256) / 2
        orct23FilteredNormalized = np.ceil(orct23FilteredNormalized)
        orct23FilteredNormalized = orct23FilteredNormalized * 2 - 256
        orct23FilteredNormalized[orct23FilteredNormalized == -256] = 0
        orct23Filtered = orct23FilteredNormalized
        orct23FilteredInversed = compute_orct2plus3inverse(orct23Filtered, precisionFloatingPoint=self.precisionFloatingPoint)
        orct1FilteredInversed = compute_orct1inverseV2(orct23FilteredInversed, precisionFloatingPoint=self.precisionFloatingPoint)

        print("PSNR: {}".format(self.evaluation.calculate_psnr(np.round(orct23FilteredInversed), np.round(orct1Filtered))))
        print("PSNR: {}".format(self.evaluation.calculate_psnr(bayer, np.round(orct1FilteredInversed))))

    def test_orct12(self):
        bayer = self.datasetUtils.readCFAImages()

        bayer = bayer.astype("float64")

        orct1Res = compute_orct1(bayer, precisionFloatingPoint=self.precisionFloatingPoint)
        filtered = compute_orct2(orct1Res, precisionFloatingPoint=self.precisionFloatingPoint)
        # filtered = (filtered)/2

        filtered[filtered == 0] = -256
        filtered = (filtered + 256) / 2

        def inverseFunction(data):
            data = data.astype('float32')
            data = data * 2 - 256
            data[data == -256] = 0
            data = compute_orct2inverse(data, precisionFloatingPoint=self.precisionFloatingPoint)
            data = compute_orct1inverse(data, precisionFloatingPoint=self.precisionFloatingPoint)
            return data

        sampleFunctionReverse = inverseFunction
        self.evaluation.evaluate(filtered, bayer, sampleFunctionReverse, precisionFloatingPoint=self.precisionFloatingPoint)
        pass

    def test_orct123Plus(self):
        bayer = self.datasetUtils.readCFAImages()

        bayer = bayer.astype("float32")

        orct_1 = compute_orct1V2(bayer, precisionFloatingPoint=self.precisionFloatingPoint)
        filtered = compute_orct2plus3(orct_1, precisionFloatingPoint=self.precisionFloatingPoint)

        filtered[filtered == 0] = -256
        filtered = (filtered + 256) / 2
        filtered = np.ceil(filtered)

        def inverseFunction(data):
            data = data.astype('float32')
            data = data * 2 - 256
            data[data == -256] = 0
            data = compute_orct2plus3inverse(data, precisionFloatingPoint=self.precisionFloatingPoint)
            data = compute_orct1inverseV2(data, precisionFloatingPoint=self.precisionFloatingPoint)
            return np.round(data)

        sampleFunctionReverse = inverseFunction
        self.evaluation.evaluate(filtered, bayer, sampleFunctionReverse, precisionFloatingPoint=self.precisionFloatingPoint, roundingMethod="ceil")

        pass

    def test_ocrtShahedWithDataset(self):
        rgbImages = self.datasetUtils.loadArri()
        cfaImages, image_size = self.datasetUtils.convertDatasetToCFA(rgbImages)
        psnrs = []
        ssims = []
        jpeg2000CompressionRatioAfters = []
        JpegLsCompressionRatios = []
        compressionRatioLZWs = []
        compressionRatiojpeg2000LossyAfters = []

        # filtered = (filtered + 128)
        # filtered[:, 1::2] = filtered[:, 1::2] / 2 + 128

        def inverseFunction(data):
            data = data.astype('float32') * 2 - 256

            data = compute_orct1inverse(data, precisionFloatingPoint=self.precisionFloatingPoint)
            return data

        sampleFunctionReverse = inverseFunction

        for bayer in cfaImages:
            bayer = bayer.astype("float32")

            filtered = compute_orct2(compute_orct1(bayer, precisionFloatingPoint=self.precisionFloatingPoint), precisionFloatingPoint=self.precisionFloatingPoint)
            test_sample = filtered
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
        psnrs = []
        ssims = []
        jpeg2000CompressionRatioAfters = []
        JpegLsCompressionRatiosAfters = []
        compressionRatioLZWsAfters = []
        jpeg2000CompressionRatioBefores = []
        JpegLsCompressionRatiosBefores = []
        compressionRatioLZWsBefores = []
        datasetName = []
        nameOfdatasets = ["Akademie", "Arri exterior", "Color test chart", "Face", "Kodak", "Lake locked", "Lake pan", "Night Odeplatz", "Nikon D40", "Nikon D90", "Nikon D7000", "Siegestor",
                          "Pool interior"]
        for nameOfdataset in nameOfdatasets:
            print(nameOfdataset)
            if nameOfdataset in ["Kodak", "Nikon D90", "Nikon D7000", "Nikon D40"]:
                self.precisionFloatingPoint = 0
                bias = 256
            else:
                self.precisionFloatingPoint = 4
                bias = 65536
            rgbImages = self.datasetUtils.loadOtherDataset(nameOfdataset)
            cfaImages, image_size = self.datasetUtils.convertDatasetToCFA(rgbImages)

            def inverseFunction(data):
                data = data.astype('float32')
                data = data * 2 - bias
                data = compute_orct2plus3inverse(data, precisionFloatingPoint=self.precisionFloatingPoint)
                data = compute_orct1inverseV2(data, precisionFloatingPoint=self.precisionFloatingPoint)
                return np.round(data)

            sampleFunctionReverse = inverseFunction

            for bayer in cfaImages:
                bayer = bayer.astype("float32")

                filtered = compute_orct2plus3(compute_orct1(bayer, precisionFloatingPoint=self.precisionFloatingPoint), precisionFloatingPoint=self.precisionFloatingPoint)
                test_sample = filtered
                filtered = (filtered + bias) / 2
                filtered = np.ceil(filtered-np.min(filtered))

                psnr, ssim, jpeg2000CompressionRatioAfter, JpegLsCompressionRatioAfter, compressionRatioLZWAfter, jpeg2000CompressionRatioBefore, JpegLsCompressionRatioBefore, compressionRatioLZWBefore = self.evaluation.evaluate(
                    filtered, bayer,
                    sampleFunctionReverse,
                    precisionFloatingPoint=self.precisionFloatingPoint,
                    roundingMethod="ceil")
                datasetName.append(nameOfdataset)
                psnrs.append(psnr)
                ssims.append(ssim)
                jpeg2000CompressionRatioAfters.append(jpeg2000CompressionRatioAfter)
                JpegLsCompressionRatiosAfters.append(JpegLsCompressionRatioAfter)
                compressionRatioLZWsAfters.append(compressionRatioLZWAfter)
                jpeg2000CompressionRatioBefores.append(jpeg2000CompressionRatioBefore)
                JpegLsCompressionRatiosBefores.append(JpegLsCompressionRatioBefore)
                compressionRatioLZWsBefores.append(compressionRatioLZWBefore)

        pd.DataFrame({"Image set name": datasetName,
                      "psnr": psnrs,
                      "ssim": ssims,
                      "jpeg200-LS After": jpeg2000CompressionRatioAfters,
                      "jpeg-Ls After": JpegLsCompressionRatiosAfters,
                      "LZW After": compressionRatioLZWsAfters,
                      "jpeg200-LS Before": jpeg2000CompressionRatioBefores,
                      "jpeg-Ls Before": JpegLsCompressionRatiosBefores,
                      "LZW Before": compressionRatioLZWsBefores}).to_excel("resultsAll2.xlsx")

    def test_simpleORCT(self):
        bayer = np.array([[145, 77, 142, 73], [76, 67, 72, 62], [127, 67, 125, 65], [65, 54, 65, 57],
                          [145, 75, 142, 73], [46, 61, 72, 62], [117, 47, 105, 65], [87, 31, 53, 17]])
        bayer = bayer.astype("float32")
        data = compute_orct2plus3(bayer)
        data2 = compute_orct2plus3inverse(data)
        pass
