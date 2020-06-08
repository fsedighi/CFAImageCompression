import os

import numpy as np
import cv2

from Utils.CFAGeneratorUtils import RGB2CFAUtils


class DataUtils:

    def loadKodakDataset(self):
        # download from https://www.kaggle.com/sherylmehta/kodak-dataset
        root = "../Data/kodak/"
        files = os.listdir(root)
        rgbImages = []
        for file in files:
            img = cv2.imread(root + file)
            img = cv2.resize(img, (256, 256))
            rgbImages.append(img)

        rgbImages = np.asarray(rgbImages)
        return rgbImages

    def convertDatasetToCFA(self, rgbImages):
        # Convert to CFA
        rgb2CFAUtils = RGB2CFAUtils()
        n_data, h, w, c = rgbImages.shape

        cfaImages = []
        for i in range(n_data):
            cfaImages.append(rgb2CFAUtils.rgb2CFA(rgbImages[i], show=False))
            print("converting image {0} to CFA: Training".format(i))
        cfaImages = np.asarray(cfaImages)
        image_size = cfaImages.shape[1]
        cfaImages = np.reshape(cfaImages, [-1, image_size, image_size])
        cfaImages = cfaImages.astype('uint8')
        return cfaImages, image_size

    def twoComplementMatrix(self, data):
        comp2 = np.zeros(data.shape, dtype=int)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                number = data[i, j]
                binary_number = int("{0:08b}".format(number))
                flipped_binary_number = ~ binary_number
                flipped_binary_number = flipped_binary_number + 1
                str_twos_complement = str(flipped_binary_number)
                twos_complement = int(str_twos_complement, 2)
                comp2[i, j] = twos_complement
        return comp2

    def readCFAImages(self, add="../Data/image.bmp"):
        bayer = cv2.imread(add)
        bayer = np.sum(bayer, axis=2)
        return bayer
