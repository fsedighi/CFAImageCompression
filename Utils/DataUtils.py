import io
import os

import numpy as np
import cv2
import pandas as pd
import rawpy
import requests

from Utils.CFAGeneratorUtils import RGB2CFAUtils


class DataUtils:

    def loadKodakDataset(self):
        # download from https://www.kaggle.com/sherylmehta/kodak-dataset
        root = "../Data/kodak/"
        files = os.listdir(root)
        rgbImages = []
        for file in files:
            img = cv2.imread(root + file)
            # img = cv2.resize(img, (256, 256))
            rgbImages.append(img)

        rgbImages = np.asarray(rgbImages)
        return rgbImages

    def loadNikonDataset(self, type="D40"):
        # download from https://www.kaggle.com/sherylmehta/kodak-dataset
        root = "../Data/Nikon {}/".format(type)
        files = os.listdir(root)
        rgbImages = []
        for file in files:
            img = cv2.imread(root + file, -1)
            img = cv2.resize(img, (256, 256))
            rgbImages.append(img)

        rgbImages = np.asarray(rgbImages)
        return rgbImages

    def loadOtherDataset(self, type="Arri"):
        # download from https://www.kaggle.com/sherylmehta/kodak-dataset
        root = "../Data/{}/".format(type)
        files = os.listdir(root)
        rgbImages = []
        for file in files:
            img = cv2.imread(root + file, -1)
            # img = cv2.resize(img, (256, 256))
            rgbImages.append(img)

        rgbImages = np.asarray(rgbImages)
        return rgbImages

    def convertDatasetToCFA(self, rgbImages):
        # Convert to CFA
        rgb2CFAUtils = RGB2CFAUtils()
        n_data = len(rgbImages)
        h, w, c = rgbImages[0].shape

        cfaImages = []
        for i in range(n_data):
            cfaImages.append(rgb2CFAUtils.rgb2CFA(rgbImages[i], show=False, res=0 if rgbImages[0].dtype == np.uint8 else 1))
            print("converting image {0} to CFA: Training".format(i))
        cfaImages = np.asarray(cfaImages)
        # cfaImages = np.reshape(cfaImages, [-1, cfaImages.shape[0], cfaImages.shape[1]])
        # cfaImages = cfaImages.astype('uint8')
        return cfaImages, cfaImages.shape

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

    def saveImageNikon(self):
        data = pd.read_csv("../Data/RAISE_127.csv")

        addresses = data["NEF"].values
        Device = data["Device"].values
        for ind, address in enumerate(addresses):
            if Device[ind] not in ["Nikon D90", "Nikon D7000", "Nikon D40"]:
                resp = requests.get(address)
                with rawpy.imread(io.BytesIO(resp.content)) as raw:
                    rgbImage = raw.postprocess()
                    print("read image from {}".format(address))
                    if not os.path.exists("../Data/" + Device[ind]):
                        os.mkdir("../Data/" + Device[ind])
                    cv2.imwrite("../Data/" + Device[ind] + "/" + str(ind) + ".tiff", rgbImage)
