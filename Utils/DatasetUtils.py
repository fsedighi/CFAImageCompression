import tensorflow_datasets as tfds
import numpy as np
import cv2

from Utils.CFAGeneratorUtils import RGB2CFAUtils


class DatasetUtils:

    def loadFoodDataset(self):
        food101 = tfds.builder("cifar10")
        food101.download_and_prepare()
        datasets = food101.as_dataset()
        train_dataset = datasets['train']

        rgbImages = []

        for ex in train_dataset:
            image = tfds.as_numpy(ex['image'])
            w, h, c = image.shape
            image = cv2.resize(image, (256, 256))
            if len(rgbImages) < 1000:
                rgbImages.append(image)
            else:
                break

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
