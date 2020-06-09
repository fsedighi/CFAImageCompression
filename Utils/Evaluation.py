import math
import os

import cv2
import glymur as glymur
import numpy as np
from PIL import Image


class Evaluation:

    def __init__(self) -> None:
        super().__init__()
        self.jpeg200Name = "myfile.jp2"
        self.jpegLossy = "myfile.jpeg"

    def calculate_psnr(self, img1, img2):
        # img1 and img2 have range [0, 255]
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def ssim(self, img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def calculate_ssim(self, img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return self.ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self.ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self.ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')

    def compressionRatio(self, data, imageName):
        jp2 = glymur.Jp2k(self.jpeg200Name, data=data, cratios=[1])
        imPillow = Image.fromarray(data)
        imPillow.save(self.jpegLossy, "JPEG", quality=90)
        # Sizes.
        originalSize = len(data.tostring()) / 1024
        jpeg2000Size = os.stat(self.jpeg200Name).st_size / 1024
        jpegLossySize = os.stat(self.jpegLossy).st_size / 1024
        print('Size of uncompressed {0}: {1} KB'.format(imageName, originalSize))
        jpeg2000CompressionRatio = originalSize / jpeg2000Size
        print('compression ratio of JPEG-2000 encoded {0}: {1}'.format(imageName, jpeg2000CompressionRatio))
        JpegLossyCompressionRatio = originalSize / jpegLossySize
        print('compression ratio of JPEG-lossy encoded {0}: {1}'.format(imageName, JpegLossyCompressionRatio))

        return jpeg2000CompressionRatio, JpegLossyCompressionRatio

    def evaluate(self, filteredData, originalData, inverseFilterFunction=None):
        filteredData = filteredData.astype('uint8')
        originalData = np.abs(originalData).astype('uint8')

        jpeg2000CompressionRatioBefore, JpegLossyCompressionRatioBefore = self.compressionRatio(originalData, "Before")
        jpeg2000CompressionRatioAfter, JpegLossyCompressionRatioAfter = self.compressionRatio(filteredData, "After")

        # Decompress.
        jp2Decoded = glymur.Jp2k(self.jpeg200Name).read()
        if inverseFilterFunction is not None:
            retrivedData = inverseFilterFunction(jp2Decoded)
            retrivedData = np.abs(retrivedData).astype('uint8')
            ssim = self.calculate_ssim(retrivedData, originalData)
            psnr = self.calculate_psnr(retrivedData, originalData)
            print("JPEG 2000 : PSNR= {0};  SSIM={1}".format(psnr, ssim))

        # Compare image data, before and after.
        is_same = (filteredData == jp2Decoded).all()
        print('\nRestored data is identical to original? {:s}\n'.format(str(is_same)))
