import math
import os
import sys

import cv2
import glymur as glymur
import lz4.frame
import numpy as np
from PIL import Image

from CharPyLS import jpeg_ls


class Evaluation:

    def __init__(self) -> None:
        super().__init__()
        self.jpeg200Name = "myfile.jp2"
        self.jpeg200NameLossy = "myfileLossy2000.jp2"
        self.jpegLossy = "myfileJpegLossy.jpeg"
        self.lzw = "lzw.lz"

    def calculate_psnr(self, img1, img2):
        # img1 and img2 have range [0, 255]
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def applyLZWCompressionOnImage(self, image):
        compressedImage = lz4.frame.compress(image)
        originalSize = sys.getsizeof(image)
        compressedSize = sys.getsizeof(compressedImage)
        compressionRatio = originalSize / compressedSize
        return compressedImage, compressionRatio, compressedSize

    def decompress(self, shapes):
        image_bytes = lz4.frame.decompress(self.lzw)
        return np.reshape(np.frombuffer(image_bytes, np.uint8), shapes)

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

    def compressionRatio(self, data, imageName, verbose):
        # compressions
        glymur.Jp2k(self.jpeg200Name, data=data, cratios=[1])
        glymur.Jp2k(self.jpeg200NameLossy, data=data, cratios=[5])

        # array_buffer = data.tobytes()

        # cv2.imwrite(self.jpegLossy, data)
        # imPillow = Image.new("I", data.T.shape)
        # imPillow.frombytes(array_buffer, 'raw', "I;16")
        # imPillow.save(self.jpegLossy, "JPEG", quality=90)

        jpeg_lsBuffer = jpeg_ls.encode(data)
        compressedImage, compressionRatioLZW, compressedSize = self.applyLZWCompressionOnImage(data)
        # Sizes.
        originalSize = len(data.tostring()) / 1024
        jpeg2000Size = os.stat(self.jpeg200Name).st_size / 1024
        jpeg200NameLossySize = os.stat(self.jpeg200NameLossy).st_size / 1024
        # jpegLossySize = os.stat(self.jpegLossy).st_size / 1024
        jpegLsSize = len(jpeg_lsBuffer) / 1024
        jpeg2000CompressionRatio = originalSize / jpeg2000Size
        # JpegLossyCompressionRatio = originalSize / jpegLossySize
        JpegLsCompressionRatio = originalSize / jpegLsSize
        Jpeg2000LossyCompressionRatio = originalSize / jpeg200NameLossySize
        if verbose:
            print('Size of uncompressed {0}: {1} KB'.format(imageName, originalSize))
            print('compression ratio of JPEG-2000 Lossless encoded {0}: {1}'.format(imageName, jpeg2000CompressionRatio))
            print('compression ratio of JPEG-2000 Lossy encoded {0}: {1}'.format(imageName, Jpeg2000LossyCompressionRatio))
            # print('compression ratio of JPEG-lossy encoded {0}: {1}'.format(imageName, JpegLossyCompressionRatio))
            print('compression ratio of LZW encoded {0}: {1}'.format(imageName, compressionRatioLZW))
            print('compression ratio of JPEG-LS encoded {0}: {1}'.format(imageName, JpegLsCompressionRatio))

        return jpeg2000CompressionRatio, JpegLsCompressionRatio, compressionRatioLZW

    def evaluate(self, filteredData, originalData, inverseFilterFunction=None, verbose=True, precisionFloatingPoint=0):
        if precisionFloatingPoint == 0:
            filteredData = np.abs(np.round(filteredData)).astype('uint8')
            originalData = np.abs(np.round(originalData)).astype('uint8')
        else:
            filteredData = np.abs(np.round(filteredData)).astype('uint16')
            originalData = np.abs(np.round(originalData)).astype('uint16')

        jpeg2000CompressionRatioBefore, JpegLsCompressionRatio, compressionRatioLZWBefore = self.compressionRatio(originalData, "Before", verbose)
        if verbose:
            print("**************************************************")
        jpeg2000CompressionRatioAfter, JpegLsCompressionRatio, compressionRatioLZWAfter = self.compressionRatio(filteredData, "After", verbose)

        # Decompress.
        psnr = None
        ssim = None
        jp2Decoded = glymur.Jp2k(self.jpeg200Name).read()
        if inverseFilterFunction is not None:
            retrivedData = inverseFilterFunction(jp2Decoded)
            retrivedData = np.abs(np.round(retrivedData)).astype('uint8')
            ssim = self.calculate_ssim(retrivedData, originalData)
            psnr = self.calculate_psnr(retrivedData, originalData)
            if verbose:
                print("**************Quality******************")
                print("JPEG 2000 : PSNR= {0};  SSIM={1}".format(psnr, ssim))

        return psnr, ssim, jpeg2000CompressionRatioAfter, jpeg2000CompressionRatioBefore
